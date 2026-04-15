# -*- coding: utf-8 -*-
from __future__ import annotations
"""
BURNUP – Post-frontal combustion model (Albini & Reinhardt).

Python port of the original C++ BURNUP model. Simulates the consumption
of woody fuel particles after a surface fire passes, partitioning mass
loss into flaming and smoldering combustion phases.

Converted from: burnupcw.h / burnupw.cpp
Original authors: Frank Albini & Elizabeth Reinhardt

Port notes vs. C++ original
----------------------------
- NumPy arrays replace fixed-size C arrays; no MAXNO/MAXKL compile-time caps.
- Triangular-index helper ``_loc`` uses pure integer arithmetic.
- ``_heat_exchange`` returns a plain tuple ``(hfm, hbar, en)`` for speed.
- All temperatures accepted in **°C**; converted to K internally.
- File I/O (Stash/Summary) replaced with in-memory results list.
- Regression / adaptive-downsampling code removed (unused in active paths).
- ``dataclass`` used for structured fuel input.

Performance optimisations for batch / multi-worker use
------------------------------------------------------
- Pre-computed ``_kl_map`` lookup table replaces all per-iteration ``_loc()``
  function calls.  Built once before the time loop and used by direct
  indexing throughout (eliminates ~tens-of-thousands of Python function calls
  per simulation).
- ``_heat_exchange`` returns a plain tuple instead of allocating a
  ``_HeatResult`` object on every call (~hundreds of allocations per step
  eliminated).  Accepts pre-computed effective velocity ``v_eff`` instead of
  recomputing ``sqrt(u² + 0.53·g·d)`` on every call.
- ``_temp_fire`` results are cached per-step via a local dict keyed on
  ``(gi, r_val)``; duplicate calls with the same intensity/mixing
  parameters (common for ``l == 0`` pairs) are served from cache.
- ``work`` array computed via vectorised NumPy expression instead of a
  Python ``for`` loop.
- ``wo.sum()`` used for remaining-weight calculation in ``_record()``
  instead of a nested Python loop.
- Per-component ``wodot`` summation in ``_fire_intensity()`` uses
  pre-built index slices from ``_kl_map``.
- Per-component ``comp_done`` flag skips the inner ``l`` loop entirely for
  fuel classes that are fully burned out on all pairs.
- ``_ff_ignite`` body inlined into ``_t_ignite`` to avoid per-iteration
  function-call overhead in the binary search hot path.
- ``_func_dry`` body inlined into ``_dry_time`` for the same reason.
- Module-level ``_sqrt``, ``_exp``, ``_fabs`` aliases avoid repeated
  ``math.*`` attribute lookups in tight loops.
- Per-component fuel properties hoisted to local variables before the inner
  ``l`` loop (avoid repeated NumPy scalar indexing).
- Validation in ``_check_fire`` uses direct ``if`` statements instead of
  building and iterating a list of tuples.

@author: Gregory A. Greene (port)
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

# ---------------------------------------------------------------------------
# Fast math aliases  (avoid repeated attribute lookup in tight loops)
# ---------------------------------------------------------------------------
_sqrt = math.sqrt
_exp = math.exp
_fabs = math.fabs

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_CH2O: float = 4186.0          # Specific heat of water (J/kg·K)
_TPDRY: float = 353.0          # Temperature at onset of drying (K, ≈80 °C)
_SMALLX: float = 1.0e-08       # Near-zero threshold (C++ e_small = 1e-8)
_BIG: float = 1.0e+06          # Near-infinity threshold
_RINDEF: float = 1.0e+30       # "Infinite time" sentinel
_MXSTEP: int = 20              # Rolling history window for qdot averaging


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FuelParticle:
    """Physical properties for a single fuel size-class.

    All temperatures are in **°C** (converted to K internally).

    :param wdry: Oven-dry mass loading (kg/m²).
    :param htval: Low heat of combustion (J/kg).
    :param fmois: Moisture content, fraction of dry weight.
    :param dendry: Oven-dry mass density (kg/m³).
    :param sigma: Surface-to-volume ratio (1/m).
    :param cheat: Specific heat capacity (J/kg·K).
    :param condry: Oven-dry thermal conductivity (W/m·K).
    :param tpig: Piloted-ignition temperature (°C).
    :param tchar: End-of-pyrolysis (char) temperature (°C).
    :param ash: Mineral ash content, mass fraction.
    """
    wdry: float
    htval: float
    fmois: float
    dendry: float
    sigma: float
    cheat: float = 2750.0
    condry: float = 0.133
    tpig: float = 300.0    # °C  (typical wood ignition ~300 °C)
    tchar: float = 350.0   # °C  (typical wood char ~350 °C)
    ash: float = 0.05


@dataclass
class BurnResult:
    """Single time-step output from the BURNUP simulation.

    :param time: Elapsed time since ignition (s).
    :param wdf: Dry-weight remaining fraction (0–1).
    :param ff: Flaming fraction of mass-loss rate (0–1).
    :param comp_flaming: Per-component cumulative flaming mass consumed
        (kg/m²) since the last record, in sorted component order.
        ``None`` when not tracked.
    :param comp_smoldering: Per-component cumulative smoldering mass consumed
        (kg/m²) since the last record, in sorted component order.  Index
        ``number`` holds the duff smoldering rate.  ``None`` when not tracked.
    """
    time: float
    wdf: float
    ff: float
    comp_flaming: Optional[List[float]] = None
    comp_smoldering: Optional[List[float]] = None


@dataclass
class BurnSummaryRow:
    """Per-component summary emitted at end of simulation.

    :param component: 1-based fuel component index.
    :param wdry: Oven-dry loading (kg/m²).
    :param fmois: Moisture fraction.
    :param diam: Approximate diameter (m).
    :param t_ignite: Earliest ignition time across all pairs (s).
    :param t_burnout: Latest burnout time across all pairs (s).
    :param remaining: Remaining dry loading (kg/m²).
    :param frac_remaining: Remaining fraction of original loading.
    """
    component: int
    wdry: float
    fmois: float
    diam: float
    t_ignite: float
    t_burnout: float
    remaining: float
    frac_remaining: float


# ---------------------------------------------------------------------------
# Validation bounds  (all exclusive – matching C++ strict-inequality checks)
# ---------------------------------------------------------------------------
# Sources: BUR_BRN.H defines (e_tig1/2, e_tch1/2, e_cht1/2, e_fms1/2,
#          e_dfm1/2, e_small, e_big) and BRN_CheckData() local constants
#          (ash1/2, htv1/2, den1/2, sig1/2, con1/2, fir1/2, ti1/2,
#           u1/2, d1/2, tam1/2) in BUR_BRN.cpp lines 1052-1061.
#
# Note: tpig/tchar are stored internally in Kelvin but validated here in °C
#       (the input unit), matching the °C bounds from BUR_BRN.H (e_tig1=200,
#       e_tig2=400, e_tch1=250, e_tch2=500) rather than the Kelvin-offset
#       equivalents used inside BRN_CheckData.
_FUEL_BOUNDS = {
    'wdry':   (_SMALLX, _BIG,     'dry loading (kg/m²)'),          # e_small=1e-8, e_big=1e6
    'ash':    (0.0001,  0.1,      'ash content (fraction)'),        # ash1=0.0001,  ash2=0.1
    'htval':  (1.0e7,   3.0e7,    'heat content (J/kg)'),           # htv1=1e7,     htv2=3e7
    'fmois':  (0.01,    3.0,      'fuel moisture (fraction)'),      # e_fms1=0.01,  e_fms2=3.0
    'dendry': (200.0,   1000.0,   'dry mass density (kg/m³)'),      # den1=200,     den2=1000
    'sigma':  (4.0,     1.0e4,    'SAV (1/m)'),                     # sig1=4,       sig2=1e4
    'cheat':  (1000.0,  3000.0,   'heat capacity (J/kg·K)'),        # e_cht1=1000,  e_cht2=3000
    'condry': (0.025,   0.25,     'thermal conductivity (W/m·K)'),  # con1=0.025,   con2=0.25
    'tpig':   (200.0,   400.0,    'ignition temperature (°C)'),     # e_tig1=200,   e_tig2=400
    'tchar':  (250.0,   500.0,    'char temperature (°C)'),         # e_tch1=250,   e_tch2=500
}

_FIRE_BOUNDS = {
    'fistart': (10.0,   1.0e5,   'igniting fire intensity (kW/m²)'),        # fir1=40,  fir2=1e5
    'ti':      (10.0,   200.0,   'surface fire residence time (s)'),        # ti1=10,   ti2=200
    'u':       (0.0,    5.0,     'windspeed at fuelbed top (m/s)'),         # u1=0,     u2=5
    'd':       (0.1,    5.0,     'fuel bed depth (m)'),                     # d1=0.1,   d2=5
    'tamb_c':  (-40.0,  40.0,    'ambient temperature (°C)'),               # tam1=-40, tam2=40
    'dfm':     (0.1,    1.972,   'duff moisture (fraction)'),               # e_dfm1=0.1, e_dfm2=1.972
}

# ---------------------------------------------------------------------------
# Burnup limit adjustment codes
# ---------------------------------------------------------------------------
# When a burnup input variable is clipped to stay within ``_FIRE_BOUNDS`` /
# ``_FUEL_BOUNDS``, a single-digit numeric code is recorded.  If multiple
# variables are clipped for the same cell, the codes are concatenated (e.g.
# codes 1 and 3 → adjustment value 13; codes 2, 4 and 6 → 246).
#
# A value of 0 means no adjustment was applied.
_BURNUP_LIMIT_ADJUST = {
    1: 'fistart – igniting fire intensity clipped to max (1e5 kW/m²)',
    2: 'ti – surface fire residence time clipped to max (200 s)',
    3: 'u – windspeed at fuelbed top clipped to max (5 m/s)',
    4: 'd – fuel bed depth clipped to min (0.1 m) or max (5 m)',
    5: 'tamb_c – ambient temperature clipped to max (40 °C)',
    6: 'dfm – duff moisture (fraction) clipped to min (0.1)',
}

# ---------------------------------------------------------------------------
# Burnup limit error codes
# ---------------------------------------------------------------------------
# Two-digit error codes recorded when burnup fails to run because a fuel
# or fire-environment parameter falls outside the physical bounds that
# cannot be safely clipped.  Unlike ``_BURNUP_LIMIT_ADJUST`` (which
# silently clips recoverable values), these represent hard failures where
# the burnup model returns ``None`` and falls back to simplified defaults.
#
# Codes are grouped by source:
#   10–19  fire-environment parameters (_FIRE_BOUNDS / burnup runtime)
#   20–29  fuel-particle parameters (_FUEL_BOUNDS)
#   90–99  runtime / catch-all errors
#
# A value of 0 means burnup ran successfully (no error).
# If multiple errors would apply, only the first detected code is stored.
_BURNUP_LIMIT_ERROR = {
    # -- Fire environment (not clipped) -----------------------------------
    10: 'fistart – igniting fire intensity below min (40 kW/m²)',
    11: 'ti – surface fire residence time below min (10 s)',
    12: 'u – windspeed at fuelbed top below min (0 m/s)',
    13: 'tamb_c – ambient temperature below min (−40 °C)',
    14: 'dfm – duff moisture (fraction) above max (1.972)',
    15: 'fire cannot dry fuel (fire temperature too low)',
    16: 'no fuel ignited (fire intensity/residence time too low)',
    # -- Fuel particle ----------------------------------------------------
    20: 'wdry – dry loading out of range',
    21: 'ash – ash content out of range',
    22: 'htval – heat content out of range',
    23: 'fmois – fuel moisture (fraction) out of range',
    24: 'dendry – dry mass density out of range',
    25: 'sigma – surface-area-to-volume ratio out of range',
    26: 'cheat – heat capacity out of range',
    27: 'condry – thermal conductivity out of range',
    28: 'tpig – ignition temperature out of range',
    29: 'tchar – char temperature out of range',
    # -- Runtime / other --------------------------------------------------
    90: 'no fuel particles (all loadings ≤ 0)',
    91: 'ntimes ≤ 0',
    99: 'unexpected burnup exception',
}


# ---------------------------------------------------------------------------
# Helper: triangular pair index  (pure-integer, no float division)
# ---------------------------------------------------------------------------
def _loc(k: int, l: int) -> int:
    """Return flat index into the lower-triangular pair array.

    Equivalent to C++ ``k*(k+1)/2 + l - 1`` but uses integer arithmetic
    to avoid any floating-point truncation risk.

    :param k: 1-based size-class index (the larger particle).
    :param l: 0-based interacting-partner index (0 = alone, 1..k = partner class).
    :return: 0-based flat index into the pair arrays.
    """
    return k * (k + 1) // 2 + l - 1


def _maxkl(n: int) -> int:
    """Return total number of interaction-pair slots for *n* fuel classes.

    :param n: Number of fuel classes.
    :return: Array length required to store all (k, l) pair data.
    """
    return n * (n + 1) // 2 + n


def _build_kl_map(number: int) -> Dict[str, object]:
    """Pre-compute all (k, l) → flat-index mappings once.

    Eliminates per-iteration ``_loc()`` function calls throughout the
    simulation by providing direct index lookups.

    :param number: Number of fuel components.
    :return: Dict with keys ``'pairs'``, ``'by_k'``, ``'k0'``,
        ``'all_kl'``, ``'kl_slices'``.
    """
    pairs = []
    by_k: Dict[int, List[Tuple[int, int]]] = {}
    k0_map: Dict[int, int] = {}
    kl_slices: Dict[int, List[int]] = {}
    all_kl = []

    for k in range(1, number + 1):
        k_pairs = []
        k_indices = []
        for l in range(0, k + 1):
            kl = k * (k + 1) // 2 + l - 1
            pairs.append((k, l, kl))
            k_pairs.append((l, kl))
            k_indices.append(kl)
            all_kl.append(kl)
            if l == 0:
                k0_map[k] = kl
        by_k[k] = k_pairs
        kl_slices[k] = k_indices

    return {
        'pairs': pairs,
        'by_k': by_k,
        'k0': k0_map,
        'all_kl': all_kl,
        'kl_slices': kl_slices,
    }


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def _heat_exchange(
    v_eff: float,
    dia: float,
    tf: float,
    ts: float,
    cond: float,
) -> Tuple[float, float, float]:
    """Compute combined convective + radiative heat-transfer coefficients.

    Returns ``(hfm, hbar, en)`` as a plain tuple (no object allocation).

    :param v_eff: Effective velocity ``sqrt(u² + 0.53·g·d)``, pre-computed (m/s).
    :param dia: Particle diameter (m).
    :param tf: Fire environment temperature (K).
    :param ts: Mean surface temperature (K).
    :param cond: Thermal conductivity of fuel (W/m·K).
    :return: ``(hfm, hbar, en)`` – film coeff, effective coeff, Biot number.
    """
    B_COND = 5.75e-05
    SIGMA_SB = 5.67e-08
    HRADF = 0.5

    hfm = 0.0
    if dia > B_COND:
        VIS = 7.5e-05
        A_COND = 8.75e-03
        FMFAC = 0.382
        re = v_eff * dia / VIS
        enuair = 0.344 * re ** 0.56
        conair = A_COND + B_COND * tf
        fac = _sqrt(_fabs(tf - ts) / dia)
        hfmin = FMFAC * _sqrt(fac)
        hfm = enuair * conair / dia
        if hfmin > hfm:
            hfm = hfmin

    hrad = HRADF * SIGMA_SB * (tf + ts) * (tf * tf + ts * ts)
    hbar = hfm + hrad
    en = hbar * dia / cond
    return hfm, hbar, en


def _temp_fire(q: float, r: float, tamb: float) -> float:
    """Compute fire environment temperature via iterative mixing model.

    :param q: Fire intensity (kW/m²).
    :param r: Dimensionless mixing parameter.
    :param tamb: Ambient temperature (K).
    :return: Fire environment temperature (K).
    """
    ERR = 1.0e-04
    AA = 20.0

    term = r / (AA * q)
    rlast = r
    for _ in range(500):
        den = 1.0 + term * (rlast + 1.0) * (rlast * rlast + 1.0)
        rnext = 0.5 * (rlast + 1.0 + r / den)
        if rnext - rlast < ERR and rlast - rnext < ERR:
            return rnext * tamb
        rlast = rnext
    return rlast * tamb


def _t_ignite(
    tpdr: float,
    tpig: float,
    tpfi: float,
    cond: float,
    chtd: float,
    fmof: float,
    dend: float,
    hbar: float,
    tamb: float,
) -> float:
    """Predict time to piloted ignition via binary search.

    The ``_ff_ignite`` auxiliary is inlined to eliminate per-iteration
    function-call overhead.

    :param tpdr: Fuel temperature at start of drying (K).
    :param tpig: Fuel surface ignition temperature (K).
    :param tpfi: Fire environment temperature (K).
    :param cond: Oven-dry thermal conductivity (W/m·K).
    :param chtd: Oven-dry specific heat capacity (J/kg·K).
    :param fmof: Moisture content, dry-weight fraction.
    :param dend: Oven-dry density (kg/m³).
    :param hbar: Effective film heat-transfer coefficient (W/m²·K).
    :param tamb: Ambient temperature (K).
    :return: Predicted time to piloted ignition (s).
    """
    PINV = 2.125534
    HVAP = 2.177e+06
    CPM = 4186.0
    CONC = 4.27e-04

    # Inlined _ff_ignite constants
    A03 = -1.3371565
    A13 = 0.4653628
    A23 = -0.1282064
    b03 = A03 * (tpfi - tpig) / (tpfi - tamb)

    # Binary search for root
    xlo = 0.0
    xhi = 1.0
    xav = 0.5
    for _ in range(60):
        xav = 0.5 * (xlo + xhi)
        fav = b03 + xav * (A13 + xav * (A23 + xav))
        if fav < 0.0:
            xlo = xav
        elif fav > 0.0:
            xhi = xav
        else:
            break
        if xhi - xlo < 1.0e-12:
            break

    beta = PINV * (1.0 - xav) / xav
    conw = cond + CONC * dend * fmof
    dtb = tpdr - tamb
    dti = tpig - tamb
    ratio = (HVAP + CPM * dtb) / (chtd * dti)
    rhoc = dend * chtd * (1.0 + fmof * ratio)
    return (beta / hbar) ** 2 * conw * rhoc


def _dry_time(enu: float, theta: float) -> float:
    """Compute dimensionless time to onset of surface drying.

    The ``_func_dry`` auxiliary is inlined to eliminate per-iteration
    function-call overhead.

    :param enu: Biot number (``hbar * dia / k``).
    :param theta: Dimensionless temperature ratio.
    :return: Dimensionless drying time.
    """
    P = 0.47047
    # Inlined constants
    A = 0.7478556
    B = 0.4653628
    C = 0.1282064
    rhs = (1.0 - theta) / A

    xl = 0.0
    xh = 1.0
    xm = 0.5
    for _ in range(15):
        xm = 0.5 * (xl + xh)
        if xm * (B - xm * (C - xm)) - rhs < 0.0:
            xl = xm
        else:
            xh = xm
    x = (1.0 / xm - 1.0) / P
    return (0.5 * x / enu) ** 2


def _duff_burn(wdf_load: float, dfm: float,
               duff_pct_consumed: float = -1.0) -> Tuple[float, float, float]:
    """Compute duff burning intensity, duration, and smoldering rate.

    Mirrors ``DuffBurn()`` in the C++ source (``BUR_BRN.cpp``).  When
    *duff_pct_consumed* is a valid FOFEM-calculated percent (0–100), it is
    used as the consumed fraction ``ff`` exactly as in the C++ code
    (``f_DufConPerCent / 100.0``).  Otherwise the original moisture-only
    fallback formula is used (``ff = 0.837 – 0.426 × dfm``).

    :param wdf_load: Duff dry-weight loading (kg/m²).
    :param dfm: Duff moisture content (fraction).
    :param duff_pct_consumed: Percent of duff consumed as calculated by
        FOFEM's ``DUF_Mngr`` (0–100, whole number).  Pass ``-1`` (default)
        to use the fallback formula (matches running burnup standalone
        without a prior FOFEM duff calculation).
    :return: ``(dfi, tdf, smolder_rate)`` – duff fire intensity (kW/m²),
        duff burn duration (s), and duff smoldering mass rate (kg/m²·s).
    """
    if wdf_load <= 0.0 or dfm >= 1.96:
        return 0.0, 0.0, 0.0
    dfi = 11.25 - 4.05 * dfm
    # C++ DuffBurn: use FOFEM pdc when valid, else moisture-only fallback
    if 0.0 <= duff_pct_consumed <= 100.0:
        ff = duff_pct_consumed / 100.0
    else:
        ff = 0.837 - 0.426 * dfm
    if ff <= 0.0:
        return 0.0, 0.0, 0.0
    tdf = 1.0e4 * ff * wdf_load / (7.5 - 2.7 * dfm)
    smolder_rate = (ff * wdf_load / tdf) if tdf > 0.0 else 0.0
    return dfi, tdf, smolder_rate


# ---------------------------------------------------------------------------
# Sorting & interaction matrix
# ---------------------------------------------------------------------------
def _sort_fuels(
    sigma: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
) -> np.ndarray:
    """Sort fuel classes by increasing size (decreasing SAV), then moisture, then density.

    :param sigma: Surface-to-volume ratios (1/m), shape ``(n,)``.
    :param fmois: Moisture fractions, shape ``(n,)``.
    :param dendry: Oven-dry densities (kg/m³), shape ``(n,)``.
    :return: Integer array of original indices in sorted order, shape ``(n,)``.
    """
    return np.lexsort((dendry, fmois, 1.0 / sigma))


def _overlaps(
    wdry: np.ndarray,
    sigma: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the interaction (overlap) matrix for sorted fuel classes.

    :param wdry: Oven-dry loadings (kg/m²), sorted, shape ``(n,)``.
    :param sigma: SAV (1/m), sorted, shape ``(n,)``.
    :param fmois: Moisture fractions, sorted, shape ``(n,)``.
    :param dendry: Densities (kg/m³), sorted, shape ``(n,)``.
    :return: ``(elam, alone, area)`` – interaction matrix, alone fractions, self-areas.
    """
    n = len(wdry)
    nkl = _maxkl(n)
    pi = math.pi

    xmat = np.zeros(nkl)
    elam = np.zeros((n, n))
    alone = np.zeros(n)
    area = np.zeros(n)

    for k in range(1, n + 1):
        for l in range(1, k + 1):
            ak = 3.25 * _exp(-20.0 * fmois[l - 1] ** 2)
            siga = ak * sigma[k - 1] / pi
            kl = k * (k + 1) // 2 + l - 1  # inlined _loc
            a = siga * wdry[l - 1] / dendry[l - 1]
            if k == l:
                bb = 1.0 - _exp(-a)
                if bb < 1e-30:
                    bb = 1e-30
                area[k - 1] = bb
            else:
                bb = a if a < 1.0 else 1.0
            xmat[kl] = bb

    if n == 1:
        elam[0, 0] = xmat[1]
        alone[0] = 1.0 - elam[0, 0]
        return elam, alone, area

    for k in range(1, n + 1):
        frac = 0.0
        for l in range(1, k + 1):
            frac += xmat[k * (k + 1) // 2 + l - 1]
        if frac > 1.0:
            inv_frac = 1.0 / frac
            for l in range(1, k + 1):
                elam[k - 1, l - 1] = xmat[k * (k + 1) // 2 + l - 1] * inv_frac
            alone[k - 1] = 0.0
        else:
            for l in range(1, k + 1):
                elam[k - 1, l - 1] = xmat[k * (k + 1) // 2 + l - 1]
            alone[k - 1] = 1.0 - frac

    return elam, alone, area


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class BurnupValidationError(ValueError):
    """Raised when input parameters fall outside physically valid ranges."""
    pass


def _check_fuel(particles: Sequence[FuelParticle]) -> None:
    """Validate all fuel-particle parameters against physical bounds.

    :param particles: Sequence of :class:`FuelParticle` objects.
    :raises BurnupValidationError: If any parameter is out of range.
    """
    for i, p in enumerate(particles):
        for attr, (lo, hi, label) in _FUEL_BOUNDS.items():
            val = getattr(p, attr)
            if val <= lo or val >= hi:
                raise BurnupValidationError(
                    f"Fuel class {i}: {label} = {val} out of range ({lo}, {hi})"
                )


def _check_fire(
    fistart: float,
    ti: float,
    u: float,
    d: float,
    tamb_c: float,
    wdf_load: float,
    dfm: float,
) -> Tuple[float, float]:
    """Validate fire-environment parameters against C++ ``BRN_CheckData`` bounds.

    Bounds are taken directly from ``BRN_CheckData()`` local constants in
    ``BUR_BRN.cpp`` (lines 1057–1061):

    * ``fir1 = 40.0``,  ``fir2 = 1.0e5``  – igniting fire intensity (kW/m²)
    * ``ti1  = 10.0``,  ``ti2  = 200.0``  – flame residence time (s)
    * ``u1   = 0.0``,   ``u2   = 5.0``    – windspeed at fuelbed top (m/s)
    * ``d1   = 0.1``,   ``d2   = 5.0``    – fuel bed depth (m)
    * ``tam1 = -40.0``, ``tam2 = 40.0``   – ambient temperature (°C)
    * ``e_dfm1 = 0.1``, ``e_dfm2 = 1.972``– duff moisture fraction

    No auto-adjustment is applied; the C++ code performs straight range checks
    and returns an error string when any value is out of bounds.

    :param fistart: Igniting fire intensity (kW/m²).
    :param ti: Surface fire residence time (s).
    :param u: Windspeed at fuelbed top (m/s).
    :param d: Fuel bed depth (m).
    :param tamb_c: Ambient temperature (°C).
    :param wdf_load: Duff dry-weight loading (kg/m²).
    :param dfm: Duff moisture fraction.
    :return: ``(fistart, ti)`` unchanged.
    :raises BurnupValidationError: If any parameter is out of range.
    """

    if fistart < _FIRE_BOUNDS['fistart'][0] or fistart > _FIRE_BOUNDS['fistart'][1]:
        raise BurnupValidationError(
            f"igniting fire intensity = {fistart} out of range "
            f"({_FIRE_BOUNDS['fistart'][0]}, {_FIRE_BOUNDS['fistart'][1]}) kW/m²")
    if ti < _FIRE_BOUNDS['ti'][0] or ti > _FIRE_BOUNDS['ti'][1]:
        raise BurnupValidationError(
            f"surface fire residence time = {ti} out of range (10, 200) s")
    if u < _FIRE_BOUNDS['u'][0] or u > _FIRE_BOUNDS['u'][1]:
        raise BurnupValidationError(
            f"windspeed = {u} out of range (0, 5) m/s")
    if d < _FIRE_BOUNDS['d'][0] or d > _FIRE_BOUNDS['d'][1]:
        raise BurnupValidationError(
            f"fuel bed depth = {d} out of range (0.1, 5) m")
    if tamb_c < _FIRE_BOUNDS['tamb_c'][0] or tamb_c > _FIRE_BOUNDS['tamb_c'][1]:
        raise BurnupValidationError(
            f"ambient temperature = {tamb_c} out of range (-40, 40) °C")
    if wdf_load > 0.0 and (dfm < _FIRE_BOUNDS['dfm'][0] or dfm > _FIRE_BOUNDS['dfm'][1]):
        raise BurnupValidationError(
            f"duff moisture = {dfm} out of range (0.1, 1.972)")

    return fistart, ti


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def burnup(
    particles: Sequence[FuelParticle],
    fi: float,
    ti: float,
    u: float,
    d: float,
    tamb: float,
    r0: float,
    dr: float,
    dt: float,
    ntimes: int,
    wdf: float = 0.0,
    dfm: float = 2.0,
    duff_pct_consumed: float = -1.0,
    fint_switch: float = 15.0,
    validate: bool = True,
    hsf_consumed: float = 0.0,
    brafol_consumed: float = 0.0,
) -> Tuple[List[BurnResult], List[BurnSummaryRow]]:
    """Run the complete BURNUP post-frontal combustion simulation.

    This is the primary entry-point, equivalent to the C++ ``Burnup()`` batch
    method.  It sorts fuel classes, computes interaction matrices, initialises
    the drying / ignition / burnout state for every fuel-pair combination,
    then steps forward in time until fire intensity drops below the minimum
    threshold or the iteration limit is reached.

    :param particles: Fuel descriptions — one :class:`FuelParticle` per
        size class.
    :param fi: Igniting fire intensity (kW/m²).
    :param ti: Spreading-fire residence time (s).
    :param u: Mean horizontal windspeed at fuelbed top (m/s).
    :param d: Fuel bed depth (m).
    :param tamb: Ambient temperature (**°C**).
    :param r0: Minimum mixing parameter (dimensionless).
    :param dr: Range (max − min) of mixing parameter.
    :param dt: Simulation time-step (s).
    :param ntimes: Maximum number of time steps.
    :param wdf: Duff dry-weight loading (kg/m²). Default 0 (no duff).
    :param dfm: Duff moisture content, fraction. Default 2.0 (suppresses duff).
    :param duff_pct_consumed: Percent of duff consumed as pre-calculated by
        FOFEM's ``DUF_Mngr`` (0–100, whole number).  Passed to
        :func:`_duff_burn` to set the consumed fraction ``ff``, matching
        C++ ``BRN_Run`` / ``DuffBurn``.  Pass ``-1`` (default) for the
        moisture-only fallback (standalone burnup runs without FOFEM).
    :param fint_switch: Flaming / smoldering intensity threshold (kW/m²).
        Default 15.
    :param validate: If ``True`` (default), run range-checks on all inputs.
    :param hsf_consumed: Total herb + shrub consumed (kg/m²).  In C++,
        ``BRN_Run`` receives herb and shrub consumed amounts and distributes
        them at a linear rate (default 10 T/ac/min) adding fire intensity
        at each timestep via ``BRN_Intensity()``.  Default 0 (no HSF
        contribution — standalone burnup without FOFEM pipeline).
    :param brafol_consumed: Total branch + foliage consumed (kg/m²).  In
        C++, this is consumed entirely in the first timestep and adds fire
        intensity via ``BRN_Intensity()``.  Default 0.
    :return: ``(results, summary)`` where *results* is a list of
        :class:`BurnResult` (one per completed timestep) and *summary* is
        a list of :class:`BurnSummaryRow` (one per fuel component).
    :raises BurnupValidationError: If *validate* is True and any parameter
        is out of range.
    """
    if ntimes <= 0:
        raise BurnupValidationError("ntimes must be > 0")
    if len(particles) == 0:
        raise BurnupValidationError("at least one fuel particle is required")

    # ------------------------------------------------------------------
    # 1. Validate
    # ------------------------------------------------------------------
    fistart = fi
    tamb_k = tamb + 273.0

    if validate:
        _check_fuel(particles)

    number = len(particles)
    nkl = _maxkl(number)

    # ------------------------------------------------------------------
    # 2. Unpack fuel particles into per-component arrays  (°C → K)
    # ------------------------------------------------------------------
    wdry  = np.array([p.wdry   for p in particles], dtype=np.float64)
    ash   = np.array([p.ash    for p in particles], dtype=np.float64)
    htval = np.array([p.htval  for p in particles], dtype=np.float64)
    fmois = np.array([p.fmois  for p in particles], dtype=np.float64)
    dendry= np.array([p.dendry for p in particles], dtype=np.float64)
    sigma = np.array([p.sigma  for p in particles], dtype=np.float64)
    cheat = np.array([p.cheat  for p in particles], dtype=np.float64)
    condry= np.array([p.condry for p in particles], dtype=np.float64)
    tpig  = np.array([p.tpig + 273.0 for p in particles], dtype=np.float64)
    tchar = np.array([p.tchar + 273.0 for p in particles], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Sort by increasing size (decreasing sigma)
    # ------------------------------------------------------------------
    key = _sort_fuels(sigma, fmois, dendry)
    inv_key = np.argsort(key)   # inverse permutation – restores original input order
    wdry   = wdry[key]
    ash    = ash[key]
    htval  = htval[key]
    fmois  = fmois[key]
    dendry = dendry[key]
    sigma  = sigma[key]
    cheat  = cheat[key]
    condry = condry[key]
    tpig   = tpig[key]
    tchar  = tchar[key]

    # ------------------------------------------------------------------
    # 4. Compute interaction matrix (OverLaps)
    # ------------------------------------------------------------------
    elam, alone, area = _overlaps(wdry, sigma, fmois, dendry)

    # ------------------------------------------------------------------
    # 4b. Pre-compute pair-index lookup table  (performance)
    # ------------------------------------------------------------------
    kl_map = _build_kl_map(number)
    by_k = kl_map['by_k']
    k0_map = kl_map['k0']
    kl_slices = kl_map['kl_slices']

    # ------------------------------------------------------------------
    # 5. Build pair-level arrays
    # ------------------------------------------------------------------
    diam_arr = np.zeros(nkl)
    xmat     = np.zeros(nkl)
    wo       = np.zeros(nkl)

    for k in range(1, number + 1):
        diak = 4.0 / sigma[k - 1]
        wtk  = wdry[k - 1]
        for l, kl in by_k[k]:
            diam_arr[kl] = diak
            xmat[kl] = alone[k - 1] if l == 0 else elam[k - 1, l - 1]
            wo[kl] = wtk * xmat[kl]

    # ------------------------------------------------------------------
    # 6. Emissions accumulators
    # ------------------------------------------------------------------
    smoldering = np.zeros(number + 1)
    flaming    = np.zeros(number)

    # ------------------------------------------------------------------
    # 7. Duff burning
    # ------------------------------------------------------------------
    dfi, tdf, duff_smolder_rate = _duff_burn(wdf, dfm, duff_pct_consumed)
    smoldering[number] = duff_smolder_rate

    # ------------------------------------------------------------------
    # 8. Start — initialise drying/ignition/burnout state
    # ------------------------------------------------------------------
    flit  = np.zeros(number)
    fout  = np.zeros(number)
    alfa  = condry / (dendry * cheat)  # vectorised thermal diffusivity
    fint  = np.zeros(number)

    # Vectorised empirical burn-rate factor
    work = 1.0 / (255.0 * (dendry / 446.0) * 2.01e6 * (1.0 + 1.67 * fmois))

    tout  = np.full(nkl, _RINDEF)
    tign  = np.full(nkl, _RINDEF)
    tdry  = np.full(nkl, _RINDEF)
    tcum  = np.zeros(nkl)
    qcum  = np.zeros(nkl)
    acum  = np.zeros(nkl)
    ddot  = np.zeros(nkl)
    wodot = np.zeros(nkl)
    qdot  = np.zeros((nkl, _MXSTEP))

    # Pre-compute effective velocity (constant for the entire simulation)
    v_eff = _sqrt(u * u + 0.53 * 9.8 * d)

    # ---- first fire-temperature estimate ----
    fi_cur = fistart
    tf = _temp_fire(fi_cur, r0 + 0.25 * dr, tamb_k)

    if tf <= _TPDRY + 10.0:
        raise BurnupValidationError("Igniting fire cannot dry fuel")

    thd = (_TPDRY - tamb_k) / (tf - tamb_k)
    tx  = 0.5 * (tamb_k + _TPDRY)

    # ---- estimate drying start times ----
    for k in range(1, number + 1):
        ki = k - 1
        conwet = condry[ki] + 4.27e-04 * dendry[ki] * fmois[ki]
        cpwet = cheat[ki] + fmois[ki] * _CH2O
        fac_base = dendry[ki] * cpwet / conwet
        for l, kl in by_k[k]:
            dia = diam_arr[kl]
            _, _, en = _heat_exchange(v_eff, dia, tf, tx, conwet)
            tdry[kl] = (0.5 * dia) ** 2 * fac_base * _dry_time(en, thd)

    # ---- determine which components ignite during spreading fire ----
    for k in range(1, number + 1):
        ki = k - 1
        c    = condry[ki]
        tigk = tpig[ki]
        for l, kl in by_k[k]:
            if tdry[kl] >= ti:
                continue
            dia = diam_arr[kl]
            ts  = 0.5 * (_TPDRY + tigk)
            _, hbar, _ = _heat_exchange(v_eff, dia, tf, ts, c)
            tcum[kl] = max((tf - ts) * (ti - tdry[kl]), 0.0)
            qcum[kl] = hbar * tcum[kl]
            if tf <= tigk + 10.0:
                continue
            dtign = _t_ignite(_TPDRY, tpig[ki], tf, condry[ki],
                              cheat[ki], fmois[ki], dendry[ki], hbar, tamb_k)
            trt = tdry[kl] + dtign
            tign[kl] = 0.5 * trt
            if ti > trt:
                flit[ki] += xmat[kl]

    # ---- verify at least one component ignited ----
    if not np.any(flit > 0.0):
        raise BurnupValidationError("No fuel ignited")

    # ---- reset time origin to earliest ignition (vectorised) ----
    trt_min = float(np.min(tign))
    mask = tdry < _RINDEF
    tdry[mask] -= trt_min
    mask = tign < _RINDEF
    tign[mask] -= trt_min

    # ---- establish initial burning rates for ignited components ----
    for k in range(1, number + 1):
        ki = k - 1
        if flit[ki] == 0.0:
            for l, kl in by_k[k]:
                ddot[kl] = 0.0
                tout[kl] = _RINDEF
                wodot[kl] = 0.0
        else:
            ts = tchar[ki]
            c  = condry[ki]
            wk = work[ki]
            for l, kl in by_k[k]:
                dia = diam_arr[kl]
                _, hbar, _ = _heat_exchange(v_eff, dia, tf, ts, c)
                qdot[kl, 0] = hbar * max(tf - ts, 0.0)
                ddt_val = ti - tign[kl]
                acum[kl] = (c / hbar) ** 2 * ddt_val
                ddot[kl] = qdot[kl, 0] * wk
                if ddot[kl] > 0.0:
                    tout[kl] = dia / ddot[kl]
                dnext = max(0.0, dia - ddt_val * ddot[kl])
                wnext = wo[kl] * (dnext / dia) ** 2 if dia > 0.0 else 0.0
                wodot[kl] = (wo[kl] - wnext) / ddt_val if ddt_val > 0.0 else 0.0
                diam_arr[kl] = dnext
                wo[kl] = wnext
                if dnext <= 0.0:
                    flit[ki] -= xmat[kl]
                    fout[ki] += xmat[kl]
                    # Keep wodot[kl] non-zero here so _fire_intensity() (called
                    # immediately below) sees the frontal-pass burn rate for
                    # particles that are fully consumed during ignition. The C++
                    # code achieves the same effect via gd_Fudge1/gd_Fudge2.
                    # The time-step loop handles zeroing via "if tnow >= tdun".
                    ddot[kl]  = 0.0

    ncalls = 0
    comp_done = np.zeros(number, dtype=np.bool_)

    # ------------------------------------------------------------------
    # 9. Fire-intensity computation (uses pre-built index slices)
    # ------------------------------------------------------------------
    def _fire_intensity() -> float:
        """Compute site-average fire intensity and partition flaming/smoldering.

        :return: Site-average fire intensity (kW/m²).
        """
        total = 0.0
        for k in range(1, number + 1):
            ki = k - 1
            wdotk = 0.0
            for kl in kl_slices[k]:
                wdotk += wodot[kl]
            term = (1.0 - ash[ki]) * htval[ki] * wdotk * 1.0e-03
            ark = area[ki]
            fint[ki] = (term / ark - term) if ark > _SMALLX else 0.0

            smoldering[ki] = wodot[k0_map[k]]
            wnoduff = wdotk - smoldering[ki]
            test = (wnoduff / wdotk) * fint[ki] if wnoduff > 0.0 else 0.0

            # C++ FireIntensity line 1710: changed from > to >= to ensure litter
            # loads over ~11.4 T/ac go to flaming (matches comment in BUR_BRN.cpp)
            if test >= (fint_switch / ark - fint_switch) if ark > _SMALLX else False:
                flaming[ki] += wnoduff
            else:
                smoldering[ki] += wnoduff
            total += term
        return total

    # ------------------------------------------------------------------
    # 10. BurnStruct helper (uses wo.sum() instead of nested loop)
    # ------------------------------------------------------------------
    wd0 = float(wo.sum())
    if wd0 == 0.0:
        wd0 = 1.0

    results: List[BurnResult] = []

    def _record(time_val: float) -> None:
        """Append a :class:`BurnResult` snapshot.

        :param time_val: Current simulation time (s).
        """
        wdf_val = float(wo.sum()) / wd0
        wt_flam = float(flaming.sum())
        wt_smol = float(smoldering.sum())
        # Snapshot per-component masses before zeroing
        cf = flaming[:number].tolist()
        cs = smoldering[:number + 1].tolist()
        flaming[:] = 0.0
        smoldering[:number] = 0.0
        denom = wt_flam + wt_smol
        results.append(BurnResult(
            time=time_val,
            wdf=wdf_val,
            ff=wt_flam / denom if denom > 0.0 else 0.0,
            comp_flaming=cf,
            comp_smoldering=cs,
        ))

    # ------------------------------------------------------------------
    # 10b. Herb / shrub / branch / foliage fire-intensity contribution
    # ------------------------------------------------------------------
    # Mirrors C++ HSB_Init / HSB_Get / BRN_Intensity mechanism.
    # Herb+shrub are distributed linearly at a fixed rate (C++ default:
    # 10 T/ac per minute = 10 / 4.4609 / 60 kg/m²/s in SI).
    # Branch+foliage is consumed entirely in the first timestep.
    # Fire intensity (kW/m²) = htval (J/kg) × consumed_rate (kg/m²/s) × 1e-3.
    # We use 1.86e7 J/kg as the heat content, matching C++ e_htval*1000.
    _HSF_HTVAL = 1.86e7  # J/kg — same heat content used for all FOFEM fuels
    _HSF_RATE_TPAC_PER_MIN = 10.0  # C++ default: 10 tons/acre/minute
    _HSF_RATE_SI = _HSF_RATE_TPAC_PER_MIN / 4.4609 / 60.0  # kg/m²/s
    _hsf_remaining = float(hsf_consumed)
    _brafol_remaining = float(brafol_consumed)

    def _hsf_fi(step_seconds: float) -> float:
        """Compute and consume herb+shrub fire intensity for this timestep.

        Returns intensity contribution in kW/m².
        """
        nonlocal _hsf_remaining
        amount = min(_HSF_RATE_SI * step_seconds, _hsf_remaining)
        _hsf_remaining -= amount
        if amount <= 0.0 or step_seconds <= 0.0:
            return 0.0
        return _HSF_HTVAL * amount / step_seconds * 1.0e-3

    def _brafol_fi(step_seconds: float) -> float:
        """Consume all branch+foliage fire intensity in one call.

        Returns intensity contribution in kW/m².
        """
        nonlocal _brafol_remaining
        amount = _brafol_remaining
        _brafol_remaining = 0.0
        if amount <= 0.0 or step_seconds <= 0.0:
            return 0.0
        return _HSF_HTVAL * amount / step_seconds * 1.0e-3

    # ------------------------------------------------------------------
    # 11. Time-step loop
    # ------------------------------------------------------------------
    fi_cur = _fire_intensity()

    # First-timestep HSF/brafol intensity (consumed over ti seconds)
    fi_cur += _hsf_fi(ti) + _brafol_fi(ti)

    _record(ti)

    fimin = 0.1
    tis = ti
    half_dr = 0.5 * dr
    r0_half_dr = r0 + half_dr

    if fi_cur > fimin:
        while True:
            ncalls += 1
            tnow  = tis
            tnext = tnow + dt
            tifi  = tnow - (ncalls - 1) * dt

            # Per-step cache for _temp_fire
            tf_cache: Dict[Tuple[float, float], float] = {}
            fid_cur = dfi if tis < tdf else 0.0

            for k in range(1, number + 1):
                ki = k - 1
                if comp_done[ki]:
                    continue

                # Hoist per-component properties to locals
                c = condry[ki]
                wk = work[ki]
                alfa_k = alfa[ki]
                tchar_k = tchar[ki]
                tpig_k = tpig[ki]
                condry_k = c
                cheat_k = cheat[ki]
                fmois_k = fmois[ki]
                dendry_k = dendry[ki]

                all_out = True

                for l, kl in by_k[k]:
                    tdun = tout[kl]

                    # --- burned out ---
                    if tnow >= tdun:
                        ddot[kl]  = 0.0
                        wodot[kl] = 0.0
                        continue

                    all_out = False

                    # --- will burn out this step ---
                    if tnext >= tdun:
                        tgo = tdun - tnow
                        if tgo > 0.0:
                            ddot[kl]  = diam_arr[kl] / tgo
                            wodot[kl] = wo[kl] / tgo
                        else:
                            ddot[kl] = 0.0
                            wodot[kl] = 0.0
                        wo[kl]       = 0.0
                        diam_arr[kl] = 0.0
                        continue

                    tlit = tign[kl]

                    # --- ignited & burning ---
                    if tnow >= tlit:
                        ts = tchar_k
                        if l == 0:
                            r_val = float(r0_half_dr)
                            gi    = float(fi_cur + fid_cur)
                        elif l == k:
                            r_val = float(r0 + half_dr * (1.0 + flit[ki]))
                            gi    = float(fi_cur + flit[ki] * fint[ki])
                        else:
                            li = l - 1
                            r_val = float(r0 + half_dr * (1.0 + flit[li]))
                            gi    = float(fi_cur + fint[ki] + flit[li] * fint[li])

                        cache_key = cast(Tuple[float, float], (gi, r_val))
                        tf_loc = tf_cache.get(cache_key)
                        if tf_loc is None:
                            tf_loc = _temp_fire(gi, r_val, tamb_k)
                            tf_cache[cache_key] = tf_loc

                        dia    = diam_arr[kl]
                        _, hbar, _ = _heat_exchange(v_eff, dia, tf_loc, ts, c)
                        qqq    = hbar * (tf_loc - ts) if tf_loc > ts else 0.0
                        tst    = tlit if tlit > tifi else tifi
                        nspan  = max(1, round((tnext - tst) / dt))

                        if nspan <= _MXSTEP:
                            qdot[kl, nspan - 1] = qqq
                        else:
                            qdot[kl, :-1] = qdot[kl, 1:]
                            qdot[kl, _MXSTEP - 1] = qqq

                        acum[kl] += (c / hbar) ** 2 * dt
                        tav1 = tnext - tlit
                        tav2 = acum[kl] / alfa_k
                        dia4 = dia * 0.25
                        tav3 = dia4 * dia4 / alfa_k
                        tavg = tav1
                        if tav2 < tavg:
                            tavg = tav2
                        if tav3 < tavg:
                            tavg = tav3

                        idx = nspan if nspan < _MXSTEP else _MXSTEP
                        qdsum  = 0.0
                        tspan  = 0.0
                        deltim = dt
                        while True:
                            idx -= 1
                            if idx == 0:
                                deltim = tnext - tspan - tlit
                            if tspan + deltim >= tavg:
                                deltim = tavg - tspan
                            qdsum += qdot[kl, idx] * deltim
                            tspan += deltim
                            if tspan >= tavg or idx <= 0:
                                break

                        qdavg = qdsum / tspan if tspan > 0.0 else 0.0
                        if qdavg < 0.0:
                            qdavg = 0.0
                        ddot[kl] = qdavg * wk
                        dnext = dia - dt * ddot[kl]
                        if dnext < 0.0:
                            dnext = 0.0
                        wnext = wo[kl] * (dnext / dia) ** 2 if dia > 0.0 else 0.0

                        if dnext == 0.0 and ddot[kl] > 0.0:
                            tout[kl] = tnow + dia / ddot[kl]
                        elif 0.0 < dnext < dia:
                            tout[kl] = tnow + dia / (dia - dnext) * dt

                        if qdavg <= _MXSTEP:
                            tout[kl] = 0.5 * (tnow + tnext)

                        ddt_val = tout[kl] - tnow
                        if ddt_val > dt:
                            ddt_val = dt
                        wodot[kl] = (wo[kl] - wnext) / ddt_val if ddt_val > 0.0 else 0.0
                        diam_arr[kl] = dnext
                        wo[kl] = wnext
                        continue

                    # --- drying stage ---
                    dryt_val = tdry[kl]
                    if tnow >= dryt_val and tnow < tlit:
                        if l == 0:
                            r_val2 = float(r0)
                            gi2 = float(fi_cur + fid_cur)
                        elif l == k:
                            r_val2 = float(r0)
                            gi2 = float(fi_cur)
                        else:
                            li = l - 1
                            r_val2 = float(r0 + half_dr * flit[li])
                            gi2 = float(fi_cur + flit[li] * fint[li])

                        cache_key = cast(Tuple[float, float], (gi2, r_val2))
                        tf_loc = tf_cache.get(cache_key)
                        if tf_loc is None:
                            tf_loc = _temp_fire(gi2, r_val2, tamb_k)
                            tf_cache[cache_key] = tf_loc

                        ts     = tamb_k
                        dia    = diam_arr[kl]
                        _, hbar, _ = _heat_exchange(v_eff, dia, tf_loc, ts, c)
                        dtemp  = tf_loc - ts
                        if dtemp < 0.0:
                            dtemp = 0.0
                        qcum[kl] += hbar * dtemp * dt
                        tcum[kl] += dtemp * dt
                        span = tnext - dryt_val
                        dteff  = tcum[kl] / span if span > 0.0 else 0.0
                        heff   = qcum[kl] / tcum[kl] if tcum[kl] > 0.0 else 0.0
                        tfe    = ts + dteff
                        dtlite = _RINDEF
                        if tfe > tpig_k + 10.0:
                            dtlite = _t_ignite(_TPDRY, tpig_k, tfe,
                                               condry_k, cheat_k,
                                               fmois_k, dendry_k, heff, tamb_k)
                        tign[kl] = 0.5 * (dryt_val + dtlite)

                        if tnext > tign[kl]:
                            ts2 = tchar_k
                            _, hbar2, _ = _heat_exchange(v_eff, dia, tf_loc, ts2, c)
                            qdot[kl, 0] = hbar2 * max(tf_loc - ts2, 0.0)
                            ddot[kl] = qdot[kl, 0] * wk
                            dnext = dia - (tnext - tign[kl]) * ddot[kl]
                            if dnext < 0.0:
                                dnext = 0.0
                            wnext = wo[kl] * (dnext / dia) ** 2 if dia > 0.0 else 0.0
                            if dnext == 0.0:
                                tout[kl] = tnow + dia / ddot[kl] if ddot[kl] > 0.0 else _RINDEF
                            elif dnext < dia:
                                tout[kl] = tnow + dia / (dia - dnext) * dt
                            if tout[kl] > tnow:
                                ddt_val = tout[kl] - tnow
                                if ddt_val > dt:
                                    ddt_val = dt
                                wodot[kl] = (wo[kl] - wnext) / ddt_val if ddt_val > 0.0 else 0.0
                            else:
                                wodot[kl] = 0.0
                            diam_arr[kl] = dnext
                            wo[kl] = wnext
                        continue

                    # --- pre-drying stage ---
                    if tnow < dryt_val:
                        conwet = condry_k + 4.27e-04 * fmois_k * dendry_k
                        gi3: float
                        r_val3: float
                        if l == 0:
                            r_val3 = float(r0)
                            gi3 = float(fi_cur + fid_cur)
                        elif l == k:
                            r_val3 = float(r0)
                            gi3 = float(fi_cur)
                        else:
                            li = l - 1
                            r_val3 = float(r0 + half_dr * flit[li])
                            gi3 = float(fi_cur + flit[li] * fint[li])

                        cache_key = cast(Tuple[float, float], (gi3, r_val3))
                        tf_loc = tf_cache.get(cache_key)
                        if tf_loc is None:
                            tf_loc = _temp_fire(gi3, r_val3, tamb_k)
                            tf_cache[cache_key] = tf_loc

                        if tf_loc <= _TPDRY + 10.0:
                            continue

                        dia = diam_arr[kl]
                        ts  = 0.5 * (tamb_k + _TPDRY)
                        _, hbar, _ = _heat_exchange(v_eff, dia, tf_loc, ts, conwet)
                        dtcum = (tf_loc - ts) * dt
                        if dtcum < 0.0:
                            dtcum = 0.0
                        tcum[kl] += dtcum
                        qcum[kl] += hbar * dtcum
                        he   = qcum[kl] / tcum[kl] if tcum[kl] > 0.0 else 0.0
                        dtef = tcum[kl] / tnext if tnext > 0.0 else 0.0
                        thd_val = (_TPDRY - tamb_k) / dtef if dtef > 0.0 else 1.0
                        if thd_val > 0.9:
                            continue

                        biot = he * dia / conwet if conwet > 0.0 else 0.0
                        dryt_new = _dry_time(biot, thd_val)
                        cpwet = cheat_k + _CH2O * fmois_k
                        tdry[kl] = (0.5 * dia) ** 2 / conwet * cpwet * dendry_k * dryt_new

                        if tdry[kl] < tnext:
                            _, hbar2, _ = _heat_exchange(v_eff, dia, tf_loc, _TPDRY, c)
                            dqdt = hbar2 * (tf_loc - _TPDRY)
                            delt = tnext - tdry[kl]
                            qcum[kl] = dqdt * delt
                            tcum[kl] = (tf_loc - _TPDRY) * delt

                            if tf_loc <= tpig_k + 10.0:
                                continue
                            dtlite = _t_ignite(_TPDRY, tpig_k, tf_loc,
                                               condry_k, cheat_k,
                                               fmois_k, dendry_k, hbar2, tamb_k)
                            tign[kl] = 0.5 * (tdry[kl] + dtlite)
                            if tnext > tign[kl]:
                                qdot[kl, 0] = hbar2 * max(tf_loc - tchar_k, 0.0)

                if all_out:
                    comp_done[ki] = True

            # ---- update ignited / burned-out fractions ----
            for k in range(1, number + 1):
                ki = k - 1
                flit[ki] = 0.0
                fout[ki] = 0.0
                for l, kl in by_k[k]:
                    if tnext >= tign[kl]:
                        if tnext <= tout[kl]:
                            flit[ki] += xmat[kl]
                    if tnext > tout[kl]:
                        fout[ki] += xmat[kl]

            # ---- advance time, recompute, record ----
            tis += dt
            fi_cur = _fire_intensity()
            fi_cur += _hsf_fi(dt)  # herb/shrub fire-intensity contribution
            _record(tis)

            # ---- termination ----
            if fi_cur <= fimin or ncalls >= ntimes:
                break

    # ------------------------------------------------------------------
    # 12. Build summary
    # ------------------------------------------------------------------
    summary: List[BurnSummaryRow] = []
    for m in range(1, number + 1):
        mi = m - 1
        rem = 0.0
        ts_min = _RINDEF
        tf_max = 0.0
        for l, mn in by_k[m]:
            t_ig = tign[mn]
            t_ou = tout[mn]
            if t_ig < ts_min:
                ts_min = t_ig
            if t_ou > tf_max:
                tf_max = t_ou
            rem += wo[mn]
        summary.append(BurnSummaryRow(
            component=m,
            wdry=float(wdry[mi]),
            fmois=float(fmois[mi]),
            diam=4.0 / float(sigma[mi]),
            t_ignite=ts_min,
            t_burnout=tf_max,
            remaining=rem,
            frac_remaining=rem / wdry[mi] if wdry[mi] > 0.0 else 0.0,
        ))

    # ------------------------------------------------------------------
    # 13. Restore original input order
    # ------------------------------------------------------------------
    # burnup() sorted particles internally (by sigma/fmois/dendry), interleaving
    # sound and rotten classes.  summary[] and comp_flaming/comp_smoldering inside
    # each BurnResult are in that sorted order.  Callers (_extract_burnup_consumption
    # etc.) expect them in the original input order so that summary[i] and
    # comp_flaming[i] match class_order[i].  Permute back using inv_key.
    summary = [summary[int(inv_key[i])] for i in range(number)]
    for r in results:
        if r.comp_flaming is not None:
            r.comp_flaming = [r.comp_flaming[int(inv_key[i])] for i in range(number)]
        if r.comp_smoldering is not None:
            orig = r.comp_smoldering
            r.comp_smoldering = (
                [orig[int(inv_key[i])] for i in range(number)] + [orig[number]]
            )

    return results, summary

