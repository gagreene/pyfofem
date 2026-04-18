# -*- coding: utf-8 -*-
from __future__ import annotations
"""
BURNUP Array – Vectorized multi-cell post-frontal combustion model.

Prototype NumPy-vectorized implementation of the Albini & Reinhardt burnup
model that processes **C spatial cells simultaneously**.  All per-cell scalar
loops from ``burnup.py`` are replaced by batched array operations; iterative
solvers use fixed-iteration NumPy broadcasts; branching combustion stages use
boolean masks.

Design constraints
------------------
- All cells **must have the same number of fuel particle classes** (``P``).
  This is a reasonable constraint since FOFEM uses a fixed set of 8 standard
  size classes.  Cells with fewer active classes should pad with
  ``wdry = 0`` (zero-load particles are masked out internally).
- A **uniform time-step** ``dt`` is used across all cells (simplifies output
  alignment).  Each cell may still terminate early when its fire intensity
  drops below the threshold — tracked via an ``active`` mask.
- The time loop runs up to ``max_timesteps`` iterations; cells that finish
  early are NaN-padded in the output arrays.
- The ``qdot`` rolling-window average (20-slot history) is fully implemented
  using a fixed-iteration vectorized backward walk matching the scalar
  ``burnup.py`` logic.

Array conventions
-----------------
- ``C`` — number of spatial cells (rows)
- ``P`` — number of fuel particle classes per cell (columns)
- ``NKL`` — triangular pair-interaction slots = ``P*(P+1)//2 + P``
- ``T`` — time-step axis (pre-allocated to ``max_timesteps``)

Input arrays
~~~~~~~~~~~~
- Fire-environment: 1-D ``(C,)`` — ``fi, ti, u, d, tamb, r0, dr, wdf, dfm,
  duff_pct_consumed, fint_switch``.
- Fuel-particle properties: 2-D ``(C, P)`` — ``wdry, htval, fmois, dendry,
  sigma, cheat, condry, tpig, tchar, ash``.

Output
~~~~~~
A :class:`BurnupArrayResult` dataclass with NaN-padded 2-D / 3-D arrays.

@author: Gregory A. Greene (prototype)
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

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
    comp_flaming: Optional[list] = None
    comp_smoldering: Optional[list] = None


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
_FUEL_BOUNDS = {
    'wdry':   (_SMALLX, _BIG,     'dry loading (kg/m²)'),
    'ash':    (0.0001,  0.1,      'ash content (fraction)'),
    'htval':  (1.0e7,   3.0e7,    'heat content (J/kg)'),
    'fmois':  (0.01,    3.0,      'fuel moisture (fraction)'),
    'dendry': (200.0,   1000.0,   'dry mass density (kg/m³)'),
    'sigma':  (4.0,     1.0e4,    'SAV (1/m)'),
    'cheat':  (1000.0,  3000.0,   'heat capacity (J/kg·K)'),
    'condry': (0.025,   0.25,     'thermal conductivity (W/m·K)'),
    'tpig':   (200.0,   400.0,    'ignition temperature (°C)'),
    'tchar':  (250.0,   500.0,    'char temperature (°C)'),
}

_FIRE_BOUNDS = {
    'fistart': (10.0,   1.0e5,   'igniting fire intensity (kW/m²)'),
    'ti':      (10.0,   200.0,   'surface fire residence time (s)'),
    'u':       (0.0,    5.0,     'windspeed at fuelbed top (m/s)'),
    'd':       (0.1,    5.0,     'fuel bed depth (m)'),
    'tamb_c':  (-40.0,  40.0,    'ambient temperature (°C)'),
    'dfm':     (0.1,    1.972,   'duff moisture (fraction)'),
}

# ---------------------------------------------------------------------------
# Burnup limit adjustment codes
# ---------------------------------------------------------------------------
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
_BURNUP_LIMIT_ERROR = {
    10: 'fistart – igniting fire intensity below min (40 kW/m²)',
    11: 'ti – surface fire residence time below min (10 s)',
    12: 'u – windspeed at fuelbed top below min (0 m/s)',
    13: 'tamb_c – ambient temperature below min (−40 °C)',
    14: 'dfm – duff moisture (fraction) above max (1.972)',
    15: 'fire cannot dry fuel (fire temperature too low)',
    16: 'no fuel ignited (fire intensity/residence time too low)',
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
    90: 'no fuel particles (all loadings ≤ 0)',
    91: 'ntimes ≤ 0',
    99: 'unexpected burnup exception',
}


# ---------------------------------------------------------------------------
# Validation exception
# ---------------------------------------------------------------------------
class BurnupValidationError(ValueError):
    """Raised when input parameters fall outside physically valid ranges."""
    pass

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class BurnupArrayResult:
    """Complete output from a vectorized multi-cell burnup simulation.

    All 2-D+ arrays have the cell axis first (``C``).  Time-step arrays are
    NaN-padded beyond each cell's termination point.

    :param time: ``(C, T)`` — elapsed time (s) at each recorded step.
    :param wdf: ``(C, T)`` — dry-weight remaining fraction (0–1).
    :param ff: ``(C, T)`` — flaming fraction of mass-loss rate (0–1).
    :param comp_flaming: ``(C, T, P)`` — per-component flaming mass rate.
    :param comp_smoldering: ``(C, T, P+1)`` — per-component smoldering rate
        (last slot = duff).
    :param n_steps: ``(C,)`` int — valid timestep count per cell.
    :param summary_wdry: ``(C, P)`` — original oven-dry loading per component.
    :param summary_remaining: ``(C, P)`` — remaining dry mass per component.
    :param summary_frac_remaining: ``(C, P)`` — remaining fraction.
    :param summary_t_ignite: ``(C, P)`` — earliest ignition time.
    :param summary_t_burnout: ``(C, P)`` — latest burnout time.
    :param sort_key: ``(C, P)`` int — sorted → original index mapping.
    :param error_code: ``(C,)`` int — 0 = success, >0 = error code from
        :data:`~burnup._BURNUP_LIMIT_ERROR`.
    """
    time: np.ndarray
    wdf: np.ndarray
    ff: np.ndarray
    comp_flaming: np.ndarray
    comp_smoldering: np.ndarray
    n_steps: np.ndarray
    summary_wdry: np.ndarray
    summary_remaining: np.ndarray
    summary_frac_remaining: np.ndarray
    summary_t_ignite: np.ndarray
    summary_t_burnout: np.ndarray
    sort_key: np.ndarray
    error_code: np.ndarray


# ---------------------------------------------------------------------------
# Vectorized physics helpers
# ---------------------------------------------------------------------------

def _temp_fire(q: np.ndarray, r: np.ndarray,
                     tamb: np.ndarray) -> np.ndarray:
    """Vectorized fire environment temperature (fixed 500 iterations).

    :param q: Fire intensity, any broadcastable shape.
    :param r: Mixing parameter, same shape.
    :param tamb: Ambient temperature (K), same shape.
    :return: Fire environment temperature (K), same shape.
    """
    AA = 20.0
    ERR = 1.0e-04

    q_safe = np.where(np.abs(q) < 1e-30, 1e-30, q)
    term = r / (AA * q_safe)
    rlast = r.copy()
    converged = np.zeros(rlast.shape, dtype=bool)

    for _ in range(500):
        den = 1.0 + term * (rlast + 1.0) * (rlast * rlast + 1.0)
        rnext = 0.5 * (rlast + 1.0 + r / den)
        newly_converged = (np.abs(rnext - rlast) < ERR) & ~converged
        converged |= newly_converged
        # Freeze converged values
        rlast = np.where(converged, rlast, rnext)
        if converged.all():
            break

    return rlast * tamb


def _heat_exchange(
    v_eff: np.ndarray,
    dia: np.ndarray,
    tf: np.ndarray,
    ts: np.ndarray,
    cond: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized convective + radiative heat-transfer coefficients.

    :return: ``(hfm, hbar, en)`` arrays matching input shapes.
    """
    B_COND = 5.75e-05
    SIGMA_SB = 5.67e-08
    HRADF = 0.5
    VIS = 7.5e-05
    A_COND = 8.75e-03
    FMFAC = 0.382

    # Convective film coefficient (only where dia > B_COND)
    big_mask = dia > B_COND
    dia_safe = np.where(big_mask, dia, 1.0)  # avoid /0

    re = v_eff * dia_safe / VIS
    enuair = 0.344 * np.power(re, 0.56)
    conair = A_COND + B_COND * tf
    fac = np.sqrt(np.abs(tf - ts) / dia_safe)
    hfmin = FMFAC * np.sqrt(fac)
    hfm_conv = enuair * conair / dia_safe
    hfm_conv = np.maximum(hfm_conv, hfmin)
    hfm = np.where(big_mask, hfm_conv, 0.0)

    # Radiative
    hrad = HRADF * SIGMA_SB * (tf + ts) * (tf * tf + ts * ts)
    hbar = hfm + hrad
    cond_safe = np.where(np.abs(cond) < 1e-30, 1e-30, cond)
    en = hbar * dia / cond_safe
    return hfm, hbar, en


def _t_ignite(
    tpdr: np.ndarray,
    tpig: np.ndarray,
    tpfi: np.ndarray,
    cond: np.ndarray,
    chtd: np.ndarray,
    fmof: np.ndarray,
    dend: np.ndarray,
    hbar: np.ndarray,
    tamb: np.ndarray,
) -> np.ndarray:
    """Vectorized time-to-ignition via fixed 60-iteration bisection.

    :return: Predicted ignition time (s), same shape as inputs.
    """
    PINV = 2.125534
    HVAP = 2.177e+06
    CPM = 4186.0
    CONC = 4.27e-04
    A03 = -1.3371565
    A13 = 0.4653628
    A23 = -0.1282064

    # Guard the denominator with a dtype-appropriate epsilon. Using 1e-30 can
    # be a no-op for float32 and still trigger divide-by-zero warnings.
    den = tpfi - tamb
    eps = np.array(1e-12, dtype=den.dtype)
    den_safe = np.where(np.abs(den) < eps, np.where(den >= 0.0, eps, -eps), den)
    b03 = A03 * (tpfi - tpig) / den_safe

    xlo = np.zeros_like(b03)
    xhi = np.ones_like(b03)
    for _ in range(60):
        xav = 0.5 * (xlo + xhi)
        fav = b03 + xav * (A13 + xav * (A23 + xav))
        xlo = np.where(fav < 0.0, xav, xlo)
        xhi = np.where(fav > 0.0, xav, xhi)
        # Early exit per-element not practical; just run all iterations

    xav = 0.5 * (xlo + xhi)
    xav_safe = np.where(np.abs(xav) < 1e-30, 1e-30, xav)
    beta = PINV * (1.0 - xav_safe) / xav_safe
    conw = cond + CONC * dend * fmof
    dtb = tpdr - tamb
    dti = np.where(np.abs(tpig - tamb) < 1e-30, 1e-30, tpig - tamb)
    ratio = (HVAP + CPM * dtb) / (chtd * dti)
    rhoc = dend * chtd * (1.0 + fmof * ratio)
    hbar_safe = np.where(np.abs(hbar) < 1e-30, 1e-30, hbar)
    return (beta / hbar_safe) ** 2 * conw * rhoc


def _dry_time(enu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorized dimensionless drying time (fixed 15-iteration bisection).

    :return: Dimensionless drying time, same shape as inputs.
    """
    P_CONST = 0.47047
    A = 0.7478556
    B = 0.4653628
    C = 0.1282064
    rhs = (1.0 - theta) / A

    xl = np.zeros_like(rhs)
    xh = np.ones_like(rhs)
    for _ in range(15):
        xm = 0.5 * (xl + xh)
        val = xm * (B - xm * (C - xm)) - rhs
        xl = np.where(val < 0.0, xm, xl)
        xh = np.where(val >= 0.0, xm, xh)

    xm = 0.5 * (xl + xh)
    xm_safe = np.where(np.abs(xm) < 1e-30, 1e-30, xm)
    x = (1.0 / xm_safe - 1.0) / P_CONST
    enu_safe = np.where(np.abs(enu) < 1e-30, 1e-30, enu)
    return (0.5 * x / enu_safe) ** 2


def _duff_burn(
    wdf_load: np.ndarray,
    dfm: np.ndarray,
    duff_pct_consumed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized duff burning computation.

    :return: ``(dfi, tdf, smolder_rate)`` each shape ``(C,)``.
    """
    suppress = (wdf_load <= 0.0) | (dfm >= 1.96)

    dfi = 11.25 - 4.05 * dfm

    # Consumed fraction
    use_pdc = (duff_pct_consumed >= 0.0) & (duff_pct_consumed <= 100.0)
    ff = np.where(use_pdc, duff_pct_consumed / 100.0, 0.837 - 0.426 * dfm)
    ff = np.where(ff <= 0.0, 0.0, ff)

    denom = 7.5 - 2.7 * dfm
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    tdf = 1.0e4 * ff * wdf_load / denom
    tdf_safe = np.where(np.abs(tdf) < 1e-30, 1e-30, tdf)
    smolder_rate = np.where(tdf > 0.0, ff * wdf_load / tdf_safe, 0.0)

    dfi = np.where(suppress | (ff <= 0.0), 0.0, dfi)
    tdf = np.where(suppress | (ff <= 0.0), 0.0, tdf)
    smolder_rate = np.where(suppress | (ff <= 0.0), 0.0, smolder_rate)

    return dfi, tdf, smolder_rate


# ---------------------------------------------------------------------------
# Vectorized sorting and overlap
# ---------------------------------------------------------------------------

def _sort_fuels(
    sigma: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
) -> np.ndarray:
    """Sort fuel classes per cell by increasing size.

    :param sigma: ``(C, P)`` SAV array.
    :param fmois: ``(C, P)`` moisture array.
    :param dendry: ``(C, P)`` density array.
    :return: ``(C, P)`` integer sort-index array.
    """
    C, P = sigma.shape
    keys = np.empty((C, P), dtype=np.float64)
    # Composite key: primary = 1/sigma (increasing size), secondary = fmois,
    # tertiary = dendry.  Scale so primary dominates.
    inv_sigma = 1.0 / np.where(np.abs(sigma) < 1e-30, 1e-30, sigma)
    # Normalise each component to [0, 1] then combine
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn = arr.min(axis=1, keepdims=True)
        mx = arr.max(axis=1, keepdims=True)
        rng = mx - mn
        rng = np.where(rng < 1e-30, 1.0, rng)
        return (arr - mn) / rng

    keys = _norm(inv_sigma) * 1e6 + _norm(fmois) * 1e3 + _norm(dendry)
    return np.argsort(keys, axis=1)


def _apply_sort(arr: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Reorder a ``(C, P)`` array by per-cell sort indices."""
    return np.take_along_axis(arr, key, axis=1)


def _overlaps(
    wdry: np.ndarray,
    sigma: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized interaction matrix for all cells.

    :param wdry: ``(C, P)``
    :param sigma: ``(C, P)``
    :param fmois: ``(C, P)``
    :param dendry: ``(C, P)``
    :return: ``(elam (C, P, P), alone (C, P), area (C, P))``
    """
    C, P = wdry.shape
    nkl = P * (P + 1) // 2 + P

    xmat = np.zeros((C, nkl), dtype=np.float64)
    elam = np.zeros((C, P, P), dtype=np.float64)
    alone = np.zeros((C, P), dtype=np.float64)
    area = np.zeros((C, P), dtype=np.float64)

    dendry_safe = np.where(np.abs(dendry) < 1e-30, 1e-30, dendry)

    for k in range(1, P + 1):
        ki = k - 1
        for l in range(1, k + 1):
            li = l - 1
            kl = k * (k + 1) // 2 + l - 1

            ak = 3.25 * np.exp(-20.0 * fmois[:, li] ** 2)  # (C,)
            siga = ak * sigma[:, ki] / np.pi
            a = siga * wdry[:, li] / dendry_safe[:, li]

            if k == l:
                bb = 1.0 - np.exp(-a)
                bb = np.maximum(bb, 1e-30)
                area[:, ki] = bb
            else:
                bb = np.minimum(a, 1.0)
            xmat[:, kl] = bb

    if P == 1:
        elam[:, 0, 0] = xmat[:, 1]
        alone[:, 0] = 1.0 - elam[:, 0, 0]
        return elam, alone, area

    for k in range(1, P + 1):
        ki = k - 1
        frac = np.zeros(C, dtype=np.float64)
        for l in range(1, k + 1):
            kl = k * (k + 1) // 2 + l - 1
            frac += xmat[:, kl]

        over = frac > 1.0
        inv_frac = np.where(over, 1.0 / np.maximum(frac, 1e-30), 1.0)

        for l in range(1, k + 1):
            li = l - 1
            kl = k * (k + 1) // 2 + l - 1
            elam[:, ki, li] = np.where(over,
                                       xmat[:, kl] * inv_frac,
                                       xmat[:, kl])
        alone[:, ki] = np.where(over, 0.0, 1.0 - frac)

    return elam, alone, area


# ---------------------------------------------------------------------------
# Pair-index helpers (same as scalar, used to build index maps)
# ---------------------------------------------------------------------------

def _maxkl(n: int) -> int:
    return n * (n + 1) // 2 + n


def _build_kl_pairs(P: int):
    """Return list of ``(k, l, kl)`` tuples and helper dicts.

    Same structure as scalar ``_build_kl_map`` but returns plain lists
    for use in the vectorized loops.
    """
    pairs = []
    by_k = {}
    k0_map = {}
    kl_slices = {}

    for k in range(1, P + 1):
        k_pairs = []
        k_indices = []
        for l in range(0, k + 1):
            kl = k * (k + 1) // 2 + l - 1
            pairs.append((k, l, kl))
            k_pairs.append((l, kl))
            k_indices.append(kl)
            if l == 0:
                k0_map[k] = kl
        by_k[k] = k_pairs
        kl_slices[k] = k_indices

    return pairs, by_k, k0_map, kl_slices


# ---------------------------------------------------------------------------
# Vectorized validation
# ---------------------------------------------------------------------------

def _validate_fuel(
    wdry: np.ndarray,
    ash: np.ndarray,
    htval: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
    sigma: np.ndarray,
    cheat: np.ndarray,
    condry: np.ndarray,
    tpig: np.ndarray,
    tchar: np.ndarray,
) -> np.ndarray:
    """Per-cell fuel validation.  Returns error code ``(C,)``; 0 = OK."""
    C = wdry.shape[0]
    err = np.zeros(C, dtype=np.int32)

    # Map attribute name → (array, error_code)
    checks = [
        (wdry,   'wdry',   20),
        (ash,    'ash',    21),
        (htval,  'htval',  22),
        (fmois,  'fmois',  23),
        (dendry, 'dendry', 24),
        (sigma,  'sigma',  25),
        (cheat,  'cheat',  26),
        (condry, 'condry', 27),
        (tpig,   'tpig',   28),
        (tchar,  'tchar',  29),
    ]

    for arr, attr, code in checks:
        lo, hi, _ = _FUEL_BOUNDS[attr]
        # Only check particles with non-zero load
        has_load = wdry > 0.0
        bad = has_load & ((arr <= lo) | (arr >= hi))
        # Any particle bad → cell fails
        cell_bad = bad.any(axis=1)
        err = np.where((err == 0) & cell_bad, code, err)

    return err


def _validate_fire(
    fi: np.ndarray,
    ti: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    tamb: np.ndarray,
    wdf: np.ndarray,
    dfm: np.ndarray,
) -> np.ndarray:
    """Per-cell fire-environment validation.  Returns error code ``(C,)``."""
    C = fi.shape[0]
    err = np.zeros(C, dtype=np.int32)

    lo, hi, _ = _FIRE_BOUNDS['fistart']
    err = np.where((err == 0) & ((fi < lo) | (fi > hi)), 10, err)

    lo, hi, _ = _FIRE_BOUNDS['ti']
    err = np.where((err == 0) & ((ti < lo) | (ti > hi)), 11, err)

    lo, hi, _ = _FIRE_BOUNDS['u']
    err = np.where((err == 0) & ((u < lo) | (u > hi)), 12, err)

    # d (fuel bed depth) is intentionally omitted here: it is always clipped
    # to [lo, hi] by run_burnup's _clip_both before reaching this
    # validator, and the scalar _run_burnup_cell also clips rather than
    # raising for d.  Using error code 10 here was a bug (collides with
    # fistart).  Direct callers of burnup who skip clipping still get
    # protection from the physics (depth only affects v_eff).

    lo, hi, _ = _FIRE_BOUNDS['tamb_c']
    err = np.where((err == 0) & ((tamb < lo) | (tamb > hi)), 13, err)

    lo, hi, _ = _FIRE_BOUNDS['dfm']
    has_duff = wdf > 0.0
    err = np.where((err == 0) & has_duff & ((dfm < lo) | (dfm > hi)), 14, err)

    return err


# ---------------------------------------------------------------------------
# Main vectorized simulation
# ---------------------------------------------------------------------------

def burnup(
    *,
    # Fuel-particle arrays — all (C, P)
    wdry: np.ndarray,
    htval: np.ndarray,
    fmois: np.ndarray,
    dendry: np.ndarray,
    sigma: np.ndarray,
    cheat: np.ndarray,
    condry: np.ndarray,
    tpig: np.ndarray,
    tchar: np.ndarray,
    ash: np.ndarray,
    # Fire-environment arrays — all (C,)
    fi: np.ndarray,
    ti: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    tamb: np.ndarray,
    r0: np.ndarray,
    dr: np.ndarray,
    # Simulation parameters — all (C,) or scalar
    dt: float,
    max_timesteps: int = 3000,
    wdf: Optional[np.ndarray] = None,
    dfm: Optional[np.ndarray] = None,
    duff_pct_consumed: Optional[np.ndarray] = None,
    fint_switch: float = 15.0,
    validate: bool = True,
) -> BurnupArrayResult:
    """Run the BURNUP simulation for C cells simultaneously.

    All fuel-particle arrays have shape ``(C, P)`` where ``C`` is the number
    of spatial cells and ``P`` is the number of particle classes (must be
    uniform across cells; pad unused classes with ``wdry=0``).

    Fire-environment arrays have shape ``(C,)``.

    :param wdry: Oven-dry mass loading (kg/m²), ``(C, P)``.
    :param htval: Low heat of combustion (J/kg), ``(C, P)``.
    :param fmois: Moisture content (fraction), ``(C, P)``.
    :param dendry: Oven-dry density (kg/m³), ``(C, P)``.
    :param sigma: SAV (1/m), ``(C, P)``.
    :param cheat: Heat capacity (J/kg·K), ``(C, P)``.
    :param condry: Thermal conductivity (W/m·K), ``(C, P)``.
    :param tpig: Ignition temperature (°C), ``(C, P)``.
    :param tchar: Char temperature (°C), ``(C, P)``.
    :param ash: Ash fraction, ``(C, P)``.
    :param fi: Fire intensity (kW/m²), ``(C,)``.
    :param ti: Residence time (s), ``(C,)``.
    :param u: Windspeed at fuelbed top (m/s), ``(C,)``.
    :param d: Fuel bed depth (m), ``(C,)``.
    :param tamb: Ambient temperature (°C), ``(C,)``.
    :param r0: Minimum mixing parameter, ``(C,)``.
    :param dr: Mixing parameter range, ``(C,)``.
    :param dt: Integration time-step (s), uniform across cells.
    :param max_timesteps: Maximum number of time steps.
    :param wdf: Duff loading (kg/m²), ``(C,)``.  Default 0.
    :param dfm: Duff moisture (fraction), ``(C,)``.  Default 2.0.
    :param duff_pct_consumed: FOFEM duff pct consumed, ``(C,)``.  Default -1.
    :param fint_switch: Flaming/smoldering threshold (kW/m²).
    :param validate: If ``True``, validate all inputs.
    :return: :class:`BurnupArrayResult` with all output arrays.
    """
    # ------------------------------------------------------------------
    # 0. Ensure proper array shapes
    # ------------------------------------------------------------------
    wdry   = np.asarray(wdry,   dtype=np.float64)
    htval  = np.asarray(htval,  dtype=np.float64)
    fmois  = np.asarray(fmois,  dtype=np.float64)
    dendry = np.asarray(dendry, dtype=np.float64)
    sigma  = np.asarray(sigma,  dtype=np.float64)
    cheat  = np.asarray(cheat,  dtype=np.float64)
    condry = np.asarray(condry, dtype=np.float64)
    tpig   = np.asarray(tpig,   dtype=np.float64)
    tchar  = np.asarray(tchar,  dtype=np.float64)
    ash    = np.asarray(ash,    dtype=np.float64)

    fi    = np.atleast_1d(np.asarray(fi,    dtype=np.float64))
    ti    = np.atleast_1d(np.asarray(ti,    dtype=np.float64))
    u     = np.atleast_1d(np.asarray(u,     dtype=np.float64))
    d     = np.atleast_1d(np.asarray(d,     dtype=np.float64))
    tamb  = np.atleast_1d(np.asarray(tamb,  dtype=np.float64))
    r0    = np.atleast_1d(np.asarray(r0,    dtype=np.float64))
    dr    = np.atleast_1d(np.asarray(dr,    dtype=np.float64))

    C, P = wdry.shape
    nkl = _maxkl(P)

    if wdf is None:
        wdf_arr = np.zeros(C, dtype=np.float64)
    else:
        wdf_arr = np.atleast_1d(np.asarray(wdf, dtype=np.float64))
    if dfm is None:
        dfm_arr = np.full(C, 2.0, dtype=np.float64)
    else:
        dfm_arr = np.atleast_1d(np.asarray(dfm, dtype=np.float64))
    if duff_pct_consumed is None:
        dpc_arr = np.full(C, -1.0, dtype=np.float64)
    else:
        dpc_arr = np.atleast_1d(np.asarray(duff_pct_consumed, dtype=np.float64))

    error_code = np.zeros(C, dtype=np.int32)

    # ------------------------------------------------------------------
    # 1. Validate
    # ------------------------------------------------------------------
    if validate:
        fuel_err = _validate_fuel(wdry, ash, htval, fmois, dendry,
                                        sigma, cheat, condry, tpig, tchar)
        fire_err = _validate_fire(fi, ti, u, d, tamb, wdf_arr, dfm_arr)
        error_code = np.where(fuel_err != 0, fuel_err,
                              np.where(fire_err != 0, fire_err, 0))

    # Check for cells with no fuel
    has_fuel = (wdry > 0.0).any(axis=1)
    error_code = np.where((error_code == 0) & ~has_fuel, 90, error_code)

    # Active mask — only cells with no errors proceed
    active = error_code == 0

    # ------------------------------------------------------------------
    # 2. Convert temperatures °C → K
    # ------------------------------------------------------------------
    tamb_k = tamb + 273.0  # (C,)
    tpig_k = tpig + 273.0  # (C, P)
    tchar_k = tchar + 273.0  # (C, P)

    # ------------------------------------------------------------------
    # 3. Sort fuel classes per cell
    # ------------------------------------------------------------------
    sort_key = _sort_fuels(sigma, fmois, dendry)
    wdry   = _apply_sort(wdry,   sort_key)
    ash    = _apply_sort(ash,    sort_key)
    htval  = _apply_sort(htval,  sort_key)
    fmois  = _apply_sort(fmois,  sort_key)
    dendry = _apply_sort(dendry, sort_key)
    sigma  = _apply_sort(sigma,  sort_key)
    cheat  = _apply_sort(cheat,  sort_key)
    condry = _apply_sort(condry, sort_key)
    tpig_k = _apply_sort(tpig_k, sort_key)
    tchar_k = _apply_sort(tchar_k, sort_key)

    # ------------------------------------------------------------------
    # 4. Compute interaction matrices
    # ------------------------------------------------------------------
    elam, alone, area = _overlaps(wdry, sigma, fmois, dendry)

    # ------------------------------------------------------------------
    # 4b. Pair index map (structural, independent of cells)
    # ------------------------------------------------------------------
    pairs, by_k, k0_map, kl_slices = _build_kl_pairs(P)

    # ------------------------------------------------------------------
    # 5. Build pair-level arrays  (C, NKL)
    # ------------------------------------------------------------------
    diam_arr = np.zeros((C, nkl), dtype=np.float64)
    xmat     = np.zeros((C, nkl), dtype=np.float64)
    wo       = np.zeros((C, nkl), dtype=np.float64)

    for k in range(1, P + 1):
        ki = k - 1
        sigma_safe = np.where(np.abs(sigma[:, ki]) < 1e-30,
                              1e-30, sigma[:, ki])
        diak = 4.0 / sigma_safe  # (C,)
        wtk = wdry[:, ki]  # (C,)
        for l, kl in by_k[k]:
            diam_arr[:, kl] = diak
            if l == 0:
                xmat[:, kl] = alone[:, ki]
            else:
                xmat[:, kl] = elam[:, ki, l - 1]
            wo[:, kl] = wtk * xmat[:, kl]

    # ------------------------------------------------------------------
    # 6. Emissions accumulators
    # ------------------------------------------------------------------
    smoldering = np.zeros((C, P + 1), dtype=np.float64)
    flaming    = np.zeros((C, P), dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. Duff burning
    # ------------------------------------------------------------------
    dfi, tdf, duff_smolder_rate = _duff_burn(wdf_arr, dfm_arr, dpc_arr)
    smoldering[:, P] = duff_smolder_rate

    # ------------------------------------------------------------------
    # 8. Initialise state arrays  (C, NKL) or (C, P)
    # ------------------------------------------------------------------
    flit = np.zeros((C, P), dtype=np.float64)
    fout = np.zeros((C, P), dtype=np.float64)
    alfa = condry / np.where(dendry * cheat < 1e-30, 1e-30,
                             dendry * cheat)  # (C, P)
    fint = np.zeros((C, P), dtype=np.float64)

    dendry_safe = np.where(np.abs(dendry) < 1e-30, 1e-30, dendry)
    work = 1.0 / (255.0 * (dendry_safe / 446.0) * 2.01e6
                   * (1.0 + 1.67 * fmois))  # (C, P)

    tout  = np.full((C, nkl), _RINDEF, dtype=np.float64)
    tign  = np.full((C, nkl), _RINDEF, dtype=np.float64)
    tdry  = np.full((C, nkl), _RINDEF, dtype=np.float64)
    tcum  = np.zeros((C, nkl), dtype=np.float64)
    qcum  = np.zeros((C, nkl), dtype=np.float64)
    acum  = np.zeros((C, nkl), dtype=np.float64)
    ddot  = np.zeros((C, nkl), dtype=np.float64)
    wodot = np.zeros((C, nkl), dtype=np.float64)
    qdot  = np.zeros((C, nkl, _MXSTEP), dtype=np.float64)

    # Pre-compute effective velocity  (C,)
    v_eff = np.sqrt(u * u + 0.53 * 9.8 * d)

    # ---- First fire-temperature estimate ----
    fi_cur = fi.copy()  # (C,)
    r_init = r0 + 0.25 * dr  # (C,)
    tf = _temp_fire(fi_cur, r_init, tamb_k)  # (C,)

    # Cells where fire can't dry fuel
    cant_dry = tf <= (_TPDRY + 10.0)
    error_code = np.where((error_code == 0) & cant_dry, 15, error_code)
    active &= ~cant_dry

    tf_safe = np.where(np.abs(tf - tamb_k) < 1e-30, tamb_k + 1.0, tf)
    thd = (_TPDRY - tamb_k) / (tf_safe - tamb_k)  # (C,)
    tx = 0.5 * (tamb_k + _TPDRY)  # (C,)

    # ---- Estimate drying start times ----
    for k in range(1, P + 1):
        ki = k - 1
        conwet = condry[:, ki] + 4.27e-04 * dendry[:, ki] * fmois[:, ki]
        cpwet = cheat[:, ki] + fmois[:, ki] * _CH2O
        conwet_safe = np.where(np.abs(conwet) < 1e-30, 1e-30, conwet)
        fac_base = dendry[:, ki] * cpwet / conwet_safe  # (C,)
        for l, kl in by_k[k]:
            dia = diam_arr[:, kl]  # (C,)
            _, _, en = _heat_exchange(
                v_eff, dia, tf, tx, conwet)
            dt_val = _dry_time(en, thd)
            tdry[:, kl] = (0.5 * dia) ** 2 * fac_base * dt_val

    # ---- Determine which components ignite during spreading fire ----
    for k in range(1, P + 1):
        ki = k - 1
        c_cond = condry[:, ki]  # (C,)
        tigk = tpig_k[:, ki]  # (C,)
        for l, kl in by_k[k]:
            can_dry = tdry[:, kl] < ti  # (C,) bool
            dia = diam_arr[:, kl]
            ts_est = 0.5 * (_TPDRY + tigk)
            _, hbar, _ = _heat_exchange(v_eff, dia, tf, ts_est, c_cond)

            tcum_val = np.maximum((tf - ts_est) * (ti - tdry[:, kl]), 0.0)
            qcum_val = hbar * tcum_val
            tcum[:, kl] = np.where(can_dry, tcum_val, tcum[:, kl])
            qcum[:, kl] = np.where(can_dry, qcum_val, qcum[:, kl])

            can_ignite = can_dry & (tf > tigk + 10.0)
            dtign = _t_ignite(
                np.full(C, _TPDRY), tpig_k[:, ki], tf,
                condry[:, ki], cheat[:, ki], fmois[:, ki],
                dendry[:, ki], hbar, tamb_k)
            trt = tdry[:, kl] + dtign
            tign[:, kl] = np.where(can_ignite, 0.5 * trt, tign[:, kl])
            lit_mask = can_ignite & (ti > trt)
            flit[:, ki] += np.where(lit_mask, xmat[:, kl], 0.0)

    # ---- Verify at least one component ignited ----
    no_ignition = ~(flit > 0.0).any(axis=1)
    error_code = np.where((error_code == 0) & no_ignition, 16, error_code)
    active &= ~no_ignition

    # ---- Reset time origin to earliest ignition ----
    trt_min = np.min(tign, axis=1)  # (C,)
    for kl_idx in range(nkl):
        finite_dry = tdry[:, kl_idx] < _RINDEF
        tdry[:, kl_idx] = np.where(finite_dry,
                                    tdry[:, kl_idx] - trt_min, tdry[:, kl_idx])
        finite_ign = tign[:, kl_idx] < _RINDEF
        tign[:, kl_idx] = np.where(finite_ign,
                                    tign[:, kl_idx] - trt_min, tign[:, kl_idx])

    # ---- Initial burning rates for ignited components ----
    for k in range(1, P + 1):
        ki = k - 1
        not_lit = flit[:, ki] == 0.0  # (C,)
        ts_k = tchar_k[:, ki]
        c_k = condry[:, ki]
        wk = work[:, ki]
        for l, kl in by_k[k]:
            dia = diam_arr[:, kl]
            _, hbar, _ = _heat_exchange(v_eff, dia, tf, ts_k, c_k)
            qdot[:, kl, 0] = np.where(~not_lit,
                                       hbar * np.maximum(tf - ts_k, 0.0), 0.0)
            ddt_val = ti - tign[:, kl]
            hbar_safe = np.where(np.abs(hbar) < 1e-30, 1e-30, hbar)
            acum[:, kl] = np.where(~not_lit,
                                    (c_k / hbar_safe) ** 2 * ddt_val, 0.0)
            ddot[:, kl] = np.where(~not_lit, qdot[:, kl, 0] * wk, 0.0)
            dia_safe = np.where(dia > 0, dia, 1.0)
            tout[:, kl] = np.where(~not_lit & (ddot[:, kl] > 0),
                                    dia / np.maximum(ddot[:, kl], 1e-30),
                                    np.where(not_lit, _RINDEF, tout[:, kl]))
            ddt_safe = np.where(np.abs(ddt_val) < 1e-30, 1e-30, ddt_val)
            dnext = np.maximum(0.0, dia - ddt_val * ddot[:, kl])
            wnext = np.where(dia > 0, wo[:, kl] * (dnext / dia_safe) ** 2, 0.0)
            wodot[:, kl] = np.where(~not_lit,
                                     (wo[:, kl] - wnext) / ddt_safe, 0.0)
            diam_arr[:, kl] = np.where(~not_lit, dnext, diam_arr[:, kl])
            wo[:, kl] = np.where(~not_lit, wnext, wo[:, kl])
            # Fully consumed
            consumed = ~not_lit & (dnext <= 0.0)
            flit[:, ki] -= np.where(consumed, xmat[:, kl], 0.0)
            fout[:, ki] += np.where(consumed, xmat[:, kl], 0.0)
            wodot[:, kl] = np.where(consumed, 0.0, wodot[:, kl])
            ddot[:, kl] = np.where(consumed, 0.0, ddot[:, kl])

    # ------------------------------------------------------------------
    # 9. Fire intensity (vectorised)
    # ------------------------------------------------------------------
    def _fire_intensity_v() -> np.ndarray:
        """Compute site-average fire intensity for all cells.  Returns (C,)."""
        total = np.zeros(C, dtype=np.float64)
        for k in range(1, P + 1):
            ki = k - 1
            wdotk = np.zeros(C, dtype=np.float64)
            for kl_idx in kl_slices[k]:
                wdotk += wodot[:, kl_idx]
            term = (1.0 - ash[:, ki]) * htval[:, ki] * wdotk * 1.0e-03
            ark = area[:, ki]
            fint[:, ki] = np.where(ark > _SMALLX, term / ark - term, 0.0)

            smoldering[:, ki] = wodot[:, k0_map[k]]
            wnoduff = wdotk - smoldering[:, ki]
            test = np.where(wnoduff > 0.0,
                            (wnoduff / np.maximum(wdotk, 1e-30)) * fint[:, ki],
                            0.0)
            threshold = np.where(ark > _SMALLX,
                                 fint_switch / ark - fint_switch, np.inf)
            is_flaming = test >= threshold
            flaming[:, ki] += np.where(is_flaming, wnoduff, 0.0)
            smoldering[:, ki] += np.where(~is_flaming, wnoduff, 0.0)
            total += term
        return total

    # ------------------------------------------------------------------
    # 10. Initial record and output allocation
    # ------------------------------------------------------------------
    wd0 = wo.sum(axis=1)  # (C,)
    wd0 = np.where(wd0 == 0.0, 1.0, wd0)

    T = max_timesteps + 1  # +1 for initial record at t=ti
    out_time = np.full((C, T), np.nan, dtype=np.float64)
    out_wdf  = np.full((C, T), np.nan, dtype=np.float64)
    out_ff   = np.full((C, T), np.nan, dtype=np.float64)
    out_comp_fla = np.full((C, T, P), np.nan, dtype=np.float64)
    out_comp_smo = np.full((C, T, P + 1), np.nan, dtype=np.float64)
    n_steps = np.zeros(C, dtype=np.int32)

    def _record(step_idx: int, time_arr: np.ndarray, mask: np.ndarray):
        wdf_val = wo.sum(axis=1) / wd0  # (C,)
        wt_flam = flaming.sum(axis=1)
        wt_smol = smoldering.sum(axis=1)
        denom = wt_flam + wt_smol
        denom_safe = np.where(denom > 0, denom, 1.0)
        ff_val = np.where(denom > 0, wt_flam / denom_safe, 0.0)

        out_time[mask, step_idx] = time_arr[mask]
        out_wdf[mask, step_idx] = wdf_val[mask]
        out_ff[mask, step_idx] = ff_val[mask]
        out_comp_fla[mask, step_idx, :] = flaming[mask, :P]
        out_comp_smo[mask, step_idx, :] = smoldering[mask, :P + 1]
        n_steps[mask] = step_idx + 1

        flaming[mask] = 0.0
        smoldering[mask, :P] = 0.0

    # Initial record
    fi_cur[:] = _fire_intensity_v()
    _record(0, ti.copy(), active)

    # ------------------------------------------------------------------
    # 11. Time-step loop
    # ------------------------------------------------------------------
    fimin = 0.1
    tis = ti.copy()  # (C,)
    half_dr = 0.5 * dr  # (C,)
    r0_half_dr = r0 + half_dr  # (C,)
    ncalls = np.zeros(C, dtype=np.int32)

    for step in range(1, max_timesteps + 1):
        if not active.any():
            break

        ncalls[active] += 1
        tnow = tis.copy()
        tnext = tnow + dt
        tifi = tnow - (ncalls - 1) * dt

        fid_cur = np.where(tis < tdf, dfi, 0.0)  # (C,)

        # ---- Process each (k, l) pair ----
        for k in range(1, P + 1):
            ki = k - 1
            c_k = condry[:, ki]
            wk = work[:, ki]
            alfa_k = alfa[:, ki]
            tchar_ki = tchar_k[:, ki]
            tpig_ki = tpig_k[:, ki]
            cheat_ki = cheat[:, ki]
            fmois_ki = fmois[:, ki]
            dendry_ki = dendry[:, ki]
            condry_ki = condry[:, ki]

            for l, kl in by_k[k]:
                # Only operate on active cells
                m = active.copy()

                tdun = tout[:, kl]
                tlit = tign[:, kl]
                dryt_val = tdry[:, kl]
                dia = diam_arr[:, kl]

                # --- burned out ---
                burned_out = m & (tnow >= tdun)
                ddot[burned_out, kl] = 0.0
                wodot[burned_out, kl] = 0.0

                m_remaining = m & ~burned_out

                # --- burning out this step ---
                burning_out = m_remaining & (tnext >= tdun)
                tgo = tdun - tnow
                tgo_safe = np.where(tgo > 0, tgo, 1.0)
                ddot[:, kl] = np.where(burning_out & (tgo > 0),
                                        diam_arr[:, kl] / tgo_safe,
                                        ddot[:, kl])
                wodot[:, kl] = np.where(burning_out & (tgo > 0),
                                         wo[:, kl] / tgo_safe,
                                         np.where(burning_out, 0.0,
                                                  wodot[:, kl]))
                wo[:, kl] = np.where(burning_out, 0.0, wo[:, kl])
                diam_arr[:, kl] = np.where(burning_out, 0.0, diam_arr[:, kl])

                m_remaining &= ~burning_out

                # --- ignited & burning ---
                ignited = m_remaining & (tnow >= tlit)

                if ignited.any():
                    ts = tchar_ki  # (C,)

                    # Compute gi and r_val depending on l
                    if l == 0:
                        r_val = r0_half_dr
                        gi = fi_cur + fid_cur
                    elif l == k:
                        r_val = r0 + half_dr * (1.0 + flit[:, ki])
                        gi = fi_cur + flit[:, ki] * fint[:, ki]
                    else:
                        li = l - 1
                        r_val = r0 + half_dr * (1.0 + flit[:, li])
                        gi = fi_cur + fint[:, ki] + flit[:, li] * fint[:, li]

                    tf_loc = _temp_fire(gi, r_val, tamb_k)
                    dia_loc = diam_arr[:, kl]
                    _, hbar, _ = _heat_exchange(
                        v_eff, dia_loc, tf_loc, ts, c_k)
                    qqq = np.where(tf_loc > ts,
                                   hbar * (tf_loc - ts), 0.0)

                    tst = np.where(tlit > tifi, tlit, tifi)
                    dt_safe = np.where(dt > 0, dt, 1.0)
                    nspan = np.maximum(1, np.round((tnext - tst) / dt_safe)
                                       ).astype(int)

                    # Update qdot rolling window (simplified: just store latest)
                    for ci in np.where(ignited)[0]:
                        ns = int(nspan[ci])
                        if ns <= _MXSTEP:
                            qdot[ci, kl, ns - 1] = qqq[ci]
                        else:
                            qdot[ci, kl, :-1] = qdot[ci, kl, 1:]
                            qdot[ci, kl, _MXSTEP - 1] = qqq[ci]

                    hbar_safe = np.where(np.abs(hbar) < 1e-30, 1e-30, hbar)
                    acum[:, kl] = np.where(ignited,
                                            acum[:, kl] + (c_k / hbar_safe) ** 2 * dt,
                                            acum[:, kl])

                    # Time averaging window bound
                    tav1 = tnext - tlit
                    alfa_k_safe = np.where(
                        np.abs(alfa_k) < 1e-30, 1e-30, alfa_k)
                    tav2 = acum[:, kl] / alfa_k_safe
                    dia4 = dia_loc * 0.25
                    tav3 = dia4 * dia4 / alfa_k_safe
                    tavg = np.minimum(np.minimum(tav1, tav2), tav3)

                    # Full rolling-window qdot average (vectorized)
                    # Walk backwards from idx = min(nspan, _MXSTEP) to 0,
                    # accumulating time-weighted heat flux, matching scalar
                    # burnup.py lines 1096-1111.
                    idx_start = np.minimum(nspan, _MXSTEP)  # (C,)
                    qdsum = np.zeros(C, dtype=np.float64)
                    tspan = np.zeros(C, dtype=np.float64)
                    deltim = np.full(C, dt, dtype=np.float64)
                    still_going = ignited.copy()
                    cur_idx = idx_start.copy()  # current position (1-based count → will decrement first)

                    for _wi in range(_MXSTEP):
                        if not still_going.any():
                            break
                        cur_idx -= 1
                        # At idx==0, deltim = tnext - tspan - tlit
                        at_zero = still_going & (cur_idx == 0)
                        deltim = np.where(at_zero,
                                          tnext - tspan - tlit, deltim)
                        # Cap deltim so tspan + deltim <= tavg
                        over = still_going & ((tspan + deltim) >= tavg)
                        deltim = np.where(over, tavg - tspan, deltim)
                        # Gather qdot value at cur_idx for each cell
                        # cur_idx is (C,) int; qdot is (C, nkl, _MXSTEP)
                        safe_idx = np.clip(cur_idx, 0, _MXSTEP - 1)
                        qval = qdot[np.arange(C), kl, safe_idx]
                        qdsum = np.where(still_going,
                                         qdsum + qval * deltim, qdsum)
                        tspan = np.where(still_going,
                                         tspan + deltim, tspan)
                        # Stop conditions: tspan >= tavg or cur_idx <= 0
                        done_w = (tspan >= tavg) | (cur_idx <= 0)
                        still_going &= ~done_w

                    tspan_safe = np.where(tspan > 0, tspan, 1.0)
                    qdavg = np.where(tspan > 0, qdsum / tspan_safe, 0.0)
                    qdavg = np.maximum(qdavg, 0.0)

                    ddot_new = qdavg * wk
                    dnext = np.maximum(0.0, dia_loc - dt * ddot_new)
                    dia_safe = np.where(dia_loc > 0, dia_loc, 1.0)
                    wnext = np.where(dia_loc > 0,
                                     wo[:, kl] * (dnext / dia_safe) ** 2,
                                     0.0)

                    # Burnout time estimate
                    tout_new = tout[:, kl].copy()
                    consumed_now = (dnext == 0.0) & (ddot_new > 0.0)
                    ddot_safe = np.where(ddot_new > 0, ddot_new, 1.0)
                    tout_new = np.where(consumed_now,
                                        tnow + dia_loc / ddot_safe, tout_new)
                    partial = (dnext > 0.0) & (dnext < dia_loc)
                    diff = dia_loc - dnext
                    diff_safe = np.where(diff > 0, diff, 1.0)
                    tout_new = np.where(partial,
                                        tnow + dia_loc / diff_safe * dt,
                                        tout_new)
                    # Low heat-flux → rapid burnout
                    low_q = qdavg <= _MXSTEP
                    tout_new = np.where(low_q,
                                        0.5 * (tnow + tnext), tout_new)

                    ddt_val = np.minimum(tout_new - tnow, dt)
                    ddt_safe = np.where(ddt_val > 0, ddt_val, 1.0)
                    wodot_new = np.where(ddt_val > 0,
                                         (wo[:, kl] - wnext) / ddt_safe,
                                         0.0)

                    # Apply only to ignited cells
                    ddot[:, kl]     = np.where(ignited, ddot_new,
                                               ddot[:, kl])
                    diam_arr[:, kl] = np.where(ignited, dnext,
                                               diam_arr[:, kl])
                    wo[:, kl]       = np.where(ignited, wnext, wo[:, kl])
                    tout[:, kl]     = np.where(ignited, tout_new,
                                               tout[:, kl])
                    wodot[:, kl]    = np.where(ignited, wodot_new,
                                               wodot[:, kl])

                m_remaining &= ~ignited

                # --- drying stage ---
                drying = m_remaining & (tnow >= dryt_val) & (tnow < tlit)

                if drying.any():
                    if l == 0:
                        r_val_d = r0.copy()
                        gi_d = fi_cur + fid_cur
                    elif l == k:
                        r_val_d = r0.copy()
                        gi_d = fi_cur.copy()
                    else:
                        li = l - 1
                        r_val_d = r0 + half_dr * flit[:, li]
                        gi_d = fi_cur + flit[:, li] * fint[:, li]

                    tf_d = _temp_fire(gi_d, r_val_d, tamb_k)
                    dia_d = diam_arr[:, kl]
                    _, hbar_d, _ = _heat_exchange(
                        v_eff, dia_d, tf_d, tamb_k, c_k)
                    dtemp = np.maximum(tf_d - tamb_k, 0.0)
                    qcum[:, kl] = np.where(drying,
                                            qcum[:, kl] + hbar_d * dtemp * dt,
                                            qcum[:, kl])
                    tcum[:, kl] = np.where(drying,
                                            tcum[:, kl] + dtemp * dt,
                                            tcum[:, kl])

                    span = tnext - dryt_val
                    span_safe = np.where(span > 0, span, 1.0)
                    tcum_safe = np.where(tcum[:, kl] > 0,
                                         tcum[:, kl], 1.0)
                    dteff = tcum[:, kl] / span_safe
                    heff = qcum[:, kl] / tcum_safe
                    tfe = tamb_k + dteff

                    can_ign = drying & (tfe > tpig_ki + 10.0)
                    dtlite = np.where(
                        can_ign,
                        _t_ignite(
                            np.full(C, _TPDRY), tpig_ki, tfe,
                            condry_ki, cheat_ki, fmois_ki,
                            dendry_ki, heff, tamb_k),
                        _RINDEF)
                    tign[:, kl] = np.where(drying,
                                            0.5 * (dryt_val + dtlite),
                                            tign[:, kl])

                    # Check if newly ignited this step
                    newly_lit = drying & (tnext > tign[:, kl])
                    if newly_lit.any():
                        ts2 = tchar_ki
                        _, hbar2, _ = _heat_exchange(
                            v_eff, dia_d, tf_d, ts2, c_k)
                        qdot_new = hbar2 * np.maximum(tf_d - ts2, 0.0)
                        qdot[:, kl, 0] = np.where(newly_lit, qdot_new,
                                                   qdot[:, kl, 0])
                        ddot_n = qdot_new * wk
                        dnext_n = np.maximum(
                            0.0, dia_d - (tnext - tign[:, kl]) * ddot_n)
                        dia_safe_n = np.where(dia_d > 0, dia_d, 1.0)
                        wnext_n = np.where(
                            dia_d > 0,
                            wo[:, kl] * (dnext_n / dia_safe_n) ** 2, 0.0)

                        diff_n = dia_d - dnext_n
                        diff_safe_n = np.where(diff_n > 0, diff_n, 1.0)
                        tout_n = np.where(
                            dnext_n == 0.0,
                            np.where(ddot_n > 0,
                                     tnow + dia_d / np.maximum(ddot_n, 1e-30),
                                     _RINDEF),
                            np.where(dnext_n < dia_d,
                                     tnow + dia_d / diff_safe_n * dt,
                                     tout[:, kl]))

                        ddt_n = np.minimum(tout_n - tnow, dt)
                        ddt_safe_n = np.where(ddt_n > 0, ddt_n, 1.0)
                        wodot_n = np.where(
                            (tout_n > tnow) & (ddt_n > 0),
                            (wo[:, kl] - wnext_n) / ddt_safe_n, 0.0)

                        diam_arr[:, kl] = np.where(newly_lit, dnext_n,
                                                    diam_arr[:, kl])
                        wo[:, kl] = np.where(newly_lit, wnext_n, wo[:, kl])
                        tout[:, kl] = np.where(newly_lit, tout_n,
                                                tout[:, kl])
                        wodot[:, kl] = np.where(newly_lit, wodot_n,
                                                 wodot[:, kl])

                m_remaining &= ~drying

                # --- pre-drying stage ---
                pre_dry = m_remaining & (tnow < dryt_val)

                if pre_dry.any():
                    conwet = condry_ki + 4.27e-04 * fmois_ki * dendry_ki

                    if l == 0:
                        r_val_p = r0.copy()
                        gi_p = fi_cur + fid_cur
                    elif l == k:
                        r_val_p = r0.copy()
                        gi_p = fi_cur.copy()
                    else:
                        li = l - 1
                        r_val_p = r0 + half_dr * flit[:, li]
                        gi_p = fi_cur + flit[:, li] * fint[:, li]

                    tf_p = _temp_fire(gi_p, r_val_p, tamb_k)
                    too_cold = tf_p <= (_TPDRY + 10.0)
                    proceed = pre_dry & ~too_cold

                    if proceed.any():
                        dia_p = diam_arr[:, kl]
                        ts_p = 0.5 * (tamb_k + _TPDRY)
                        _, hbar_p, _ = _heat_exchange(
                            v_eff, dia_p, tf_p, ts_p, conwet)
                        dtcum = np.maximum((tf_p - ts_p) * dt, 0.0)
                        tcum[:, kl] = np.where(proceed,
                                                tcum[:, kl] + dtcum,
                                                tcum[:, kl])
                        qcum[:, kl] = np.where(proceed,
                                                qcum[:, kl] + hbar_p * dtcum,
                                                qcum[:, kl])

                        tcum_safe_p = np.where(tcum[:, kl] > 0,
                                               tcum[:, kl], 1.0)
                        he = qcum[:, kl] / tcum_safe_p
                        tnext_safe = np.where(tnext > 0, tnext, 1.0)
                        dtef = tcum[:, kl] / tnext_safe
                        dtef_safe = np.where(dtef > 0, dtef, 1.0)
                        thd_val = (_TPDRY - tamb_k) / dtef_safe
                        too_high = thd_val > 0.9

                        update = proceed & ~too_high
                        if update.any():
                            conwet_safe = np.where(
                                np.abs(conwet) < 1e-30, 1e-30, conwet)
                            biot = he * dia_p / conwet_safe
                            dryt_new = _dry_time(biot, thd_val)
                            cpwet_p = cheat_ki + _CH2O * fmois_ki
                            tdry_new = ((0.5 * dia_p) ** 2
                                        / conwet_safe * cpwet_p
                                        * dendry_ki * dryt_new)
                            tdry[:, kl] = np.where(update, tdry_new,
                                                    tdry[:, kl])

                            # Check if drying starts before tnext
                            early = update & (tdry_new < tnext)
                            if early.any():
                                _, hbar2_p, _ = _heat_exchange(
                                    v_eff, dia_p, tf_p,
                                    np.full(C, _TPDRY), c_k)
                                dqdt_p = hbar2_p * (tf_p - _TPDRY)
                                delt_p = tnext - tdry_new
                                qcum[:, kl] = np.where(early,
                                                        dqdt_p * delt_p,
                                                        qcum[:, kl])
                                tcum[:, kl] = np.where(early,
                                                        (tf_p - _TPDRY) * delt_p,
                                                        tcum[:, kl])

                                can_ign_p = early & (tf_p > tpig_ki + 10.0)
                                if can_ign_p.any():
                                    dtlite_p = _t_ignite(
                                        np.full(C, _TPDRY), tpig_ki, tf_p,
                                        condry_ki, cheat_ki, fmois_ki,
                                        dendry_ki, hbar2_p, tamb_k)
                                    tign[:, kl] = np.where(
                                        can_ign_p,
                                        0.5 * (tdry_new + dtlite_p),
                                        tign[:, kl])
                                    new_ign = can_ign_p & (tnext > tign[:, kl])
                                    qdot[:, kl, 0] = np.where(
                                        new_ign,
                                        hbar2_p * np.maximum(
                                            tf_p - tchar_ki, 0.0),
                                        qdot[:, kl, 0])

        # ---- Update ignited / burned-out fractions ----
        flit[:] = 0.0
        fout[:] = 0.0
        for k in range(1, P + 1):
            ki = k - 1
            for l, kl in by_k[k]:
                is_lit = active & (tnext >= tign[:, kl]) & (tnext <= tout[:, kl])
                flit[:, ki] += np.where(is_lit, xmat[:, kl], 0.0)
                is_out = active & (tnext > tout[:, kl])
                fout[:, ki] += np.where(is_out, xmat[:, kl], 0.0)

        # ---- Advance time, recompute, record ----
        tis = np.where(active, tis + dt, tis)
        fi_cur[:] = _fire_intensity_v()
        _record(step, tis, active)

        # ---- Termination check ----
        done = (fi_cur <= fimin) | (ncalls >= max_timesteps)
        active &= ~done

    # ------------------------------------------------------------------
    # 12. Build summary
    # ------------------------------------------------------------------
    sum_remaining = np.zeros((C, P), dtype=np.float64)
    sum_frac_rem  = np.zeros((C, P), dtype=np.float64)
    sum_t_ign     = np.full((C, P), _RINDEF, dtype=np.float64)
    sum_t_bout    = np.zeros((C, P), dtype=np.float64)

    for k in range(1, P + 1):
        ki = k - 1
        for l, kl in by_k[k]:
            t_ig = tign[:, kl]
            t_ou = tout[:, kl]
            sum_t_ign[:, ki] = np.minimum(sum_t_ign[:, ki], t_ig)
            sum_t_bout[:, ki] = np.maximum(sum_t_bout[:, ki], t_ou)
            sum_remaining[:, ki] += wo[:, kl]

    wdry_safe = np.where(wdry > 0, wdry, 1.0)
    sum_frac_rem = np.where(wdry > 0,
                            sum_remaining / wdry_safe, 0.0)

    return BurnupArrayResult(
        time=out_time,
        wdf=out_wdf,
        ff=out_ff,
        comp_flaming=out_comp_fla,
        comp_smoldering=out_comp_smo,
        n_steps=n_steps,
        summary_wdry=wdry,
        summary_remaining=sum_remaining,
        summary_frac_remaining=sum_frac_rem,
        summary_t_ignite=sum_t_ign,
        summary_t_burnout=sum_t_bout,
        sort_key=sort_key,
        error_code=error_code,
    )

