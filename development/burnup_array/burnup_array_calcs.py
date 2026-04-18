# -*- coding: utf-8 -*-
from __future__ import annotations
"""
burnup_calcs.py – Array-based burnup consumption helpers.

Provides the array-based equivalents of the per-cell burnup operations in
``consumption_calcs.py``:

* :func:`run_burnup` – the array analogue of ``_run_burnup_cell``.
  Accepts multi-cell fuel loadings/moistures as ``(C,)`` arrays per size
  class, builds the ``(C, P)`` particle arrays, clips fire-environment
  inputs, calls :func:`~burnup.burnup`, and extracts per-class
  consumption, durations, and error/adjustment codes for all cells at once.

* :func:`extract_burnup_consumption` – array analogue of
  ``_extract_burnup_consumption``.

* :func:`burnup_durations` – array analogue of ``_burnup_durations``.

All other consumption functions (``consm_canopy``, ``consm_duff``,
``consm_herb``, ``consm_litter``, ``consm_mineral_soil``, ``consm_shrub``)
are already vectorized in ``consumption_calcs.py`` and do not need array
wrappers.

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .burnup_array_kernel import (
    _FIRE_BOUNDS,
    BurnupArrayResult,
    burnup as _burnup,
)

# ---------------------------------------------------------------------------
# Physical constants  (mirrors consumption_calcs.py)
# ---------------------------------------------------------------------------
_DENSITY_SOUND: float = 513.0
_DENSITY_ROTTEN: float = 224.0
_SOUND_TPIG: float = 327.0
_ROTTEN_TPIG: float = 302.0
_TCHAR: float = 377.0
_HTVAL: float = 1.86e7

# Canonical class ordering  (sound + rotten 1000-hr)
_CLASS_ORDER_ALL = [
    'litter', 'dw1', 'dw10', 'dw100',
    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    'dwk_3_6_r', 'dwk_6_9_r', 'dwk_9_20_r', 'dwk_20_r',
]

# SAV defaults per size class  (sound keys only; rotten maps to sound SAV)
_SAV_DEFAULTS: Dict[str, float] = {
    'litter': 8200.0, 'dw1': 1480.0, 'dw10': 394.0, 'dw100': 105.0,
    'dwk_3_6': 39.4, 'dwk_6_9': 21.9, 'dwk_9_20': 12.7, 'dwk_20': 5.91,
}

# Rotten key → corresponding sound key
_ROTTEN_TO_SOUND = {
    'dwk_3_6_r': 'dwk_3_6',
    'dwk_6_9_r': 'dwk_6_9',
    'dwk_9_20_r': 'dwk_9_20',
    'dwk_20_r': 'dwk_20',
}

P_TOTAL = len(_CLASS_ORDER_ALL)  # 12 particle columns


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class BurnupConsumptionResult:
    """Per-class consumption output from the array burnup pipeline.

    All arrays have shape ``(C,)`` unless noted otherwise.

    :param consumed: ``Dict[str, (C,)]`` — consumed mass (kg/m²) per class.
    :param flaming: ``Dict[str, (C,)]`` — flaming partition (kg/m²) per class.
    :param smoldering: ``Dict[str, (C,)]`` — smoldering partition (kg/m²) per class.
    :param frac_remaining: ``Dict[str, (C,)]`` — remaining fraction per class.
    :param fla_dur: ``(C,)`` — flaming duration (s).
    :param smo_dur: ``(C,)`` — smoldering duration (s).
    :param burnup_limit_adjust: ``(C,)`` int — adjustment code (0 = none).
    :param burnup_error: ``(C,)`` int — error code (0 = success).
    :param raw: The underlying :class:`BurnupArrayResult` (for advanced use).
    """
    consumed: Dict[str, np.ndarray]
    flaming: Dict[str, np.ndarray]
    smoldering: Dict[str, np.ndarray]
    frac_remaining: Dict[str, np.ndarray]
    fla_dur: np.ndarray
    smo_dur: np.ndarray
    burnup_limit_adjust: np.ndarray
    burnup_error: np.ndarray
    raw: BurnupArrayResult


# ---------------------------------------------------------------------------
# Array-based durations
# ---------------------------------------------------------------------------

def burnup_durations(
    ar: BurnupArrayResult,
    dt: float,
    fla_threshold: float = 1e-05,
    smo_threshold: float = 1e-05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive flaming and smoldering durations from array burnup output.

    Vectorized equivalent of ``_burnup_durations``.  For each cell, finds the
    **last** timestep where per-step flaming (or smoldering) mass exceeds
    the threshold.

    :param ar: Result from :func:`burnup`.
    :param dt: Time-step (s) used for the simulation.
    :param fla_threshold: Minimum per-step flaming mass to count.
    :param smo_threshold: Minimum per-step smoldering mass to count.
    :return: ``(fla_dur, smo_dur)`` each shape ``(C,)``, in seconds.
        ``NaN`` where the phase never occurred.
    """
    C = ar.time.shape[0]
    # comp_flaming: (C, T, P)  — sum across P → per-step flaming total
    # comp_smoldering: (C, T, P+1) — sum across P+1 → per-step smoldering total
    # These are mass-loss rates; multiply by dt for mass per step
    step_fla = np.nansum(ar.comp_flaming, axis=2) * dt   # (C, T)
    step_smo = np.nansum(ar.comp_smoldering, axis=2) * dt  # (C, T)

    fla_dur = np.full(C, np.nan, dtype=np.float64)
    smo_dur = np.full(C, np.nan, dtype=np.float64)

    # For each cell, find the last time where threshold is exceeded
    fla_active = step_fla > fla_threshold  # (C, T) bool
    smo_active = step_smo > smo_threshold

    for c in range(C):
        n = ar.n_steps[c]
        if n == 0:
            continue
        # Find last True index
        fla_idxs = np.where(fla_active[c, :n])[0]
        if len(fla_idxs) > 0:
            fla_dur[c] = ar.time[c, fla_idxs[-1]]
        else:
            fla_dur[c] = 0.0
        smo_idxs = np.where(smo_active[c, :n])[0]
        if len(smo_idxs) > 0:
            smo_dur[c] = ar.time[c, smo_idxs[-1]]
        else:
            smo_dur[c] = 0.0

    return fla_dur, smo_dur


# ---------------------------------------------------------------------------
# Array-based consumption extraction
# ---------------------------------------------------------------------------

def extract_burnup_consumption(
    ar: BurnupArrayResult,
    class_keys: list,
    dt: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
           Dict[str, np.ndarray]]:
    """Extract per-class consumption from array burnup results.

    Vectorized equivalent of ``_extract_burnup_consumption``.

    :param ar: Result from :func:`burnup`.
    :param class_keys: List of class keys in the order they appear in the
        particle columns (length ``P``).
    :param dt: Time-step (s).
    :return: ``(consumed, flaming, smoldering, frac_remaining)`` — each is a
        dict keyed by class name with ``(C,)`` arrays.
    """
    C = ar.time.shape[0]
    P = len(class_keys)

    # Accumulate flaming/smoldering per component across time steps
    # comp_flaming: (C, T, P), comp_smoldering: (C, T, P+1)
    # Sum across T axis, multiply by dt → (C, P)
    comp_fla = np.nansum(ar.comp_flaming, axis=1) * dt   # (C, P)
    comp_smo = np.nansum(ar.comp_smoldering[:, :, :P], axis=1) * dt  # (C, P)

    consumed_d: Dict[str, np.ndarray] = {}
    flaming_d: Dict[str, np.ndarray] = {}
    smoldering_d: Dict[str, np.ndarray] = {}
    frac_remaining_d: Dict[str, np.ndarray] = {}

    for i, key in enumerate(class_keys):
        frac_rem = ar.summary_frac_remaining[:, i]  # (C,)
        wdry_i = ar.summary_wdry[:, i]              # (C,)
        cons = wdry_i * (1.0 - frac_rem)

        # Partition into flaming/smoldering
        total_partitioned = comp_fla[:, i] + comp_smo[:, i]
        has_partition = total_partitioned > 1e-12
        scale = np.where(has_partition,
                         cons / np.maximum(total_partitioned, 1e-30), 0.0)
        fla_mass = np.where(has_partition, comp_fla[:, i] * scale, 0.0)
        smo_mass = np.where(has_partition, comp_smo[:, i] * scale, cons)

        consumed_d[key] = cons
        flaming_d[key] = fla_mass
        smoldering_d[key] = smo_mass
        frac_remaining_d[key] = frac_rem

    return consumed_d, flaming_d, smoldering_d, frac_remaining_d


# ---------------------------------------------------------------------------
# Main burnup pipeline
# ---------------------------------------------------------------------------

def run_burnup_array(
    *,
    # Per-class fuel loadings — each (C,) in kg/m²  (0 if absent)
    litter: np.ndarray,
    dw1: np.ndarray,
    dw10: np.ndarray,
    dw100: np.ndarray,
    dwk_3_6: np.ndarray,
    dwk_6_9: np.ndarray,
    dwk_9_20: np.ndarray,
    dwk_20: np.ndarray,
    dwk_3_6_r: Optional[np.ndarray] = None,
    dwk_6_9_r: Optional[np.ndarray] = None,
    dwk_9_20_r: Optional[np.ndarray] = None,
    dwk_20_r: Optional[np.ndarray] = None,
    # Per-class fuel moistures — each (C,) fraction
    litter_moist: np.ndarray,
    dw1_moist: np.ndarray,
    dw10_moist: np.ndarray,
    dw100_moist: np.ndarray,
    dwk_moist: np.ndarray,
    dwk_r_moist: Optional[np.ndarray] = None,
    # Fire environment — each (C,)
    intensity: np.ndarray,
    ig_time: np.ndarray,
    windspeed: np.ndarray,
    depth: np.ndarray,
    ambient_temp: np.ndarray,
    # Duff — each (C,)
    duff_loading: np.ndarray,
    duff_moist_frac: np.ndarray,
    duff_pct_consumed: Optional[np.ndarray] = None,
    # Per-class densities — each (C,) or scalar
    density_sound: float = _DENSITY_SOUND,
    density_rotten: float = _DENSITY_ROTTEN,
    # Burnup parameters
    r0: float = 1.83,
    dr: float = 0.4,
    dt: float = 15.0,
    max_timesteps: int = 3000,
    fint_switch: float = 15.0,
    validate: bool = True,
) -> BurnupConsumptionResult:
    """Run the burnup model for C cells simultaneously via array kernel.

    This is the array analogue of ``_run_burnup_cell``.  It accepts per-class
    fuel loadings and moistures as ``(C,)`` arrays, assembles the ``(C, P)``
    particle arrays (with P=12 columns: 8 sound + 4 rotten), clips
    fire-environment inputs (tracking adjustment codes), runs the vectorized
    burnup simulation, and extracts per-class consumption.

    :param litter: Litter loading (kg/m²), ``(C,)``.
    :param dw1: 1-hr loading (kg/m²), ``(C,)``.
    :param dw10: 10-hr loading (kg/m²), ``(C,)``.
    :param dw100: 100-hr loading (kg/m²), ``(C,)``.
    :param dwk_3_6: 1000-hr sound 3–6″ loading (kg/m²), ``(C,)``.
    :param dwk_6_9: 1000-hr sound 6–9″ loading (kg/m²), ``(C,)``.
    :param dwk_9_20: 1000-hr sound 9–20″ loading (kg/m²), ``(C,)``.
    :param dwk_20: 1000-hr sound ≥20″ loading (kg/m²), ``(C,)``.
    :param dwk_3_6_r: 1000-hr rotten 3–6″ loading (kg/m²), ``(C,)``.  Optional.
    :param dwk_6_9_r: 1000-hr rotten 6–9″ loading (kg/m²), ``(C,)``.  Optional.
    :param dwk_9_20_r: 1000-hr rotten 9–20″ loading (kg/m²), ``(C,)``.  Optional.
    :param dwk_20_r: 1000-hr rotten ≥20″ loading (kg/m²), ``(C,)``.  Optional.
    :param litter_moist: Litter moisture (fraction), ``(C,)``.
    :param dw1_moist: 1-hr moisture (fraction), ``(C,)``.
    :param dw10_moist: 10-hr moisture (fraction), ``(C,)``.
    :param dw100_moist: 100-hr moisture (fraction), ``(C,)``.
    :param dwk_moist: 1000-hr sound moisture (fraction), ``(C,)``.
    :param dwk_r_moist: 1000-hr rotten moisture (fraction), ``(C,)``.  Optional.
    :param intensity: Fire intensity (kW/m²), ``(C,)``.
    :param ig_time: Residence time (s), ``(C,)``.
    :param windspeed: Windspeed at fuelbed top (m/s), ``(C,)``.
    :param depth: Fuel bed depth (m), ``(C,)``.
    :param ambient_temp: Ambient temperature (°C), ``(C,)``.
    :param duff_loading: Duff loading (kg/m²), ``(C,)``.
    :param duff_moist_frac: Duff moisture (fraction), ``(C,)``.
    :param duff_pct_consumed: Duff percent consumed (0–100), ``(C,)``.
    :param density_sound: Sound wood density (kg/m³). Default 513.
    :param density_rotten: Rotten wood density (kg/m³). Default 224.
    :param r0: Minimum mixing parameter. Default 1.83.
    :param dr: Mixing parameter range. Default 0.4.
    :param dt: Time-step (s). Default 15.
    :param max_timesteps: Maximum time steps. Default 3000.
    :param fint_switch: Flaming/smoldering threshold (kW/m²). Default 15.
    :param validate: Validate inputs. Default True.
    :return: :class:`BurnupConsumptionResult`.
    """
    # ------------------------------------------------------------------
    # 0. Ensure 1-D arrays
    # ------------------------------------------------------------------
    litter     = np.atleast_1d(np.asarray(litter,     dtype=np.float64))
    dw1        = np.atleast_1d(np.asarray(dw1,        dtype=np.float64))
    dw10       = np.atleast_1d(np.asarray(dw10,       dtype=np.float64))
    dw100      = np.atleast_1d(np.asarray(dw100,      dtype=np.float64))
    dwk_3_6    = np.atleast_1d(np.asarray(dwk_3_6,    dtype=np.float64))
    dwk_6_9    = np.atleast_1d(np.asarray(dwk_6_9,    dtype=np.float64))
    dwk_9_20   = np.atleast_1d(np.asarray(dwk_9_20,   dtype=np.float64))
    dwk_20     = np.atleast_1d(np.asarray(dwk_20,     dtype=np.float64))
    C = len(litter)

    def _ensure(arr, default=0.0):
        if arr is None:
            return np.full(C, default, dtype=np.float64)
        return np.atleast_1d(np.asarray(arr, dtype=np.float64))

    dwk_3_6_r  = _ensure(dwk_3_6_r)
    dwk_6_9_r  = _ensure(dwk_6_9_r)
    dwk_9_20_r = _ensure(dwk_9_20_r)
    dwk_20_r   = _ensure(dwk_20_r)

    litter_moist = np.atleast_1d(np.asarray(litter_moist, dtype=np.float64))
    dw1_moist    = np.atleast_1d(np.asarray(dw1_moist,    dtype=np.float64))
    dw10_moist   = np.atleast_1d(np.asarray(dw10_moist,   dtype=np.float64))
    dw100_moist  = np.atleast_1d(np.asarray(dw100_moist,  dtype=np.float64))
    dwk_moist    = np.atleast_1d(np.asarray(dwk_moist,    dtype=np.float64))
    dwk_r_moist  = _ensure(dwk_r_moist, 0.10)

    intensity    = np.atleast_1d(np.asarray(intensity,    dtype=np.float64))
    ig_time      = np.atleast_1d(np.asarray(ig_time,      dtype=np.float64))
    windspeed    = np.atleast_1d(np.asarray(windspeed,    dtype=np.float64))
    depth        = np.atleast_1d(np.asarray(depth,        dtype=np.float64))
    ambient_temp = np.atleast_1d(np.asarray(ambient_temp, dtype=np.float64))
    duff_loading = np.atleast_1d(np.asarray(duff_loading, dtype=np.float64))
    duff_moist_frac = np.atleast_1d(np.asarray(duff_moist_frac, dtype=np.float64))
    duff_pct_consumed_arr = _ensure(duff_pct_consumed, -1.0)

    # ------------------------------------------------------------------
    # 1. Clip fire-environment inputs & track adjustment codes
    # ------------------------------------------------------------------
    adj = np.zeros(C, dtype=np.int64)

    # Helper: clip and record adjustment digit
    def _clip_upper(arr, bound_key, code_digit):
        nonlocal adj
        _, hi, _ = _FIRE_BOUNDS[bound_key]
        mask = arr > hi
        arr = np.where(mask, hi, arr)
        adj = np.where(mask, adj * 10 + code_digit, adj)
        return arr

    def _clip_lower(arr, bound_key, code_digit):
        nonlocal adj
        lo, _, _ = _FIRE_BOUNDS[bound_key]
        mask = arr < lo
        arr = np.where(mask, lo, arr)
        adj = np.where(mask, adj * 10 + code_digit, adj)
        return arr

    def _clip_both(arr, bound_key, code_digit):
        nonlocal adj
        lo, hi, _ = _FIRE_BOUNDS[bound_key]
        mask_lo = arr < lo
        mask_hi = arr > hi
        arr = np.where(mask_lo, lo, np.where(mask_hi, hi, arr))
        adj = np.where(mask_lo | mask_hi, adj * 10 + code_digit, adj)
        return arr

    intensity    = _clip_upper(intensity, 'fistart', 1)
    ig_time      = _clip_upper(ig_time, 'ti', 2)
    windspeed    = _clip_upper(windspeed, 'u', 3)
    depth        = _clip_both(depth, 'd', 4)
    ambient_temp = _clip_upper(ambient_temp, 'tamb_c', 5)

    # Duff moisture: clip to min when duff present
    _dfm_lo, _, _ = _FIRE_BOUNDS['dfm']
    dfm_clip = (duff_loading > 0.0) & (duff_moist_frac < _dfm_lo)
    duff_moist_frac = np.where(dfm_clip, _dfm_lo, duff_moist_frac)
    adj = np.where(dfm_clip, adj * 10 + 6, adj)

    burnup_limit_adjust = adj.astype(np.int64)

    # ------------------------------------------------------------------
    # 1b. Pre-flight lower-bound checks  (match scalar _run_burnup_cell)
    # ------------------------------------------------------------------
    # These are hard errors that cannot be fixed by clipping — the burnup
    # model requires a minimum fire intensity, residence time, etc.
    # Recording them here ensures consistent error codes even when
    # validate=False is passed to burnup.
    burnup_error_pre = np.zeros(C, dtype=np.int64)

    _fi_lo, _, _ = _FIRE_BOUNDS['fistart']
    burnup_error_pre = np.where(
        (burnup_error_pre == 0) & (intensity < _fi_lo), 10, burnup_error_pre)

    _ti_lo, _, _ = _FIRE_BOUNDS['ti']
    burnup_error_pre = np.where(
        (burnup_error_pre == 0) & (ig_time < _ti_lo), 11, burnup_error_pre)

    _u_lo, _, _ = _FIRE_BOUNDS['u']
    burnup_error_pre = np.where(
        (burnup_error_pre == 0) & (windspeed < _u_lo), 12, burnup_error_pre)

    _tamb_lo, _, _ = _FIRE_BOUNDS['tamb_c']
    burnup_error_pre = np.where(
        (burnup_error_pre == 0) & (ambient_temp < _tamb_lo), 13, burnup_error_pre)

    _, _dfm_hi, _ = _FIRE_BOUNDS['dfm']
    burnup_error_pre = np.where(
        (burnup_error_pre == 0) & (duff_loading > 0.0) & (duff_moist_frac > _dfm_hi),
        14, burnup_error_pre)

    # ------------------------------------------------------------------
    # 2. Assemble (C, P) particle arrays
    # ------------------------------------------------------------------
    # Columns follow _CLASS_ORDER_ALL:
    # 0=litter 1=dw1 2=dw10 3=dw100 4=dwk_3_6 5=dwk_6_9 6=dwk_9_20 7=dwk_20
    # 8=dwk_3_6_r 9=dwk_6_9_r 10=dwk_9_20_r 11=dwk_20_r
    P = P_TOTAL

    wdry_2d = np.column_stack([
        litter, dw1, dw10, dw100,
        dwk_3_6, dwk_6_9, dwk_9_20, dwk_20,
        dwk_3_6_r, dwk_6_9_r, dwk_9_20_r, dwk_20_r,
    ])  # (C, 12)

    # Moisture: per-class
    fmois_2d = np.column_stack([
        litter_moist, dw1_moist, dw10_moist, dw100_moist,
        dwk_moist, dwk_moist, dwk_moist, dwk_moist,
        dwk_r_moist, dwk_r_moist, dwk_r_moist, dwk_r_moist,
    ])  # (C, 12)
    fmois_2d = np.maximum(fmois_2d, 0.02)

    # SAV (same for all cells; rotten keys map to their sound SAV)
    sigma_vals = [_SAV_DEFAULTS[k] if k in _SAV_DEFAULTS
                  else _SAV_DEFAULTS[_ROTTEN_TO_SOUND[k]]
                  for k in _CLASS_ORDER_ALL]
    sigma_2d = np.tile(np.array(sigma_vals, dtype=np.float64), (C, 1))

    # Density: sound for first 8, rotten for last 4
    dendry_2d = np.full((C, P), density_sound, dtype=np.float64)
    dendry_2d[:, 8:] = density_rotten

    # Ignition temperature: sound for first 8, rotten for last 4
    tpig_2d = np.full((C, P), _SOUND_TPIG, dtype=np.float64)
    tpig_2d[:, 8:] = _ROTTEN_TPIG

    # Uniform properties
    htval_2d  = np.full((C, P), _HTVAL,  dtype=np.float64)
    cheat_2d  = np.full((C, P), 2750.0,  dtype=np.float64)
    condry_2d = np.full((C, P), 0.133,   dtype=np.float64)
    tchar_2d  = np.full((C, P), _TCHAR,  dtype=np.float64)
    ash_2d    = np.full((C, P), 0.05,    dtype=np.float64)

    # Broadcast r0/dr to (C,)
    r0_arr = np.full(C, r0, dtype=np.float64)
    dr_arr = np.full(C, dr, dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Run vectorized burnup
    # ------------------------------------------------------------------
    ar = _burnup(
        wdry=wdry_2d,
        htval=htval_2d,
        fmois=fmois_2d,
        dendry=dendry_2d,
        sigma=sigma_2d,
        cheat=cheat_2d,
        condry=condry_2d,
        tpig=tpig_2d,
        tchar=tchar_2d,
        ash=ash_2d,
        fi=intensity,
        ti=ig_time,
        u=windspeed,
        d=depth,
        tamb=ambient_temp,
        r0=r0_arr,
        dr=dr_arr,
        dt=dt,
        max_timesteps=max_timesteps,
        wdf=duff_loading,
        dfm=duff_moist_frac,
        duff_pct_consumed=duff_pct_consumed_arr,
        fint_switch=fint_switch,
        validate=validate,
    )

    # Combine error codes: pre-flight lower-bound errors take priority
    burnup_error = np.where(burnup_error_pre != 0,
                            burnup_error_pre, ar.error_code.copy())

    # ------------------------------------------------------------------
    # 4. Map sorted particle columns back to class keys
    # ------------------------------------------------------------------
    # burnup sorts particles per cell.  sort_key (C, P) maps
    # sorted index → original index.  We need to unsort the summary arrays.
    # sort_key[c, sorted_j] = original column index
    # To get original column i, we need the inverse: for each c, find j
    # such that sort_key[c, j] == i.
    inv_sort = np.argsort(ar.sort_key, axis=1)  # (C, P)

    # Unsort summary arrays back to _CLASS_ORDER_ALL order
    def _unsort(arr_2d):
        """Reorder (C, P) from sorted order back to original column order."""
        return np.take_along_axis(arr_2d, inv_sort, axis=1)

    frac_rem_orig = _unsort(ar.summary_frac_remaining)  # (C, P)
    wdry_orig = _unsort(ar.summary_wdry)                # (C, P)

    # ------------------------------------------------------------------
    # 5. Extract per-class consumption
    # ------------------------------------------------------------------
    # Accumulate flaming/smoldering per component across time (sorted order)
    # comp_flaming: (C, T, P), comp_smoldering: (C, T, P+1)
    comp_fla_sorted = np.nansum(ar.comp_flaming, axis=1) * dt   # (C, P)
    comp_smo_sorted = np.nansum(ar.comp_smoldering[:, :, :P], axis=1) * dt  # (C, P)

    # Unsort to original column order
    comp_fla_orig = _unsort(comp_fla_sorted)  # (C, P)
    comp_smo_orig = _unsort(comp_smo_sorted)  # (C, P)

    consumed_d: Dict[str, np.ndarray] = {}
    flaming_d: Dict[str, np.ndarray] = {}
    smoldering_d: Dict[str, np.ndarray] = {}
    frac_remaining_d: Dict[str, np.ndarray] = {}


    for i, key in enumerate(_CLASS_ORDER_ALL):
        frac_rem = frac_rem_orig[:, i]  # (C,)
        cons = wdry_orig[:, i] * (1.0 - frac_rem)

        # Partition into flaming/smoldering
        total_part = comp_fla_orig[:, i] + comp_smo_orig[:, i]
        has_part = total_part > 1e-12
        scale = np.where(has_part,
                         cons / np.maximum(total_part, 1e-30), 0.0)
        fla_mass = np.where(has_part, comp_fla_orig[:, i] * scale, 0.0)
        smo_mass = np.where(has_part, comp_smo_orig[:, i] * scale, cons)

        consumed_d[key] = cons
        flaming_d[key] = fla_mass
        smoldering_d[key] = smo_mass
        frac_remaining_d[key] = frac_rem

    # ------------------------------------------------------------------
    # 6. Compute durations
    # ------------------------------------------------------------------
    fla_dur, smo_dur = burnup_durations(ar, dt)

    # Zero out results for cells with errors
    err_mask = burnup_error != 0
    if np.any(err_mask):
        for key in _CLASS_ORDER_ALL:
            consumed_d[key][err_mask] = 0.0
            flaming_d[key][err_mask] = 0.0
            smoldering_d[key][err_mask] = 0.0
            frac_remaining_d[key][err_mask] = 1.0
        fla_dur[err_mask] = 0.0
        smo_dur[err_mask] = 0.0

    return BurnupConsumptionResult(
        consumed=consumed_d,
        flaming=flaming_d,
        smoldering=smoldering_d,
        frac_remaining=frac_remaining_d,
        fla_dur=fla_dur,
        smo_dur=smo_dur,
        burnup_limit_adjust=burnup_limit_adjust,
        burnup_error=burnup_error,
        raw=ar,
    )


def _run_burnup_array_chunk(chunk_kwargs: dict) -> dict:
    """Run one array-burnup chunk and return compact outputs."""
    start = int(chunk_kwargs['start'])
    end = int(chunk_kwargs['end'])
    kwargs = dict(chunk_kwargs)
    kwargs.pop('start', None)
    kwargs.pop('end', None)
    out = run_burnup_array(**kwargs)
    return {
        'start': start,
        'end': end,
        'consumed': out.consumed,
        'flaming': out.flaming,
        'smoldering': out.smoldering,
        'fla_dur': out.fla_dur,
        'smo_dur': out.smo_dur,
        'burnup_limit_adjust': out.burnup_limit_adjust,
        'burnup_error': out.burnup_error,
    }

