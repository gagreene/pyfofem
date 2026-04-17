# -*- coding: utf-8 -*-
"""
burnup_calcs.py – Burnup model helpers and wrappers split from consumption_calcs.py.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup as _burnup,
    _FIRE_BOUNDS,
    _BURNUP_LIMIT_ADJUST,
    _BURNUP_LIMIT_ERROR,
)
from ._component_helpers import _is_scalar, _maybe_scalar, _to_str_arr

# --- Burnup physical constants and helpers ---
_DENSITY_SOUND: float = 513.0
_DENSITY_ROTTEN: float = 224.0
_SOUND_TPIG: float = 327.0
_ROTTEN_TPIG: float = 302.0
_TCHAR: float = 377.0
_HTVAL: float = 1.86e7
_TPAC_TO_KGPM2: float = 1.0 / 4.4609
_KGPM2_TO_TPAC: float = 4.4609
_IN_TO_CM: float = 2.54

_SAV_DEFAULTS: Dict[str, float] = {
    'litter': 8200.0, 'dw1': 1480.0, 'dw10': 394.0, 'dw100': 105.0,
    'dwk_3_6': 39.4,  'dwk_6_9': 21.9, 'dwk_9_20': 12.7, 'dwk_20': 5.91,
}
_CLASS_ORDER_ALL = [
    'litter', 'dw1', 'dw10', 'dw100',
    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    'dwk_3_6_r', 'dwk_6_9_r', 'dwk_9_20_r', 'dwk_20_r',
]

def _burnup_durations(
    results: List[BurnResult],
    fla_threshold: float = 1e-05,
    smo_threshold: float = 1e-05,
) -> Tuple[float, float]:
    """Derive flaming and smoldering durations from burnup time-series."""
    if not results:
        return float('nan'), float('nan')
    fla_dur = 0.0
    smo_dur = 0.0
    for r in results:
        if r.comp_flaming is not None:
            step_fla = sum(r.comp_flaming)
        else:
            step_fla = r.ff
        if r.comp_smoldering is not None:
            step_smo = sum(r.comp_smoldering)
        else:
            step_smo = (1.0 - r.ff)
        if step_fla > fla_threshold:
            fla_dur = r.time
        if step_smo > smo_threshold:
            smo_dur = r.time
    return fla_dur, smo_dur

def _extract_burnup_consumption(
    results: List[BurnResult],
    summary: List[BurnSummaryRow],
    class_order: List[str],
    dt: float,
) -> Dict[str, Dict[str, float]]:
    """Extract per-fuel-class consumption and flaming/smoldering partition from burnup."""
    n_comp = len(class_order)
    out: Dict[str, Dict[str, float]] = {}
    comp_fla = [0.0] * n_comp
    comp_smo = [0.0] * n_comp
    prev_time = 0.0
    for r in results:
        # The first record covers [0, ti] (ti = r.time); subsequent records cover dt seconds.
        interval = r.time - prev_time
        prev_time = r.time
        if r.comp_flaming is not None:
            for i in range(n_comp):
                comp_fla[i] += r.comp_flaming[i] * interval
        if r.comp_smoldering is not None:
            for i in range(n_comp):
                comp_smo[i] += r.comp_smoldering[i] * interval
    for i, key in enumerate(class_order):
        row = summary[i]
        consumed = row.wdry * (1.0 - row.frac_remaining)
        total_partitioned = comp_fla[i] + comp_smo[i]
        if total_partitioned > 1e-12:
            scale = consumed / total_partitioned if total_partitioned > 0 else 0.0
            fla_mass = comp_fla[i] * scale
            smo_mass = comp_smo[i] * scale
        else:
            fla_mass = 0.0
            smo_mass = consumed
        out[key] = {
            'consumed': consumed,
            'flaming': fla_mass,
            'smoldering': smo_mass,
            'frac_remaining': row.frac_remaining,
        }
    return out

def _run_burnup_cell(ckw: dict):
    """Run the burnup model for a single spatial cell."""
    fl     = ckw['fuel_loadings_bu']
    fm     = ckw['fuel_moistures_bu']
    rk     = ckw['rotten_keys']
    dm_map = ckw['density_map']
    intensity = ckw['intensity_kw']
    ig     = ckw['frt_s']
    ws     = ckw['ws']
    fbd    = ckw['fb_depth']
    at     = ckw['amb_temp']
    duf_si = ckw['duf_loading_si']
    duf_mf = ckw['duf_moist_frac']
    duf_pct = ckw.get('duf_pct_consumed', -1.0)
    hsf_si  = ckw.get('hsf_consumed_si', 0.0)
    brafol_si = ckw.get('brafol_consumed_si', 0.0)
    dt     = ckw['burnup_dt']
    bkw    = ckw['bkw']
    adj_codes = []
    _fi_lo, _fi_hi, _ = _FIRE_BOUNDS['fistart']
    if intensity > _fi_hi:
        intensity = _fi_hi
        adj_codes.append(1)
    _ti_lo, _ti_hi, _ = _FIRE_BOUNDS['ti']
    if ig > _ti_hi:
        ig = _ti_hi
        adj_codes.append(2)
    _u_lo, _u_hi, _ = _FIRE_BOUNDS['u']
    if ws > _u_hi:
        ws = _u_hi
        adj_codes.append(3)
    _d_lo, _d_hi, _ = _FIRE_BOUNDS['d']
    if fbd < _d_lo:
        fbd = _d_lo
        adj_codes.append(4)
    elif fbd > _d_hi:
        fbd = _d_hi
        adj_codes.append(4)
    _t_lo, _t_hi, _ = _FIRE_BOUNDS['tamb_c']
    if at > _t_hi:
        at = _t_hi
        adj_codes.append(5)
    _dfm_lo, _dfm_hi, _ = _FIRE_BOUNDS['dfm']
    if duf_si > 0.0 and duf_mf < _dfm_lo:
        duf_mf = _dfm_lo
        adj_codes.append(6)
    if adj_codes:
        burnup_limit_adjust = int(''.join(str(c) for c in adj_codes))
    else:
        burnup_limit_adjust = 0
    particles: List[FuelParticle] = []
    co: List[str] = []
    for key in _CLASS_ORDER_ALL:
        ld = fl.get(key, 0.0)
        if ld <= 0.0:
            continue
        mo = fm.get(key, 0.10)
        sv = _SAV_DEFAULTS[rk[key]] if key in rk else _SAV_DEFAULTS.get(key, 39.4)
        d  = dm_map.get(key, _DENSITY_SOUND)
        is_rotten = key in rk
        particles.append(FuelParticle(
            wdry=ld, htval=_HTVAL, fmois=mo, dendry=d, sigma=sv,
            cheat=2750.0, condry=0.133,
            tpig=_ROTTEN_TPIG if is_rotten else _SOUND_TPIG,
            tchar=_TCHAR, ash=0.05,
        ))
        co.append(key)
    if not particles:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 90}
    _fi_lo2, _, _ = _FIRE_BOUNDS['fistart']
    if intensity < _fi_lo2:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 10}
    _ti_lo2, _, _ = _FIRE_BOUNDS['ti']
    if ig < _ti_lo2:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 11}
    _u_lo2, _, _ = _FIRE_BOUNDS['u']
    if ws < _u_lo2:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 12}
    _t_lo2, _, _ = _FIRE_BOUNDS['tamb_c']
    if at < _t_lo2:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 13}
    _dfm_lo2, _dfm_hi2, _ = _FIRE_BOUNDS['dfm']
    if duf_si > 0.0 and duf_mf > _dfm_hi2:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 14}
    try:
        res, summ = _burnup(
            particles=particles, fi=intensity, ti=ig, u=ws, d=fbd,
            tamb=at, r0=bkw['r0'], dr=bkw['dr'], dt=dt,
            ntimes=bkw['max_times'], wdf=duf_si, dfm=duf_mf,
            duff_pct_consumed=duf_pct,
            fint_switch=bkw['fint_switch'], validate=bkw['validate'],
            hsf_consumed=hsf_si, brafol_consumed=brafol_si,
        )
        bcon = _extract_burnup_consumption(res, summ, co, dt)
        fla_dur, smo_dur = _burnup_durations(res)
        burnup_times_s = [float(r.time) for r in res]
        burnup_fi_wl = [float(r.fi_wl or 0.0) for r in res]
        burnup_fi_hs = [float(r.fi_hs or 0.0) for r in res]
        return {
            'bcon': bcon,
            'fla_dur': fla_dur,
            'smo_dur': smo_dur,
            'burnup_times_s': burnup_times_s,
            'burnup_fi_wl': burnup_fi_wl,
            'burnup_fi_hs': burnup_fi_hs,
            'class_order': co,
            'burnup_limit_adjust': burnup_limit_adjust,
            'burnup_error': 0,
        }
    except BurnupValidationError as exc:
        msg = str(exc).lower()
        _FUEL_ATTR_TO_CODE = {
            'dry loading': 20, 'ash content': 21, 'heat content': 22,
            'fuel moisture': 23, 'dry mass density': 24, 'sav': 25,
            'heat capacity': 26, 'thermal conductivity': 27,
            'ignition temperature': 28, 'char temperature': 29,
        }
        err_code = 99
        if 'cannot dry fuel' in msg:
            err_code = 15
        elif 'no fuel ignited' in msg:
            err_code = 16
        elif 'ntimes' in msg:
            err_code = 91
        elif 'fire intensity' in msg or 'igniting fire' in msg:
            err_code = 10
        elif 'residence time' in msg:
            err_code = 11
        elif 'windspeed' in msg:
            err_code = 12
        elif 'ambient temperature' in msg:
            err_code = 13
        elif 'duff moisture' in msg:
            err_code = 14
        else:
            for attr_fragment, code in _FUEL_ATTR_TO_CODE.items():
                if attr_fragment in msg:
                    err_code = code
                    break
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': err_code}
    except Exception:
        return {'burnup_limit_adjust': burnup_limit_adjust, 'burnup_error': 99}

def gen_burnup_in_file(
        out_brn_path=None,
        max_times=3000,
        intensity=50.0,
        ig_time=60.0,
        windspeed=0.0,
        depth=0.3,
        ambient_temp=21.0,
        r0=1.83,
        dr=0.4,
        timestep=15.0,
        surat_lit=8200,
        surat_dw1=1480,
        surat_dw10=394,
        surat_dw100=105,
        surat_dwk_3_6=39.4,
        surat_dwk_6_9=21.9,
        surat_dwk_9_20=12.7,
        surat_dwk_20=5.91
) -> None:
    """Function generates a Burnup-in.brn file from the input parameters, and saves it at the out_brn_path location"""
    if out_brn_path is None:
        raise Exception('No output path specified for Burnup-in.brn file')
    max_times = max(1, min(max_times, 100000))
    intensity = max(_FIRE_BOUNDS['fistart'][0], min(intensity, _FIRE_BOUNDS['fistart'][1]))
    ig_time = max(_FIRE_BOUNDS['ti'][0], min(ig_time, _FIRE_BOUNDS['ti'][1]))
    windspeed = max(_FIRE_BOUNDS['u'][0], min(windspeed, _FIRE_BOUNDS['u'][1]))
    depth = max(_FIRE_BOUNDS['d'][0], min(depth, _FIRE_BOUNDS['d'][1]))
    ambient_temp = max(_FIRE_BOUNDS['tamb_c'][0], min(ambient_temp, _FIRE_BOUNDS['tamb_c'][1]))
    params = [
        ('MAX_TIMES', max_times),
        ('INTENSITY', intensity),
        ('IG_TIME', ig_time),
        ('WINDSPEED', windspeed),
        ('DEPTH', depth),
        ('AMBIENT_TEMP', ambient_temp),
        ('R0', r0),
        ('DR', dr),
        ('TIMESTEP', timestep),
        ('SURat_Lit', surat_lit),
        ('SURat_DW1', surat_dw1),
        ('SURat_DW10', surat_dw10),
        ('SURat_DW100', surat_dw100),
        ('SURat_DWk_3_6', surat_dwk_3_6),
        ('SURat_DWk_6_9', surat_dwk_6_9),
        ('SURat_DWk_9_20', surat_dwk_9_20),
        ('SURat_DWk_20', surat_dwk_20)
    ]
    lines = [f'#{name} {value}' for name, value in params]
    content = '\n'.join(lines)
    with open(out_brn_path, 'w') as f:
        f.write(content)
    return

def run_burnup(
    fuel_loadings: Dict[str, float],
    fuel_moistures: Dict[str, float],
    intensity: float = 50.0,
    ig_time: float = 60.0,
    windspeed: float = 0.0,
    depth: float = 0.3,
    ambient_temp: float = 21.0,
    r0: float = 1.83,
    dr: float = 0.4,
    timestep: float = 15.0,
    max_times: int = 3000,
    surat_lit: float = 8200.0,
    surat_dw1: float = 1480.0,
    surat_dw10: float = 394.0,
    surat_dw100: float = 105.0,
    surat_dwk_3_6: float = 39.4,
    surat_dwk_6_9: float = 21.9,
    surat_dwk_9_20: float = 12.7,
    surat_dwk_20: float = 5.91,
    heat_content: float = 1.86e7,
    density: float = 513.0,
    heat_capacity: float = 2750.0,
    conductivity: float = 0.133,
    ignition_temp: float = 327.0,
    char_temp: float = 377.0,
    ash_content: float = 0.05,
    duff_loading: float = 0.0,
    duff_moisture: float = 2.0,
    duff_pct_consumed: float = -1.0,
    densities: Optional[Dict[str, float]] = None,
    fint_switch: float = 15.0,
    validate: bool = True,
    hsf_consumed: float = 0.0,
    brafol_consumed: float = 0.0,
) -> Tuple[List[BurnResult], List[BurnSummaryRow], List[str]]:
    """Run the BURNUP post-frontal combustion model.

    Recognized fuel-loading / moisture keys:
      - sound: ``litter``, ``dw1``, ``dw10``, ``dw100``,
        ``dwk_3_6``, ``dwk_6_9``, ``dwk_9_20``, ``dwk_20``
      - rotten: ``dwk_3_6_r``, ``dwk_6_9_r``, ``dwk_9_20_r``, ``dwk_20_r``
    """
    _sigma_map: Dict[str, float] = {
        'litter':   surat_lit,
        'dw1':      surat_dw1,
        'dw10':     surat_dw10,
        'dw100':    surat_dw100,
        'dwk_3_6':  surat_dwk_3_6,
        'dwk_3_6_r':  surat_dwk_3_6,
        'dwk_6_9':  surat_dwk_6_9,
        'dwk_6_9_r':  surat_dwk_6_9,
        'dwk_9_20': surat_dwk_9_20,
        'dwk_9_20_r': surat_dwk_9_20,
        'dwk_20':   surat_dwk_20,
        'dwk_20_r':   surat_dwk_20,
    }
    _class_order = [
        'litter', 'dw1', 'dw10', 'dw100',
        # Match C++ component order in load.txt: sound+rotten interleaved by size.
        'dwk_3_6', 'dwk_3_6_r', 'dwk_6_9', 'dwk_6_9_r',
        'dwk_9_20', 'dwk_9_20_r', 'dwk_20', 'dwk_20_r',
    ]
    default_moisture = 0.10
    particles: List[FuelParticle] = []
    class_order: List[str] = []
    for key in _class_order:
        loading = fuel_loadings.get(key, 0.0)
        if loading <= 0.0:
            continue
        moisture = fuel_moistures.get(key, default_moisture)
        sigma = _sigma_map[key]
        is_rotten = key.endswith('_r')
        default_density = _DENSITY_ROTTEN if is_rotten else density
        d = densities.get(key, default_density) if densities else default_density
        tpig = _ROTTEN_TPIG if is_rotten else ignition_temp
        particles.append(FuelParticle(
            wdry=loading,
            htval=heat_content,
            fmois=moisture,
            dendry=d,
            sigma=sigma,
            cheat=heat_capacity,
            condry=conductivity,
            tpig=tpig,
            tchar=char_temp,
            ash=ash_content,
        ))
        class_order.append(key)
    if not particles:
        raise ValueError(
            'run_burnup requires at least one fuel size class with loading > 0. '
            f'Recognised keys: {_class_order}'
        )
    results, summary = _burnup(
        particles=particles,
        fi=intensity,
        ti=ig_time,
        u=windspeed,
        d=depth,
        tamb=ambient_temp,
        r0=r0,
        dr=dr,
        dt=timestep,
        ntimes=max_times,
        wdf=duff_loading,
        dfm=duff_moisture,
        duff_pct_consumed=duff_pct_consumed,
        fint_switch=fint_switch,
        validate=validate,
        hsf_consumed=hsf_consumed,
        brafol_consumed=brafol_consumed,
    )
    return results, summary, class_order

