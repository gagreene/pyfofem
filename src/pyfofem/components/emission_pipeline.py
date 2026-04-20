# -*- coding: utf-8 -*-
"""
Helper logic for run_fofem_emissions orchestration.

This module contains pure computation helpers extracted from pyfofem.py to
keep the top-level emissions driver smaller and easier to maintain.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .consumption_calcs import (
    consm_canopy,
    consm_duff,
    consm_herb,
    consm_litter,
    consm_mineral_soil,
    consm_shrub,
)


def compute_pre_burnup_consumption(
    *,
    lit_a: np.ndarray,
    l_m_a: np.ndarray,
    cvr_a: np.ndarray,
    reg_a: np.ndarray,
    units: str,
    her_a: np.ndarray,
    sea_a: np.ndarray,
    shr_a: np.ndarray,
    pcb_a: np.ndarray,
    fol_a: np.ndarray,
    bra_a: np.ndarray,
    ft_a: np.ndarray,
    duf_m_a: np.ndarray,
    duf_a: np.ndarray,
    duf_dep_a: np.ndarray,
    dw10_a: np.ndarray,
    dw1_a: np.ndarray,
    dw1k_m_a: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all consumption arrays that are independent of burnup execution."""
    n = len(lit_a)
    lit_con_arr = np.asarray(
        consm_litter(lit_a, l_m_a, cvr_grp=cvr_a, reg=reg_a, units=units),
        dtype=float,
    )
    lit_pre_arr = lit_a.copy()
    lit_pos_arr = lit_pre_arr - lit_con_arr

    her_pre_arr = her_a.copy()
    her_con_arr = np.asarray(
        consm_herb(reg_a, cvr_a, lit_a, her_a, season=sea_a, units=units),
        dtype=float,
    )
    her_pos_arr = her_pre_arr - her_con_arr

    shr_pre_arr = shr_a.copy()

    crown_res = consm_canopy(pcb_a, fol_a, bra_a, units=units)
    fol_pre_arr = fol_a.copy()
    fol_con_arr = np.asarray(crown_res["flc"], dtype=float)
    fol_pos_arr = fol_pre_arr - fol_con_arr
    bra_pre_arr = bra_a.copy()
    bra_con_arr = np.asarray(crown_res["blc"], dtype=float)
    bra_pos_arr = bra_pre_arr - bra_con_arr

    mse_arr = np.clip(
        np.asarray(consm_mineral_soil(reg_a, cvr_a, ft_a, duf_m_a, "edm"), dtype=float),
        0.0,
        100.0,
    )

    duf_pre_arr = duf_a.copy()
    duf_dep_pre_arr = duf_dep_a.copy()

    pdc_list = np.empty(n, dtype=float)
    ddc_list = np.empty(n, dtype=float)

    for i in range(n):
        pre_l110 = float(lit_a[i] + dw10_a[i] + dw1_a[i])
        pre_dl110 = float(pre_l110 + duf_a[i])
        res = consm_duff(
            float(duf_a[i]),
            float(duf_m_a[i]),
            reg=str(reg_a[i]) if reg_a.size > 0 else None,
            cvr_grp=str(cvr_a[i]) if cvr_a.size > 0 else None,
            duff_moist_cat="edm",
            d_pre=float(duf_dep_a[i]),
            dw1000_moist=float(dw1k_m_a[i]),
            pre_l110=pre_l110,
            pre_dl110=pre_dl110,
            units=units,
        )
        pdc_list[i] = float(np.asarray(res["pdc"]).ravel()[0])
        ddc_list[i] = (
            float(np.asarray(res["ddc"]).ravel()[0]) if res["ddc"] is not None else np.nan
        )

    pdc_arr = np.clip(pdc_list, 0.0, 100.0)
    duf_dep_con_arr = np.clip(ddc_list, 0.0, duf_dep_pre_arr)
    duf_dep_pos_arr = duf_dep_pre_arr - duf_dep_con_arr

    slc_pct_arr = np.clip(
        np.asarray(
            consm_shrub(
                reg_a,
                cvr_a,
                shr_a,
                season=sea_a,
                pre_ll=lit_a,
                pre_dl=duf_a,
                pre_rl=np.zeros_like(shr_a),
                duff_moist=duf_m_a,
                llc=lit_con_arr,
                ddc=duf_a * pdc_arr / 100.0,
                units=units,
            ),
            dtype=float,
        ),
        0.0,
        100.0,
    )

    # C++ Eq 234 parity for SouthEast non-Pocosin non-Flatwoods shrub.
    fw = ("Flatwood", "Pine Flatwoods", "PFL", "PinFltwd")
    is_se_np = (
        (reg_a == "SouthEast")
        & ~np.isin(cvr_a, ("Pocosin", "PC"))
        & ~np.isin(cvr_a, fw)
    )
    if np.any(is_se_np):
        wpre = lit_a + duf_a + dw10_a + dw1_a
        wpre_safe = np.maximum(wpre, 1e-12)
        eq16_w = 3.4958 + (0.3833 * wpre) - (0.0237 * duf_m_a) - (5.6075 / wpre_safe)
        shr_safe = np.maximum(shr_pre_arr, 1e-12)
        f = (
            (3.2484 + (0.4322 * wpre) + (0.6765 * shr_pre_arr)
             - (0.0276 * duf_m_a) - (5.0796 / wpre_safe) - eq16_w)
            / shr_safe
        )
        f = np.where((wpre <= 0.0) | (shr_pre_arr <= 0.0) | (eq16_w == 0.0), 0.0, f)
        eq234_pct = np.clip(f * 100.0, 0.0, 100.0)
        slc_pct_arr = np.where(is_se_np, eq234_pct, slc_pct_arr)

    shr_con_arr = shr_pre_arr * slc_pct_arr / 100.0
    shr_pos_arr = shr_pre_arr - shr_con_arr

    return {
        "lit_pre_arr": lit_pre_arr,
        "lit_con_arr": lit_con_arr,
        "lit_pos_arr": lit_pos_arr,
        "her_pre_arr": her_pre_arr,
        "her_con_arr": her_con_arr,
        "her_pos_arr": her_pos_arr,
        "shr_pre_arr": shr_pre_arr,
        "shr_con_arr": shr_con_arr,
        "shr_pos_arr": shr_pos_arr,
        "fol_pre_arr": fol_pre_arr,
        "fol_con_arr": fol_con_arr,
        "fol_pos_arr": fol_pos_arr,
        "bra_pre_arr": bra_pre_arr,
        "bra_con_arr": bra_con_arr,
        "bra_pos_arr": bra_pos_arr,
        "mse_arr": mse_arr,
        "duf_pre_arr": duf_pre_arr,
        "duf_dep_pre_arr": duf_dep_pre_arr,
        "duf_dep_con_arr": duf_dep_con_arr,
        "duf_dep_pos_arr": duf_dep_pos_arr,
        "pdc_arr": pdc_arr,
    }


def initialize_burnup_outputs(
    *,
    n: int,
    duf_pre_arr: np.ndarray,
    pdc_arr: np.ndarray,
    dw1_a: np.ndarray,
    dw10_a: np.ndarray,
    dw100_a: np.ndarray,
    dw10_m_a: np.ndarray,
    dw1k_m_a: np.ndarray,
    dw1ks_pre: np.ndarray,
    dw1kr_pre: np.ndarray,
    lit_con_arr: np.ndarray,
    frt_a: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Initialize simplified defaults before burnup adjustments are merged."""
    duf_con_arr = duf_pre_arr * pdc_arr / 100.0
    duf_pos_arr = duf_pre_arr - duf_con_arr

    dw1_pre_arr = dw1_a.copy()
    dw1_con_arr = dw1_pre_arr.copy()
    dw1_pos_arr = np.zeros(n)

    dw10_pre_arr = dw10_a.copy()
    dw10_con_arr = dw10_pre_arr.copy()
    dw10_pos_arr = np.zeros(n)

    pct100 = np.clip(0.95 - 0.008 * np.maximum(dw10_m_a - 10.0, 0.0), 0.50, 1.00)
    dw100_pre_arr = dw100_a.copy()
    dw100_con_arr = dw100_pre_arr * pct100
    dw100_pos_arr = dw100_pre_arr - dw100_con_arr

    pct1ks = np.clip(0.15 - 0.001 * dw1k_m_a, 0.02, 0.20)
    dw1ks_con_arr = dw1ks_pre * pct1ks
    dw1ks_pos_arr = dw1ks_pre - dw1ks_con_arr

    pct1kr = np.clip(0.45 - 0.003 * dw1k_m_a, 0.05, 0.50)
    dw1kr_con_arr = dw1kr_pre * pct1kr
    dw1kr_pos_arr = dw1kr_pre - dw1kr_con_arr

    snd_fla_arr = np.zeros(n)
    snd_smo_arr = dw1ks_con_arr.copy()
    rot_fla_arr = np.zeros(n)
    rot_smo_arr = dw1kr_con_arr.copy()
    fine_fla_arr = dw1_con_arr + dw10_con_arr + dw100_con_arr
    fine_smo_arr = np.zeros(n)
    lit_fla_arr = lit_con_arr.copy()
    lit_smo_arr = np.zeros(n)
    fla_dur_arr = np.where(np.isnan(frt_a), np.nan, frt_a)
    smo_dur_arr = np.full(n, np.nan)
    burnup_adj_arr = np.zeros(n, dtype=int)
    burnup_err_arr = np.zeros(n, dtype=int)
    burnup_ran = np.zeros(n, dtype=bool)

    lay0_arr = np.full(n, np.nan)
    lay2_arr = np.full(n, np.nan)
    lay4_arr = np.full(n, np.nan)
    lay6_arr = np.full(n, np.nan)
    lay60d_arr = np.full(n, np.nan)
    lay275d_arr = np.full(n, np.nan)

    return {
        "duf_con_arr": duf_con_arr,
        "duf_pos_arr": duf_pos_arr,
        "dw1_pre_arr": dw1_pre_arr,
        "dw1_con_arr": dw1_con_arr,
        "dw1_pos_arr": dw1_pos_arr,
        "dw10_pre_arr": dw10_pre_arr,
        "dw10_con_arr": dw10_con_arr,
        "dw10_pos_arr": dw10_pos_arr,
        "dw100_pre_arr": dw100_pre_arr,
        "dw100_con_arr": dw100_con_arr,
        "dw100_pos_arr": dw100_pos_arr,
        "dw1ks_con_arr": dw1ks_con_arr,
        "dw1ks_pos_arr": dw1ks_pos_arr,
        "dw1kr_con_arr": dw1kr_con_arr,
        "dw1kr_pos_arr": dw1kr_pos_arr,
        "snd_fla_arr": snd_fla_arr,
        "snd_smo_arr": snd_smo_arr,
        "rot_fla_arr": rot_fla_arr,
        "rot_smo_arr": rot_smo_arr,
        "fine_fla_arr": fine_fla_arr,
        "fine_smo_arr": fine_smo_arr,
        "lit_fla_arr": lit_fla_arr,
        "lit_smo_arr": lit_smo_arr,
        "fla_dur_arr": fla_dur_arr,
        "smo_dur_arr": smo_dur_arr,
        "burnup_adj_arr": burnup_adj_arr,
        "burnup_err_arr": burnup_err_arr,
        "burnup_ran": burnup_ran,
        "lay0_arr": lay0_arr,
        "lay2_arr": lay2_arr,
        "lay4_arr": lay4_arr,
        "lay6_arr": lay6_arr,
        "lay60d_arr": lay60d_arr,
        "lay275d_arr": lay275d_arr,
    }


def compute_equation_arrays(
    reg_a: np.ndarray,
    cvr_a: np.ndarray,
    sea_a: np.ndarray,
    ft_a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute C++-aligned equation-id outputs for litter/duff/herb/shrub/MSE."""
    fw = ("Flatwood", "Pine Flatwoods", "PFL", "PinFltwd")
    is_flatwood = np.isin(cvr_a, fw)
    is_se = reg_a == "SouthEast"
    is_ne = reg_a == "NorthEast"
    is_iw_pw = np.isin(reg_a, ("InteriorWest", "PacificWest"))
    is_pocosin = np.isin(cvr_a, ("Pocosin", "PC"))
    is_ponderosa = np.isin(cvr_a, ("Ponderosa pine", "PN", "Ponderosa"))
    is_redjac = np.isin(cvr_a, ("Red Jack Pine", "Red, Jack Pine", "RedJacPin", "RJP"))
    is_balsam = np.isin(
        cvr_a,
        ("Balsam", "Black Spruce", "Red Spruce", "White Spruce", "BalBRWSpr", "Balsam Fir", "BFS"),
    )
    is_chaparral = np.isin(cvr_a, ("Chaparral", "Shrub-Chaparral", "SGC", "ShrubGroupChaparral"))
    is_sage = np.isin(cvr_a, ("Sagebrush", "SB"))
    is_shrubgrp = np.isin(cvr_a, ("Shrub", "SG", "ShrubGroup"))
    is_grass = np.isin(cvr_a, ("Grass", "GG", "GrassGroup"))
    is_spring = sea_a == "Spring"
    is_summer = sea_a == "Summer"
    is_fall = sea_a == "Fall"
    is_winter = sea_a == "Winter"
    lit_eq_arr = np.where(is_flatwood, 997, np.where(is_se, 998, 999))

    duf_con_eq_arr = np.full(reg_a.shape, 2, dtype=int)
    duf_red_eq_arr = np.full(reg_a.shape, 6, dtype=int)
    mse_eq_arr = np.full(reg_a.shape, 10, dtype=int)

    duf_con_eq_arr = np.where(is_chaparral, 19, duf_con_eq_arr)
    duf_red_eq_arr = np.where(is_chaparral, 19, duf_red_eq_arr)
    mse_eq_arr = np.where(is_chaparral, 19, mse_eq_arr)

    iw_pw_ponderosa = is_iw_pw & is_ponderosa
    iw_pw_other = is_iw_pw & ~is_ponderosa
    duf_con_eq_arr = np.where(iw_pw_ponderosa | iw_pw_other, 2, duf_con_eq_arr)
    duf_red_eq_arr = np.where(iw_pw_ponderosa | iw_pw_other, 6, duf_red_eq_arr)
    mse_eq_arr = np.where(iw_pw_ponderosa | iw_pw_other, 10, mse_eq_arr)

    ne_other = is_ne & ~is_redjac & ~is_balsam
    ne_redjac = is_ne & is_redjac
    ne_balsam = is_ne & is_balsam
    duf_con_eq_arr = np.where(ne_other, 2, duf_con_eq_arr)
    duf_red_eq_arr = np.where(ne_other, 6, duf_red_eq_arr)
    mse_eq_arr = np.where(ne_other, 10, mse_eq_arr)
    duf_con_eq_arr = np.where(ne_redjac | ne_balsam, 15, duf_con_eq_arr)
    duf_red_eq_arr = np.where(ne_redjac | ne_balsam, 15, duf_red_eq_arr)
    mse_eq_arr = np.where(ne_redjac | ne_balsam, 14, mse_eq_arr)

    se_pocosin = is_se & is_pocosin
    se_other = is_se & ~is_pocosin
    duf_con_eq_arr = np.where(se_pocosin, 20, duf_con_eq_arr)
    duf_red_eq_arr = np.where(se_pocosin, 20, duf_red_eq_arr)
    mse_eq_arr = np.where(se_pocosin, 202, mse_eq_arr)
    duf_con_eq_arr = np.where(se_other, 16, duf_con_eq_arr)
    duf_red_eq_arr = np.where(se_other, 16, duf_red_eq_arr)
    mse_eq_arr = np.where(se_other, 14, mse_eq_arr)

    herb_eq_arr = np.where(
        is_flatwood,
        223,
        np.where(is_se, 222, np.where(is_grass & is_spring, 221, 22)),
    )

    shrub_eq_arr = np.where(
        is_sage & is_fall,
        233,
        np.where(
            is_sage,
            232,
            np.where(
                is_flatwood,
                236,
                np.where(
                    is_shrubgrp,
                    231,
                    np.where(
                        se_pocosin & (is_spring | is_winter),
                        233,
                        np.where(se_pocosin & (is_summer | is_fall), 235, np.where(is_se, 234, 23)),
                    ),
                ),
            ),
        ),
    )
    return lit_eq_arr, duf_con_eq_arr, duf_red_eq_arr, herb_eq_arr, shrub_eq_arr, mse_eq_arr


def build_emissions_result(
    *,
    scalar_call: bool,
    soil_enabled: bool,
    em_mode: str,
    expanded_consumption_vars: set,
    emissions: Dict[str, np.ndarray],
    lit_pre_arr: np.ndarray,
    lit_con_arr: np.ndarray,
    lit_pos_arr: np.ndarray,
    dw1_pre_arr: np.ndarray,
    dw1_con_arr: np.ndarray,
    dw1_pos_arr: np.ndarray,
    dw10_pre_arr: np.ndarray,
    dw10_con_arr: np.ndarray,
    dw10_pos_arr: np.ndarray,
    dw100_pre_arr: np.ndarray,
    dw100_con_arr: np.ndarray,
    dw100_pos_arr: np.ndarray,
    dw1ks_pre: np.ndarray,
    dw1ks_con_arr: np.ndarray,
    dw1ks_pos_arr: np.ndarray,
    dw1kr_pre: np.ndarray,
    dw1kr_con_arr: np.ndarray,
    dw1kr_pos_arr: np.ndarray,
    duf_pre_arr: np.ndarray,
    duf_con_arr: np.ndarray,
    duf_pos_arr: np.ndarray,
    her_pre_arr: np.ndarray,
    her_con_arr: np.ndarray,
    her_pos_arr: np.ndarray,
    shr_pre_arr: np.ndarray,
    shr_con_arr: np.ndarray,
    shr_pos_arr: np.ndarray,
    fol_pre_arr: np.ndarray,
    fol_con_arr: np.ndarray,
    fol_pos_arr: np.ndarray,
    bra_pre_arr: np.ndarray,
    bra_con_arr: np.ndarray,
    bra_pos_arr: np.ndarray,
    mse_arr: np.ndarray,
    duf_dep_pre_arr: np.ndarray,
    duf_dep_con_arr: np.ndarray,
    duf_dep_pos_arr: np.ndarray,
    fla_dur_arr: np.ndarray,
    smo_dur_arr: np.ndarray,
    fla_con_arr: np.ndarray,
    smo_con_arr: np.ndarray,
    lit_eq_arr: np.ndarray,
    duf_con_eq_arr: np.ndarray,
    duf_red_eq_arr: np.ndarray,
    herb_eq_arr: np.ndarray,
    shrub_eq_arr: np.ndarray,
    mse_eq_arr: np.ndarray,
    burnup_adj_arr: np.ndarray,
    burnup_err_arr: np.ndarray,
    lay0_arr: np.ndarray,
    lay2_arr: np.ndarray,
    lay4_arr: np.ndarray,
    lay6_arr: np.ndarray,
    lay60d_arr: np.ndarray,
    lay275d_arr: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assemble final run_fofem_emissions output dictionary."""

    def out(arr):
        a = np.asarray(arr, dtype=float)
        return float(a[0]) if scalar_call else a

    def out_int(arr):
        a = np.asarray(arr, dtype=int)
        return int(a[0]) if scalar_call else a

    result = {
        "LitPre": out(lit_pre_arr), "LitCon": out(lit_con_arr), "LitPos": out(lit_pos_arr),
        "DW1Pre": out(dw1_pre_arr), "DW1Con": out(dw1_con_arr), "DW1Pos": out(dw1_pos_arr),
        "DW10Pre": out(dw10_pre_arr), "DW10Con": out(dw10_con_arr), "DW10Pos": out(dw10_pos_arr),
        "DW100Pre": out(dw100_pre_arr), "DW100Con": out(dw100_con_arr), "DW100Pos": out(dw100_pos_arr),
        "DW1kSndPre": out(dw1ks_pre), "DW1kSndCon": out(dw1ks_con_arr), "DW1kSndPos": out(dw1ks_pos_arr),
        "DW1kRotPre": out(dw1kr_pre), "DW1kRotCon": out(dw1kr_con_arr), "DW1kRotPos": out(dw1kr_pos_arr),
        "DufPre": out(duf_pre_arr), "DufCon": out(duf_con_arr), "DufPos": out(duf_pos_arr),
        "HerPre": out(her_pre_arr), "HerCon": out(her_con_arr), "HerPos": out(her_pos_arr),
        "ShrPre": out(shr_pre_arr), "ShrCon": out(shr_con_arr), "ShrPos": out(shr_pos_arr),
        "FolPre": out(fol_pre_arr), "FolCon": out(fol_con_arr), "FolPos": out(fol_pos_arr),
        "BraPre": out(bra_pre_arr), "BraCon": out(bra_con_arr), "BraPos": out(bra_pos_arr),
        "MSE": out(mse_arr),
        "DufDepPre": out(duf_dep_pre_arr), "DufDepCon": out(duf_dep_con_arr), "DufDepPos": out(duf_dep_pos_arr),
        "FlaDur": out(fla_dur_arr), "SmoDur": out(smo_dur_arr),
        "FlaCon": out(fla_con_arr), "SmoCon": out(smo_con_arr),
        "Lit-Equ": out_int(lit_eq_arr),
        "DufCon-Equ": out_int(duf_con_eq_arr),
        "DufRed-Equ": out_int(duf_red_eq_arr),
        "MSE-Equ": out_int(mse_eq_arr),
        "Herb-Equ": out_int(herb_eq_arr),
        "Shurb-Equ": out_int(shrub_eq_arr),
        "BurnupLimitAdj": out_int(burnup_adj_arr),
        "BurnupError": out_int(burnup_err_arr),
    }

    emission_out = {k: (out(v) if isinstance(v, np.ndarray) else v) for k, v in emissions.items()}
    if em_mode in ("legacy", "expanded"):
        result.update(emission_out)
    else:
        for key, val in emission_out.items():
            if key not in expanded_consumption_vars:
                result[key] = val

    if soil_enabled:
        result.update(
            {
                "Lay0": out(lay0_arr),
                "Lay2": out(lay2_arr),
                "Lay4": out(lay4_arr),
                "Lay6": out(lay6_arr),
                "Lay60d": out(lay60d_arr),
                "Lay275d": out(lay275d_arr),
            }
        )
    return result
