# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13, 12:00:00 2025

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import os
import numpy as np
from pandas import read_csv, DataFrame
from typing import Dict, List, Optional, Tuple, Union

from .components.burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup as _burnup,
)


# Load species codes lookup table
SPP_CODES = read_csv(os.path.join(os.path.dirname(__file__), 'supporting_data', 'species_codes_lut.csv'))

CONSUMPTION_VARS = [
    'LitPre', 'LitCon', 'LitPos', 'DW1Pre', 'DW1Con', 'DW1Pos', 'DW10Pre', 'DW10Con', 'DW10Pos',
    'DW100Pre', 'DW100Con', 'DW100Pos', 'DW1kSndPre', 'DW1kSndCon', 'DW1kSndPos', 'DW1kRotPre', 'DW1kRotCon',
    'DW1kRotPos', 'DufPre', 'DufCon', 'DufPos', 'HerPre', 'HerCon', 'HerPos', 'ShrPre', 'ShrCon', 'ShrPos',
    'FolPre', 'FolCon', 'FolPos', 'BraPre', 'BraCon', 'BraPos', 'MSE', 'DufDepPre', 'DufDepCon', 'DufDepPos',
    'PM10F', 'PM10S', 'PM25F', 'PM25S', 'CH4F', 'CH4S', 'COF', 'COS', 'CO2F', 'CO2S', 'NOXF', 'NOXS', 'SO2F',
    'SO2S', 'FlaDur', 'SmoDur', 'FlaCon', 'SmoCon', 'Lay0', 'Lay2', 'Lay4', 'Lay6', 'Lay60d', 'Lay275d',
    'Lit-Equ', 'DufCon-Equ', 'DufRed-Equ', 'MSE-Equ', 'Herb-Equ', 'Shurb-Equ'
]


def _is_scalar(x) -> bool:
    """Return True if *x* is a Python scalar or 0-d array (i.e. not a sequence/array)."""
    if isinstance(x, np.ndarray):
        return x.ndim == 0
    return np.ndim(x) == 0


def calc_scorch_ht(
    sfi: Union[float, np.ndarray],
    amb_t: Optional[Union[float, np.ndarray]] = None,
    instand_ws: Optional[Union[float, np.ndarray]] = None
) -> Union[float, np.ndarray]:
    """
    Van Wagner (1973) & Alexander (1982/85) lethal scorch height model.

    :param sfi: Surface fire intensity (kW/m), scalar or np.ndarray
    :param amb_t: Ambient temperature (C), scalar or np.ndarray, optional
    :param instand_ws: Instantaneous windspeed (m/s), scalar or np.ndarray, optional
    :return: Scorch height (m), scalar or np.ndarray
    """
    sfi = np.asarray(sfi)
    if np.any(sfi == None):
        raise Exception('Must enter a surface fire intensity value to estimate scorch height (fn _calc_scorch_ht)')

    if amb_t is None:
        # Equation 8
        return 0.1483 * np.power(sfi, 2 / 3)
    elif instand_ws is None:
        # Equation 9
        return 4.4713 * np.power(sfi, 2 / 3) / (60 - amb_t)
    else:
        # Equation 10
        sfi = np.asarray(sfi)
        amb_t = np.asarray(amb_t)
        instand_ws = np.asarray(instand_ws)
        return ((0.74183 * np.power(sfi, 7 / 6)) /
                (np.power((0.025574 * sfi) + (0.021433 * np.power(instand_ws, 3)), 0.5) * (60 - amb_t)))


def calc_flame_length(
    fire_intensity: Optional[Union[float, np.ndarray]] = None,
    char_ht: Optional[Union[float, np.ndarray]] = None,
    fl_model: str = 'Byram'
) -> Union[float, np.ndarray]:
    """
    Flame length model (Byram, Butler, Thomas).

    :param fire_intensity: Surface fire intensity (kW/m), scalar or np.ndarray, optional
    :param char_ht: Char height (m), scalar or np.ndarray, optional
    :param fl_model: Flame length model to use ('Byram', 'Butler', or other), default 'Byram'
    :return: Flame length (m), scalar or np.ndarray
    """
    if (fire_intensity is None) and (char_ht is None):
        raise ValueError('Must enter a surface fire intensity or char height value to estimate '
                         'flame length (fn _calc_flame_length)')

    if fire_intensity is not None:
        fire_intensity = np.asarray(fire_intensity)
        if fl_model == 'Byram':
            return 0.0775 * np.power(fire_intensity, 0.46)
        if fl_model == 'Butler':
            return 0.017500 * np.power(fire_intensity, 2 / 3)
        else:  # fl_model == 'Thomas':
            return 0.026700 * np.power(fire_intensity, 2 / 3)

    if (fire_intensity is None) and (char_ht is not None):
        char_ht = np.asarray(char_ht)
        return char_ht * 1.8


def calc_char_ht(flame_length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorized char height calculation.

    :param flame_length: Flame length (m), scalar or np.ndarray
    :return: Char height (m), scalar or np.ndarray
    """
    flame_length = np.asarray(flame_length)
    return flame_length / 1.8


def calc_bark_thickness(
    spp: np.ndarray,
    dbh: np.ndarray,
) -> np.ndarray:
    """
    Vectorized bark thickness calculation \(cm\).

    :param spp: np.ndarray of species codes \(str or int\)
    :param dbh: np.ndarray of diameters \(cm\)
    :return: np.ndarray of bark thickness values \(cm\)
    """
    spp = np.asarray(spp)
    dbh = np.asarray(dbh)

    if spp.shape != dbh.shape:
        raise ValueError('spp and dbh must have the same shape')

    if np.issubdtype(spp.dtype, np.integer):
        num_to_fofem = SPP_CODES.drop_duplicates(subset='num_cd').set_index('num_cd')['fofem_cd']
        spp_str = np.array([num_to_fofem.get(int(code), 'UNK') for code in spp], dtype=str)
    else:
        spp_str = spp.astype(str)

    bark_lookup = SPP_CODES.drop_duplicates(subset='fofem_cd').set_index('fofem_cd')['FOFEM_BrkThck_Vsp']
    bark_thick_per_dbh = bark_lookup.reindex(spp_str).to_numpy()

    if np.any(np.isnan(bark_thick_per_dbh)):
        missing = np.unique(spp_str[np.isnan(bark_thick_per_dbh)])
        raise ValueError(f'No bark thickness coefficient found for species code\(s\): {missing.tolist()}')

    return bark_thick_per_dbh * dbh


def calc_crown_length_vol_scorched(
    scorch_ht: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of crown length scorched (m), percent crown volume scorched (cvs, %), and
    percent crown length scorched (cls, %). Accepts scalars or np.ndarray inputs; values are broadcast
    to a common shape.

    :param scorch_ht: Scorch height (m), scalar or np.ndarray.
    :param ht: Total tree height (m), scalar or np.ndarray.
    :param crown_depth: Crown depth (m), scalar or np.ndarray.
    :return: Tuple of np.ndarrays [crown_length_scorched (m), cvs (%), cls (%)] matching the broadcast shape.
    """
    scorch_ht = np.asarray(scorch_ht)
    ht = np.asarray(ht)
    crown_depth = np.asarray(crown_depth)
    crown_length_scorched = scorch_ht - (ht - crown_depth)
    crown_length_scorched = np.clip(crown_length_scorched, 0, crown_depth)
    cvs = 100 * (crown_length_scorched * ((2 * crown_depth) - crown_length_scorched) / np.power(crown_depth, 2))
    cls = 100 * (crown_length_scorched / crown_depth)
    return crown_length_scorched, cvs, cls


def consm_canopy(
    crown_burn: Union[float, np.ndarray],
    pre_fl: Union[float, np.ndarray],
    pre_bl: Union[float, np.ndarray],
    units: str = 'SI',
) -> dict:
    """
    FOFEM canopy (crown fire) fuel consumption model.

    Estimates foliage and branch load consumed during a crown fire, given a
    user-provided estimate of the proportion of the stand affected by crown
    fire.

    Accepts scalar or array inputs. Scalar values in the returned dict
    correspond to scalar numeric inputs; arrays are returned otherwise.

    :param crown_burn: Proportion of stand area affected by crown fire (%).
        Scalar or np.ndarray.
    :param pre_fl: Pre-fire foliage fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_bl: Pre-fire branch fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Dict with keys:

        - ``'flc'`` – foliage load consumed (same units as input). Scalar
          ``float`` when all numeric inputs are scalars, otherwise
          ``np.ndarray``.
        - ``'blc'`` – branch load consumed (same units as input; 50% of the
          proportion consumed by crown fire). Scalar ``float`` when all
          numeric inputs are scalars, otherwise ``np.ndarray``.
    """
    scalar_input = _is_scalar(crown_burn) and _is_scalar(pre_fl) and _is_scalar(pre_bl)

    crown_burn = np.atleast_1d(np.asarray(crown_burn, dtype=float))
    pre_fl = np.atleast_1d(np.asarray(pre_fl, dtype=float))
    pre_bl = np.atleast_1d(np.asarray(pre_bl, dtype=float))

    if units == 'SI':
        pre_fl = pre_fl * 4.4609  # kg/m² → T/acre
        pre_bl = pre_bl * 4.4609

    flc = (crown_burn / 100) * pre_fl
    blc = (crown_burn / 100) * pre_bl * 0.5

    if units == 'SI':
        flc = flc / 4.4609  # T/acre → kg/m²
        blc = blc / 4.4609

    def _maybe_scalar(arr):
        return float(arr[0]) if scalar_input else arr

    return {
        'flc': _maybe_scalar(flc),
        'blc': _maybe_scalar(blc),
    }


def consm_duff(
    pre_dl: Union[float, np.ndarray],
    duff_moist: Union[float, np.ndarray],
    reg: Optional[str] = None,
    cvr_grp: Optional[str] = None,
    duff_moist_cat: Optional[str] = None,
    d_pre: Optional[Union[float, np.ndarray]] = None,
    rm_depth: Optional[Union[float, list]] = None,
    mc_lyr1: Optional[Union[float, np.ndarray]] = None,
    pre_dl110: Optional[Union[float, np.ndarray]] = None,
    pre_l110: Optional[Union[float, np.ndarray]] = None,
    pile: bool = False,
    units: str = 'SI',
) -> dict:
    """
    FOFEM duff consumption model.

    Computes percent duff consumed (``'pdc'``), duff depth consumed
    (``'ddc'``), and residual duff depth (``'rdd'``) and returns them in a
    dict. Accepts scalar or array inputs for all numeric parameters; scalar
    values in the returned dict correspond to scalar numeric inputs.

    :param pre_dl: Pre-fire duff load (Mg/ha if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param reg: Region name. Options include ``'InteriorWest'``,
        ``'PacificWest'``, ``'NorthEast'``, ``'SouthEast'``.
    :param cvr_grp: Cover group name. Options include ``'Ponderosa pine'``,
        ``'Pocosin'``, ``'Chaparral'``, etc.
    :param duff_moist_cat: Duff moisture category. One of:

        - ``'ldm'`` – lower duff moisture
        - ``'edm'`` – entire / average duff moisture
        - ``'nfdth'`` – NFDR 1,000-hour moisture content

    :param d_pre: Pre-fire duff depth (cm if ``units='SI'``, inches if
        ``units='Imperial'``). Scalar or np.ndarray. Required for duff depth
        consumption (``'ddc'``) and residual duff depth (``'rdd'``) outputs.
    :param rm_depth: Southeast Pocosin only. Depth of root mat and deep
        organic layer. Single numeric (total depth) or a list of individual
        layer depths (cm if ``units='SI'``, inches if ``units='Imperial'``).
    :param mc_lyr1: Percent soil moisture content of layer 1 (%). Scalar or
        np.ndarray. Required for Southeast Pocosin equations.
    :param pre_dl110: Pre-fire duff + 1-hr + 10-hr fuel load. Scalar or
        np.ndarray. Required for the Southeast non-Pocosin equation (Eq 16).
    :param pre_l110: Pre-fire litter + 1-hr + 10-hr fuel load. Scalar or
        np.ndarray. Required for the Southeast non-Pocosin equation (Eq 16).
    :param pile: ``True`` for pile burning (applies Eq 17, 90% consumed).
        Default ``False``.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Dict with keys:

        - ``'pdc'`` – percent duff consumed (%) or np.ndarray; ``np.nan``
          where the condition could not be determined.
        - ``'ddc'`` – duff depth consumed (inches) or np.ndarray; ``None``
          if ``d_pre`` is not provided.
        - ``'rdd'`` – residual duff depth (inches) or np.ndarray; ``None``
          if ``duff_moist_cat != 'edm'`` or ``d_pre`` is not provided.
    """
    scalar_input = _is_scalar(pre_dl) and _is_scalar(duff_moist)

    pre_dl = np.atleast_1d(np.asarray(pre_dl, dtype=float))
    duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    if d_pre is not None:
        d_pre = np.atleast_1d(np.asarray(d_pre, dtype=float))
    if mc_lyr1 is not None:
        mc_lyr1 = np.atleast_1d(np.asarray(mc_lyr1, dtype=float))
    if pre_dl110 is not None:
        pre_dl110 = np.atleast_1d(np.asarray(pre_dl110, dtype=float))
    if pre_l110 is not None:
        pre_l110 = np.atleast_1d(np.asarray(pre_l110, dtype=float))

    if units == 'SI':
        if isinstance(rm_depth, list):
            rm_depth = [x / 2.54 for x in rm_depth]   # cm → in
        elif rm_depth is not None:
            rm_depth = rm_depth / 2.54
        if pre_dl110 is not None:
            pre_dl110 = pre_dl110 * 4.4609             # kg/m² → T/acre
        if pre_l110 is not None:
            pre_l110 = pre_l110 * 4.4609

    # --- pdc: PERCENT DUFF CONSUMED ---
    pdc = np.full_like(duff_moist, np.nan)

    if (duff_moist_cat == 'ldm') and (reg in ['InteriorWest', 'PacificWest']):
        if cvr_grp not in ['Ponderosa pine', 'PN', 'Ponderosa', 'Pocosin', 'PC']:
            # Equation 1
            pdc = np.where(duff_moist <= 160, 97.1 - 0.519 * duff_moist, 13.6)
        else:
            # Equation 4
            pdc = 89.9 - 0.55 * duff_moist
    elif (duff_moist_cat == 'nfdth') and (reg in ['InteriorWest', 'PacificWest', 'NorthEast']):
        # Equation 3
        pdc = 114.7 - 4.20 * duff_moist
    elif reg == 'SouthEast':
        if cvr_grp in ['Pocosin', 'PC']:
            # Equation 20 – per-layer calculation
            if isinstance(rm_depth, list):
                n_layers = len(rm_depth)
                depths = rm_depth
                total_depth = sum(rm_depth)
            else:
                if (rm_depth % 4) > 0:
                    n_layers = int(rm_depth // 4) + 1
                else:
                    n_layers = int(rm_depth / 4)
                depths = []
                for i in range(n_layers):
                    depths.append(rm_depth % 4 if i == (n_layers - 1) else 4.0)
                total_depth = rm_depth

            mc_layers = [float(mc_lyr1[0])] * n_layers
            mc_multiplier = 1.0
            for i in range(len(mc_layers)):
                mc_multiplier += (min(3 * i, 12)) / 100
                mc_layers[i] *= mc_multiplier

            pdc_layers = []
            for mc in mc_layers:
                if mc < 10:
                    pdc_layers.append(1.0)
                elif mc < 30:
                    pdc_layers.append(float(pre_dl[0]) * (0.949932 + ((30 - mc) * 0.00251)))
                elif mc < 140:
                    pdc_layers.append(float(pre_dl[0]) * (1 / (1 + np.exp(-1 * (2.033 - (0.043 * mc) + (0.44 * 0.05))))))
                elif mc < 170:
                    pdc_layers.append(float(pre_dl[0]) * (0.143441 - ((mc - 140) * 0.0049)))
                else:
                    pdc_layers.append(0.0)

            depth_cons = np.multiply(depths, pdc_layers)
            pdc = np.atleast_1d(sum(depth_cons) / total_depth)
        else:
            # Equation 16
            duff_litt_cons = 3.4958 + (0.3833 * pre_dl110) - (0.0237 * duff_moist) - (5.6075 / pre_dl110)
            pdc = np.where(
                duff_litt_cons <= pre_l110, 0.0,
                np.where(duff_litt_cons > pre_dl110,
                         ((duff_litt_cons - pre_dl110) / (pre_dl110 - pre_l110)) * 100,
                         np.nan)
            )
    elif pile:
        # Equation 17 – pile burning: 90% consumed
        pdc = np.full_like(duff_moist, (1 - 0.1) * pre_dl)
    elif cvr_grp in ['Chaparral', 'Shrub-Chaparral', 'SGC', 'ShrubGroupChaparral']:
        # Equation 19 – Chaparral: 100% consumed
        pdc = np.full_like(duff_moist, 100.0)
    else:
        # Equation 2 – default
        pdc = 83.7 - 0.426 * duff_moist

    # --- ddc: DUFF DEPTH CONSUMED ---
    ddc = None
    if d_pre is not None:
        if (duff_moist_cat == 'ldm') and (reg in ['InteriorWest', 'PacificWest']):
            # Equation 5
            ddc = 1.028 - 0.0089 * duff_moist + 0.417 * d_pre
        elif (duff_moist_cat == 'nfdth') and (reg in ['InteriorWest', 'PacificWest', 'NorthEast']):
            # Equation 7
            ddc = 1.773 - 0.1051 * duff_moist + 0.399 * d_pre
        else:
            # Equation 6 – default
            ddc = 0.8811 - 0.0096 * duff_moist + 0.439 * d_pre

    # --- rdd: RESIDUAL DUFF DEPTH ---
    rdd = None
    if (duff_moist_cat == 'edm') and (d_pre is not None):
        pine = 1 if cvr_grp in ['Red Jack Pine', 'Red, Jack Pine', 'RedJacPin', 'RJP'] else 0
        # Equation 15
        rdd = -0.791 + 0.004 * duff_moist + 0.8 * d_pre + 0.56 * pine

    def _maybe_scalar(arr):
        if arr is None:
            return None
        return float(arr[0]) if scalar_input else arr

    return {
        'pdc': _maybe_scalar(pdc),
        'ddc': _maybe_scalar(ddc),
        'rdd': _maybe_scalar(rdd),
    }


def consm_herb(
    reg: str,
    cvr_grp: str,
    pre_ll: Union[float, np.ndarray],
    pre_hl: Union[float, np.ndarray],
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM herbaceous fuel consumption model.

    Accepts scalar or array inputs. When all numeric inputs are scalars, a
    scalar ``float`` is returned; otherwise a 1D ``np.ndarray`` is returned.

    :param reg: Region name. ``'SouthEast'`` selects Eq 222; other regions
        use cover-group-specific or default equations.
    :param cvr_grp: Cover group name. Recognises:

        - ``'Grass'`` / ``'GG'`` / ``'GrassGroup'`` → Eq 221 (10% consumed)
        - ``'Flatwood'`` / ``'Pine Flatwoods'`` / ``'PFL'`` / ``'PinFltwd'``
          → Eq 223
        - All others → Eq 22 (100% consumed)

    :param pre_ll: Pre-fire litter fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_hl: Pre-fire herbaceous fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Herbaceous load consumed (same units as input). Scalar ``float``
        when all numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(pre_hl)

    pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
    pre_hl = np.atleast_1d(np.asarray(pre_hl, dtype=float))

    if units == 'SI':
        pre_ll = pre_ll * 4.4609  # kg/m² → T/acre
        pre_hl = pre_hl * 4.4609

    if reg in ['SouthEast']:
        # Equation 222
        hlc = -0.59 + (0.004 * pre_ll) + (0.917 * pre_hl)
    elif cvr_grp in ['Grass', 'GG', 'GrassGroup']:
        # Equation 221
        hlc = pre_hl * 0.1
    elif cvr_grp in ['Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd']:
        # Equation 223 with unit conversions
        hlc = ((pre_hl * 2.24) * 0.9944) / 2.24
    else:
        # Equation 22 – default (100% consumed)
        hlc = pre_hl.copy()

    if units == 'SI':
        hlc = hlc / 4.4609  # T/acre → kg/m²

    return float(hlc[0]) if scalar_input else hlc


def consm_litter(
    pre_ll: Union[float, np.ndarray],
    l_moist: Union[float, np.ndarray],
    cvr_grp: Optional[str] = None,
    reg: Optional[str] = None,
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM litter consumption model (Eqs 997–999).

    Accepts scalar or array inputs. When all numeric inputs are scalars, a
    scalar ``float`` is returned; otherwise a 1D ``np.ndarray`` is returned.

    .. note::
        Most fuel consumption is simulated using Burnup. This function covers
        litter-specific override equations for Flatwoods and Southeast regions.

    :param pre_ll: Pre-fire litter load (Mg/ha if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param l_moist: Litter moisture content (%). Scalar or np.ndarray.
    :param cvr_grp: Cover group name. Selects the Flatwoods equation (Eq 997)
        when set to ``'Flatwood'``, ``'Pine Flatwoods'``, ``'PFL'``, or
        ``'PinFltwd'``. Optional.
    :param reg: Region name. Selects the Southeast equation (Eq 998) when set
        to ``'SouthEast'``. Optional.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``. When
        ``'Imperial'``, inputs are converted to Mg/ha internally and the result
        is converted back to T/acre before returning.

    :return: Litter load consumed (Mg/ha for ``'SI'``, T/acre for
        ``'Imperial'``). Scalar ``float`` when all numeric inputs are scalars,
        otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(l_moist)

    pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
    l_moist = np.atleast_1d(np.asarray(l_moist, dtype=float))

    if units == 'Imperial':
        pre_ll = pre_ll * 0.446089561  # T/acre → Mg/ha

    if cvr_grp in ['Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd']:
        # FOFEM litter consumption equation 997
        llc = np.power(0.2871 + (0.9140 * np.sqrt(pre_ll)) - (0.0101 * l_moist), 2)
    elif reg == 'SouthEast':
        # FOFEM litter consumption equation 998
        llc = pre_ll * 0.8
    else:
        # FOFEM litter consumption equation 999 – calculated with Burnup (generally 100%)
        llc = pre_ll.copy()

    if units == 'Imperial':
        llc = llc / 0.446089561  # Mg/ha → T/acre

    return float(llc[0]) if scalar_input else llc


def consm_mineral_soil(
    reg: str,
    cvr_grp: str,
    fuel_type: str,
    duff_moist: Union[float, np.ndarray],
    duff_moist_cat: str,
    pile: bool = False,
    pdr: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM mineral soil exposure model.

    Estimates the proportion of mineral soil exposed by fire (%).

    Accepts scalar or array inputs. When all numeric inputs are scalars, a
    scalar ``float`` is returned; otherwise a 1D ``np.ndarray`` is returned.

    :param reg: Region name. Options include ``'InteriorWest'``,
        ``'PacificWest'``, ``'NorthEast'``, ``'SouthEast'``.
    :param cvr_grp: Cover group name.
    :param fuel_type: Fuel type; ``'natural'`` or ``'slash'``.
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param duff_moist_cat: Duff moisture category. One of:

        - ``'ldm'`` – lower duff moisture
        - ``'edm'`` – entire / average duff moisture
        - ``'nfdth'`` – NFDR 1,000-hour moisture content
        - ``'%dr'`` – percent duff reduction (from ``consume_duff``)

    :param pile: ``True`` for pile burning (returns 10%). Default ``False``.
    :param pdr: Percent duff reduction (%), required when
        ``duff_moist_cat='%dr'``. Scalar or np.ndarray.

    :return: Mineral soil exposure (%). Scalar ``float`` when all numeric
        inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(duff_moist)

    duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    if pdr is not None:
        pdr = np.atleast_1d(np.asarray(pdr, dtype=float))

    mse = np.full_like(duff_moist, np.nan)

    if pile:
        # Equation 18
        mse = np.full_like(duff_moist, 10.0)
    elif reg in ['InteriorWest', 'PacificWest']:
        if fuel_type == 'slash' and duff_moist_cat == 'ldm':
            # Equation 9
            mse = np.where(duff_moist <= 135,
                           80 - 0.507 * duff_moist,
                           23.5 - 0.0914 * duff_moist)
        elif fuel_type == 'natural' and duff_moist_cat == 'ldm':
            # Equation 13
            mse = 60.4 - 0.440 * duff_moist
        elif fuel_type == 'slash' and duff_moist_cat == 'nfdth':
            # Equation 11
            mse = 93.3 - 3.55 * duff_moist
        elif fuel_type == 'natural' and duff_moist_cat == 'nfdth':
            # Equation 12
            mse = 94.3 - 4.96 * duff_moist
        elif duff_moist_cat == 'edm':
            # Equation 10
            mse = 167.4 - 31.6 * np.log(duff_moist)
    elif cvr_grp in ['Pocosin', 'PC']:
        # Equation 202
        mse = np.zeros_like(duff_moist)
    else:
        if duff_moist_cat == 'edm':
            # Default to Equation 10
            mse = 167.4 - 31.6 * np.log(duff_moist)
        elif pdr is not None:
            # Equation 14
            mse = -8.98 + 0.899 * pdr

    return float(mse[0]) if scalar_input else mse


def consm_shrub(
    reg: str,
    cvr_grp: str,
    pre_sl: Union[float, np.ndarray],
    season: Optional[str] = None,
    pre_ll: Optional[Union[float, np.ndarray]] = None,
    pre_dl: Optional[Union[float, np.ndarray]] = None,
    pre_rl: Optional[Union[float, np.ndarray]] = None,
    duff_moist: Optional[Union[float, np.ndarray]] = None,
    llc: Optional[Union[float, np.ndarray]] = None,
    ddc: Optional[Union[float, np.ndarray]] = None,
    units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM shrub fuel consumption model.

    Accepts scalar or array inputs. When all numeric inputs are scalars, a
    scalar ``float`` is returned; otherwise a 1D ``np.ndarray`` is returned.

    :param reg: Region name. ``'Southeast'`` selects region-specific
        equations; other values use cover-group equations.
    :param cvr_grp: Cover group name. Recognises:

        - ``'Pocosin'`` / ``'PC'`` – season-based fixed percentages (Eqs
          233, 235)
        - ``'Sagebrush'`` / ``'SB'`` – season-based fixed percentages (Eqs
          232, 233)
        - ``'Flatwood'`` / ``'Pine Flatwoods'`` / ``'PFL'`` / ``'PinFltwd'``
          – Eq 236
        - ``'Shrub'`` / ``'SG'`` / ``'ShrubGroup'`` – Eq 231 (80%)
        - All others – Eq 23 (60%)

    :param pre_sl: Pre-fire shrub fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param season: Burn season: ``'Spring'``, ``'Summer'``, ``'Fall'``, or
        ``'Winter'``. Required for Pocosin, Sagebrush, and Flatwoods
        equations.
    :param pre_ll: Pre-fire litter load (same units as ``pre_sl``). Required
        for the Southeast non-Pocosin equation (Eq 234). Scalar or
        np.ndarray.
    :param pre_dl: Pre-fire duff load (same units as ``pre_sl``). Required
        for the Southeast non-Pocosin equation. Scalar or np.ndarray.
    :param pre_rl: Pre-fire regeneration load (same units as ``pre_sl``).
        Required for the Southeast non-Pocosin equation. Scalar or
        np.ndarray.
    :param duff_moist: Duff moisture content (%). Required for the Southeast
        non-Pocosin equation. Scalar or np.ndarray.
    :param llc: Litter load consumed — output from :func:`consume_litter`
        (same units as ``pre_sl``). Required for the Southeast non-Pocosin
        equation. Scalar or np.ndarray.
    :param ddc: Duff depth consumed — output from :func:`consume_duff`
        (inches). Required for the Southeast non-Pocosin equation. Scalar or
        np.ndarray.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Percent shrub load consumed (%). Scalar ``float`` when all
        numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_sl)

    pre_sl = np.atleast_1d(np.asarray(pre_sl, dtype=float))

    if units == 'SI':
        pre_sl = pre_sl * 4.4609
        if pre_ll is not None:
            pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float)) * 4.4609
        if pre_dl is not None:
            pre_dl = np.atleast_1d(np.asarray(pre_dl, dtype=float)) * 4.4609
        if pre_rl is not None:
            pre_rl = np.atleast_1d(np.asarray(pre_rl, dtype=float)) * 4.4609
    else:
        if pre_ll is not None:
            pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
        if pre_dl is not None:
            pre_dl = np.atleast_1d(np.asarray(pre_dl, dtype=float))
        if pre_rl is not None:
            pre_rl = np.atleast_1d(np.asarray(pre_rl, dtype=float))

    if duff_moist is not None:
        duff_moist = np.atleast_1d(np.asarray(duff_moist, dtype=float))
    if llc is not None:
        llc = np.atleast_1d(np.asarray(llc, dtype=float))
    if ddc is not None:
        ddc = np.atleast_1d(np.asarray(ddc, dtype=float))

    slc = np.full_like(pre_sl, np.nan)

    if reg in ['Southeast']:
        if cvr_grp in ['Pocosin', 'PC']:
            if season in ['Spring', 'Winter']:
                # Equation 233
                slc = np.full_like(pre_sl, 90.0)
            elif season in ['Summer', 'Fall']:
                # Equation 235
                slc = np.full_like(pre_sl, 80.0)
        else:
            # Equation 234
            combo = pre_ll + pre_dl
            slc = (((3.2484 + (0.4322 * combo) + (0.6765 * (pre_sl + pre_rl)) -
                     (0.0276 * duff_moist) - (5.0796 / combo)) -
                    (llc + ddc)) / (pre_sl + pre_rl)) * 100
    else:
        if cvr_grp in ['Sagebrush', 'SB']:
            if season in ['Fall']:
                # Equation 233
                slc = np.full_like(pre_sl, 90.0)
            else:
                # Equation 232
                slc = np.full_like(pre_sl, 50.0)
        elif cvr_grp in ['Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd']:
            season_flag = 1.0 if season in ['Spring', 'Summer'] else 0.0
            # Equation 236
            slc = -0.1889 + (0.9049 * np.log(pre_sl)) + (0.0676 * season_flag)
        elif cvr_grp in ['Shrub', 'SG', 'ShrubGroup']:
            # Equation 231
            slc = np.full_like(pre_sl, 80.0)
        else:
            # Equation 23 – default
            slc = np.full_like(pre_sl, 60.0)

    return float(slc[0]) if scalar_input else slc


def gen_burnup_in_file(
        out_brn_path=None,
        max_times=3000,
        intensity=50.0,
        ig_time=60.0,
        windspeed=0.0,
        depth=0.3,
        ambient_temp=27.0,
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
    """
    Function generates a Burnup-in.brn file from the input parameters, and saves it at the out_brn_path location\n

    Required parameters: out_brn_path\n
    Optional parameters: All other inputs are not required if using default values. Replace otherwise.
    :param out_brn_path: folder/directory to save Burnup-in.brn file
    :param max_times: Maximum number of iterations burnup does (default = 3000); valid range: 1 - 100000
    :param intensity: Intensity of the igniting surface fire (kW/m)
                       (default = 50); valid range: 40 - 100000 kW/m
    :param ig_time: Residence time of the ignition surface fire (s)
                     (default = 60, FOFEM's burnup input default = 30); valid range: 10 - 200 s
    :param windspeed: Windspeed at the top of the fuelbed (m/s) (default = 0); valid range: 0 - 5 m/s
    :param depth: Fuel depth (m) (default = 0.3); valid range: 0.1 - 5 m
    :param ambient_temp: Ambient air temperature (C) (default = 27); valid range: -40 - 50 C
    :param r0: Fire environment minimum dimension parameter (unitless) (default = 1.83); valid range: any
    :param dr: Fire environment increment temp parameter (C) (default = 0.4); valid range: any
    :param timestep: Time step for integration of burning rates (s) (default = 15); valid range: any
    :param surat_lit: Surface area to volume ratio of litter
    :param surat_dw1: Surface area to volume ratio of 1 hr down woody fuels
    :param surat_dw10: Surface area to volume ratio of 10 hr down woody fuels
    :param surat_dw100: Surface area to volume ratio of 100 hr down woody fuels
    :param surat_dwk_3_6: Surface area to volume ratio of down woody fuels 3 - 6 in. diameter
    :param surat_dwk_6_9: Surface area to volume ratio of down woody fuels 6 - 9 in. diameter
    :param surat_dwk_9_20: Surface area to volume ratio of down woody fuels 9 - 20 in. diameter
    :param surat_dwk_20: Surface area to volume ratio of down woody fuels >= 20 in. diameter
    :return: Burnup-in.brn file\n\n
    """
    if out_brn_path is None:
        raise Exception('No output path specified for Burnup-in.brn file')

    # Validate input ranges
    max_times = max(1, min(max_times, 100000))
    intensity = max(40, min(intensity, 100000))
    ig_time = max(10, min(ig_time, 200))
    windspeed = max(0, min(windspeed, 5))
    depth = max(0.1, min(depth, 5))
    ambient_temp = max(-40, min(ambient_temp, 50))

    # Prepare the data as a list of tuples (parameter name, value)
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

    # Format each line as '#param value'
    lines = [f'#{name} {value}' for name, value in params]
    content = '\n'.join(lines)

    with open(out_brn_path, 'w') as f:
        f.write(content)

    return


def mort_bolchar(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    char_ht: Union[float, np.ndarray],
    tree_code_dict: dict = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM bole char post-fire mortality model (BOLCHAR).

    Based on Keyser (2018). Accepts a single tree (scalar inputs) or multiple
    trees (array inputs) of equal length. Models are available for the 10
    broadleaf species listed below; unsupported species codes return ``np.nan``
    with a printed warning.

    Available species:
        - Red maple        – RURU5, ACRU, ACRUD, ACRUD2, ACRUR, ACRUT2, ACRUT, ACRUT3
        - Flowering dogwood – COFL2
        - Blackgum         – NYSY, NYSYB, NYSYC, NYSYD, NYSYT, NYSYU, NYUR2, NYBI
        - Sourwood         – OXAR
        - White oak        – QUAL, QUALS, QUALS2, QUAL3, QUBI, QUGA4, QUGAG2, QUGAS
        - Scarlet oak      – QUCO2, QUCOC, QUCOT
        - Blackjack oak    – QUMA3, QUMAA2, QUMAA, QUMAM2
        - Chestnut oak     – QUMI, QUPR4
        - Black oak        – QUVE, QUVEM, QUKE
        - Sassafras        – SAAL5

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm), measured at 1.3 m above ground.
        Scalar float or np.ndarray.
    :param char_ht: Bole char height (m), measured in the field. Scalar float
        or np.ndarray.
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{316: 'ACRU'}``).

    :return: Mortality probability (float in [0, 1], or ``np.nan`` for
        unsupported species). Returns a scalar ``float`` when all inputs are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = _is_scalar(spp) and _is_scalar(dbh) and _is_scalar(char_ht)

    # Verify tree_code_dict
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings. '
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Coerce all inputs to np.ndarray
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh, dtype=float))
    char_ht = np.atleast_1d(np.asarray(char_ht, dtype=float))

    # Map numeric spp codes to FOFEM string codes if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Output array – NaN by default (unsupported species remain NaN)
    Pm = np.full(len(spp), np.nan)

    # --- Species masks ---
    mask_acru  = np.isin(spp, ['RURU5', 'ACRU', 'ACRUD', 'ACRUD2', 'ACRUR', 'ACRUT2', 'ACRUT', 'ACRUT3'])
    mask_cofl2 = spp == 'COFL2'
    mask_nysy  = np.isin(spp, ['NYSY', 'NYSYB', 'NYSYC', 'NYSYD', 'NYSYT', 'NYSYU', 'NYUR2', 'NYBI'])
    mask_oxar  = spp == 'OXAR'
    mask_qual  = np.isin(spp, ['QUAL', 'QUALS', 'QUALS2', 'QUAL3', 'QUBI', 'QUGA4', 'QUGAG2', 'QUGAS'])
    mask_quco2 = np.isin(spp, ['QUCO2', 'QUCOC', 'QUCOT'])
    mask_quma3 = np.isin(spp, ['QUMA3', 'QUMAA2', 'QUMAA', 'QUMAM2'])
    mask_qumi  = np.isin(spp, ['QUMI', 'QUPR4'])
    mask_quve  = np.isin(spp, ['QUVE', 'QUVEM', 'QUKE'])
    mask_saal5 = spp == 'SAAL5'

    # FOFEM Eq 100 - Red Maple
    if np.any(mask_acru):
        Pm[mask_acru] = 1 / (1 + np.exp(
            -(2.3017 + (-0.3267 * dbh[mask_acru]) + (1.1137 * char_ht[mask_acru]))))

    # FOFEM Eq 101 - Flowering Dogwood
    if np.any(mask_cofl2):
        Pm[mask_cofl2] = 1 / (1 + np.exp(
            -(-0.8727 + (-0.1814 * dbh[mask_cofl2]) + (4.1947 * char_ht[mask_cofl2]))))

    # FOFEM Eq 102 - Blackgum
    if np.any(mask_nysy):
        Pm[mask_nysy] = 1 / (1 + np.exp(
            -(-2.7899 + (-0.5511 * dbh[mask_nysy]) + (1.2888 * char_ht[mask_nysy]))))

    # FOFEM Eq 103 - Sourwood
    if np.any(mask_oxar):
        Pm[mask_oxar] = 1 / (1 + np.exp(
            -(1.9438 + (-0.4602 * dbh[mask_oxar]) + (1.6352 * char_ht[mask_oxar]))))

    # FOFEM Eq 104 - White Oak
    if np.any(mask_qual):
        Pm[mask_qual] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_qual]) + (0.8666 * char_ht[mask_qual]))))

    # FOFEM Eq 105 - Scarlet Oak
    if np.any(mask_quco2):
        Pm[mask_quco2] = 1 / (1 + np.exp(
            -(-1.6262 + (-0.0339 * dbh[mask_quco2]) + (0.6901 * char_ht[mask_quco2]))))

    # FOFEM Eq 106 - Blackjack Oak
    if np.any(mask_quma3):
        Pm[mask_quma3] = 1 / (1 + np.exp(
            -(0.3714 + (-0.1005 * dbh[mask_quma3]) + (1.5577 * char_ht[mask_quma3]))))

    # FOFEM Eq 107 - Chestnut Oak
    if np.any(mask_qumi):
        Pm[mask_qumi] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_qumi]) + (0.8666 * char_ht[mask_qumi]))))

    # FOFEM Eq 108 - Black Oak
    if np.any(mask_quve):
        Pm[mask_quve] = 1 / (1 + np.exp(
            -(0.1122 + (-0.1287 * dbh[mask_quve]) + (1.2612 * char_ht[mask_quve]))))

    # FOFEM Eq 109 - Sassafras
    if np.any(mask_saal5):
        Pm[mask_saal5] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_saal5]) + (0.8666 * char_ht[mask_saal5]))))

    # Warn about any unsupported species
    mask_supported = (mask_acru | mask_cofl2 | mask_nysy | mask_oxar | mask_qual |
                      mask_quco2 | mask_quma3 | mask_qumi | mask_quve | mask_saal5)
    if np.any(~mask_supported):
        unsupported = np.unique(spp[~mask_supported])
        print(f'Warning: BOLCHAR mortality model unavailable for species: {unsupported.tolist()}. '
              f'Mortality set to np.nan for those trees.')

    return float(Pm[0]) if scalar_input else Pm


def mort_crnsch(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray],
    bark_thickness: Optional[Union[float, np.ndarray]] = None,
    fire_intensity: Optional[Union[float, np.ndarray]] = None,
    amb_t: Optional[Union[float, np.ndarray]] = None,
    flame_length: Optional[Union[float, np.ndarray]] = None,
    char_ht: Optional[Union[float, np.ndarray]] = None,
    scorch_ht: Optional[Union[float, np.ndarray]] = None,
    instand_ws: Optional[Union[float, np.ndarray]] = None,
    aspen_sev: str = 'low',
    tree_code_dict: dict = None
) -> Union[float, np.ndarray]:
    """
    FOFEM crown scorch mortality model (CRNSCH).

    Accepts a single tree (scalar inputs) or multiple trees (array inputs) of
    equal length. Species without a dedicated equation fall back to the general
    bark-thickness model (FOFEM Eq 1).

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm). Scalar or np.ndarray.
    :param ht: Total tree height (m). Scalar or np.ndarray.
    :param crown_depth: Crown depth (m). Used with scorch height to derive
        percent crown volume/length scorched. Scalar or np.ndarray.
    :param bark_thickness: Bark thickness (cm). Scalar or np.ndarray. Optional;
        if ``None``, estimated from species and DBH using the lookup table.
    :param fire_intensity: Surface fire intensity (kW/m). Scalar or np.ndarray.
        Optional; used to derive flame length, char height, and/or scorch height
        when those are not supplied.
    :param amb_t: Ambient air temperature (°C). Scalar or np.ndarray. Default
        25 °C. Used when estimating scorch height.
    :param flame_length: Flame length (m). Scalar or np.ndarray. Optional; if
        not provided, derived from ``fire_intensity`` or ``char_ht``.
    :param char_ht: Char height (m). Scalar or np.ndarray. Optional; if not
        provided, derived from ``flame_length``.
    :param scorch_ht: Scorch height (m). Scalar or np.ndarray. Optional; if not
        provided, estimated from ``fire_intensity``, ``amb_t``, and
        ``instand_ws``.
    :param instand_ws: Instantaneous windspeed (m/s). Scalar or np.ndarray.
        Default 1 m/s. Used when estimating scorch height.
    :param aspen_sev: Aspen severity class for equation selection; ``'low'`` or
        ``'high'``. Default ``'low'``.
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{201: 'PIPO'}``).

    :return: Mortality probability (float in [0, 1]). Returns a scalar ``float``
        when all primary inputs (``spp``, ``dbh``, ``ht``, ``crown_depth``) are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = (_is_scalar(spp) and _is_scalar(dbh) and _is_scalar(ht)
                    and _is_scalar(crown_depth))

    # Verify tree_code_dict is dictionary if provided
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings.'
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Ensure all inputs are np.ndarrays (at least 1-D)
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh))
    ht = np.atleast_1d(np.asarray(ht))
    crown_depth = np.atleast_1d(np.asarray(crown_depth))
    if bark_thickness is not None:
        bark_thickness = np.atleast_1d(np.asarray(bark_thickness))
    if fire_intensity is not None:
        fire_intensity = np.atleast_1d(np.asarray(fire_intensity))
    if amb_t is None:
        amb_t = np.full(len(spp), 25.0)
    else:
        amb_t = np.atleast_1d(np.asarray(amb_t))
    if flame_length is not None:
        flame_length = np.atleast_1d(np.asarray(flame_length))
    if char_ht is not None:
        char_ht = np.atleast_1d(np.asarray(char_ht))
    if scorch_ht is not None:
        scorch_ht = np.atleast_1d(np.asarray(scorch_ht))
    if instand_ws is None:
        instand_ws = np.ones(len(spp))
    else:
        instand_ws = np.atleast_1d(np.asarray(instand_ws))

    # Map numeric spp to FOFEM_sppCD if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Fill missing bark_thickness
    if bark_thickness is None:
        bark_thickness = calc_bark_thickness(spp, dbh)
    # Fill missing flame_length/char_ht
    if (flame_length is None) and (char_ht is None) and (fire_intensity is None):
        raise Exception('The CRNSCH mortality model requires either flame length, char height or surface fire intensity (kW/m) as an input')
    if (fire_intensity is not None) and (flame_length is None) and (char_ht is None):
        flame_length = calc_flame_length(fire_intensity, char_ht)
        char_ht = calc_char_ht(flame_length)
    if ((fire_intensity is not None) or (char_ht is not None)) and (flame_length is None):
        flame_length = calc_flame_length(fire_intensity, char_ht)
    if (flame_length is not None) and (char_ht is None):
        char_ht = calc_char_ht(flame_length)
    if scorch_ht is None:
        scorch_ht = calc_scorch_ht(fire_intensity, amb_t, instand_ws)

    # Calculate cvs, cls
    _, cvs, cls = calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    # Output array
    Pm = np.zeros(len(spp), dtype=float)

    # Masks for species
    mask_abco = np.isin(spp, ['ABCO', 'ABCOC'])
    mask_abgr = np.isin(spp, ['ABGR', 'ABGRI2', 'ABGRG', 'ABGRI', 'ABGRJ', 'ABLA', 'ABLAL'])
    mask_abma = spp == 'ABMA'
    mask_cade = np.isin(spp, ['CADE27', 'LIDE'])
    mask_laoc = spp == 'LAOC'
    mask_pial = np.isin(spp, ['PIAL', 'PICO', 'PICOL', 'PICOL2'])
    mask_spruce = np.isin(spp, ['PICSPP', 'PIMA', 'PIMAM4', 'PIPU', 'PIPUA', 'PIPUG3', 'PIAB', 'PIRU', 'PISI', 'PIGL'])
    mask_pien = np.isin(spp, ['PIEN', 'PIENE', 'PIENM', 'PIENM2'])
    mask_pila = spp == 'PILA'
    mask_pipa2 = spp == 'PIPA2'
    mask_pipo = np.isin(spp, [
        'PIPO', 'PIPOK', 'PIPOB', 'PIPOBK', 'PIPOB2', 'PIPOB3', 'PIPOB3K',
        'PIPOP', 'PIPOPK', 'PIPOP2', 'PIPOP2K', 'PIPOS', 'PIPOSK', 'PIPOS2', 'PIPOS2K',
        'PIJE', 'PIJEK'
    ])
    mask_pipo_bh = spp == 'PIPO_BH'
    mask_aspen = np.isin(spp, ['POTR12', 'POTR5', 'POTRA', 'POTRC2', 'POTRM', 'POTRR', 'POTRV'])
    mask_psme = np.isin(spp, ['PSME', 'PSMEF', 'PSMEM'])

    # FOFEM Eq 10 - White Fir
    if np.any(mask_abco):
        Pm[mask_abco] = 1 / (1 + np.exp(-(-3.5083 +
                                          (cls[mask_abco] * 0.0956) -
                                          (np.power(cls[mask_abco], 2) * 0.00184) +
                                          (np.power(cls[mask_abco], 3) * 0.000017))))
    # FOFEM Eq 11 - Subalpine/Grand Fir
    if np.any(mask_abgr):
        Pm[mask_abgr] = 1 / (1 + np.exp(-(-1.6950 +
                                          (cvs[mask_abgr] * 0.2071) -
                                          (np.power(cvs[mask_abgr], 2) * 0.0047) +
                                          (np.power(cvs[mask_abgr], 3) * 0.000035))))
    # FOFEM Eq 16 - Red Fir
    if np.any(mask_abma):
        Pm[mask_abma] = 1 / (1 + np.exp(-(-2.3085 + (np.power(cls[mask_abma], 3) * 0.000004059))))
    # FOFEM Eq 12 - Incense Cedar
    if np.any(mask_cade):
        Pm[mask_cade] = 1 / (1 + np.exp(-(-4.2466 + (np.exp(cls[mask_cade], 3) * 0.000007172))))
    # FOFEM Eq 14 - Western Larch
    if np.any(mask_laoc):
        Pm[mask_laoc] = 1 / (1 + np.exp(-(-1.6594 + (cvs[mask_laoc] * 0.0327) - (dbh[mask_laoc] * 0.0489))))
    # FOFEM Eq 17 - Whitebark/Logepole Pine
    if np.any(mask_pial):
        Pm[mask_pial] = 1 / (1 + np.exp(-(-0.3268 +
                                          (cvs[mask_pial] * 0.1387) -
                                          (np.power(cvs[mask_pial], 2) * 0.0033) +
                                          (np.power(cvs[mask_pial], 3) * 0.000025) -
                                          (dbh[mask_pial] * 0.0266))))
    # FOFEM Eq 3 - All other spruce species
    if np.any(mask_spruce):
        dbh_in = dbh[mask_spruce] / 2.54
        _Pm = 1 / (1 + np.exp(-1.941 +
                              (6.316 * (1 - np.exp(-bark_thickness[mask_spruce] / 2.54))) -
                              (np.power(cvs[mask_spruce], 2) * 0.000535)))
        _Pm = np.where(dbh_in >= 1, np.maximum(_Pm, 0.8), _Pm)
        _Pm = np.where((dbh_in < 1) & (cls[mask_spruce] > 50), 1, _Pm)
        _Pm = np.where((dbh_in < 1) & (ht[mask_spruce] < (3 * 0.3048)), 1, _Pm)
        Pm[mask_spruce] = _Pm
    # FOFEM Eq 15 - Engelmann spruce
    if np.any(mask_pien):
        Pm[mask_pien] = 1 / (1 + np.exp(-(0.0845 + (cvs[mask_pien] * 0.0445))))
    # FOFEM Eq 18 - Sugar Pine
    if np.any(mask_pila):
        Pm[mask_pila] = 1 / (1 + np.exp(-(-2.0588 + (np.power(cls[mask_pila], 2) * 0.000814))))
    # FOFEM Eq 5 - Longleaf Pine
    if np.any(mask_pipa2):
        barkT = 0.435 + (0.031 * dbh[mask_pipa2])
        Pm[mask_pipa2] = np.where(
            scorch_ht[mask_pipa2] == 0,
            0,
            1 / (1 + np.exp(0.169 +
                            (5.136 * barkT) +
                            (14.429 * np.power(barkT, 2)) -
                            (0.348 * np.power(cvs[mask_pipa2] / 100, 2))))
        )
    # FOFEM Eq 19 - Ponderosa/Jeffrey Pine
    Pm[mask_pipo] = 1 / (1 + np.exp(-(-2.7103 + (np.power(cvs[mask_pipo], 3) * 0.000004093))))
    # FOFEM Eq 21 - Black Hills Ponderosa Pine
    if np.any(mask_pipo_bh):
        ht_bh = ht[mask_pipo_bh]
        dbh_bh = dbh[mask_pipo_bh]
        flame_bh = flame_length[mask_pipo_bh]
        cls_bh = cls[mask_pipo_bh]
        cbh_bh = ht_bh - crown_depth[mask_pipo_bh]
        scorch_ht_bh = scorch_ht[mask_pipo_bh]
        # Seedlings
        mask_seed = ht_bh <= 1.37
        Pm[mask_pipo_bh] = np.where(
            mask_seed,
            1 / (1 + np.exp(-(2.714 + (4.08 * flame_bh) - (3.63 * ht_bh)))),
            Pm[mask_pipo_bh]
        )
        # Saplings
        mask_sap = (ht_bh > 1.37) & (dbh_bh < 10.2)
        Pm[mask_pipo_bh] = np.where(
            mask_sap,
            1 / (1 + np.exp(-(-0.7661 + (2.7981 * flame_bh) - (1.2487 * ht_bh)))),
            Pm[mask_pipo_bh]
        )
        # Trees
        mask_tree = dbh_bh >= 10.2
        cls_tree = ((scorch_ht_bh - cbh_bh) / (ht_bh - cbh_bh)) * 100
        Pm[mask_pipo_bh] = np.where(
            mask_tree,
            1 / (1 + np.exp(-(1.104 - (dbh_bh * 0.156) + (0.013 * cls_tree) + (0.001 * dbh_bh * cls_tree)))),
            Pm[mask_pipo_bh]
        )
    # FOFEM Eq 4 - Aspen
    if np.any(mask_aspen):
        dbh_a = dbh[mask_aspen]
        char_ht_a = char_ht[mask_aspen]
        Pm[mask_aspen] = np.where(
            aspen_sev == 'low',
            1 / (1 + np.exp((0.251 * dbh_a) - (0.07 * char_ht_a * 12) - 4.407)),
            1 / (1 + np.exp((0.0858 * dbh_a) - (0.118 * char_ht_a * 12) - 2.157))
        )
    # FOFEM Eq 20 - Douglas-fir
    if np.any(mask_psme):
        Pm[mask_psme] = 1 / (1 + np.exp(-(-2.0346 +
                                          (cvs[mask_psme] * 0.0906) -
                                          (np.power(cvs[mask_psme], 2) * 0.0022) +
                                          (np.power(cvs[mask_psme], 3) * 0.000019))))
    # FOFEM Eq 1 - All other species
    mask_other = ~(mask_abco | mask_abgr | mask_abma | mask_cade | mask_laoc |
                   mask_pial | mask_spruce | mask_pien | mask_pila | mask_pipa2 |
                   mask_pipo | mask_pipo_bh | mask_aspen | mask_psme)
    if np.any(mask_other):
        dbh_in = dbh[mask_other] / 2.54
        _Pm = 1 / (1 + np.exp(-1.941 +
                              (6.316 * (1 - np.exp(-bark_thickness[mask_other] / 2.54))) -
                              (np.power(cvs[mask_other], 2) * 0.000535)))
        _Pm = np.where(dbh_in >= 1, _Pm, _Pm)
        _Pm = np.where((dbh_in < 1) & (cls[mask_other] > 50), 1, _Pm)
        _Pm = np.where((dbh_in < 1) & (ht[mask_other] < (3 * 0.3048)), 1, _Pm)
        Pm[mask_other] = _Pm

    return float(Pm[0]) if scalar_input else Pm


def mort_crcabe(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray],
    ckr: Union[float, np.ndarray],
    scorch_ht: Union[float, np.ndarray],
    beetles: Union[bool, np.ndarray] = False,
    cvk: Optional[Union[float, np.ndarray]] = None,
    tree_code_dict: dict = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM cambium kill / post-fire mortality model (CRCABE).

    Based on Hood and Lutes (2017). Accepts a single tree (scalar inputs) or
    multiple trees (array inputs) of equal length. Models are available for the
    12 conifer species listed below; unsupported species codes return ``np.nan``
    with a printed warning.

    Available species:
        - White fir              – ABCO, ABCOC
        - Grand/Subalpine fir    – ABGR, ABGRI2, ABGRG, ABGRI, ABGRJ, ABLA, ABLAL
        - Red fir                – ABMA
        - Incense Cedar          – CADE27, LIDE
        - Engelmann spruce       – PIEN, PIENE, PIENM, PIENM2
        - Western Larch          – LAOC
        - Douglas-fir            – PSME, PSMEF, PSMEM
        - Whitebark/Lodgepole pine – PIAL, PICO, PICOL, PICOL2
        - Sugar pine             – PILA
        - Ponderosa/Jeffrey pine – PIPO, PIPOK, PIPOB, PIPOBK, PIPOB2, PIPOB3,
          PIPOB3K, PIPOP, PIPOPK, PIPOP2, PIPOP2K, PIPOS, PIPOSK, PIPOS2,
          PIPOS2K, PIPO_BH, PIJE, PIJEK

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm). Scalar or np.ndarray.
    :param ht: Total tree height (m). Scalar or np.ndarray.
    :param crown_depth: Crown depth (m). Scalar or np.ndarray.
    :param ckr: Cambium Kill Rating (0–4), measured in the field. Scalar or
        np.ndarray.
    :param scorch_ht: Scorch height (m). Scalar or np.ndarray.
    :param beetles: Beetle attack status. A single ``bool`` (applied to all
        trees) or a boolean np.ndarray of the same length as ``spp``. Default
        ``False``. Species-specific ``atk`` factor values are assigned
        internally. Relevant beetle species: Ambrosia, Red turpentine, Mountain
        pine, Douglas-fir beetle, IPS.
    :param cvk: Percent total crown volume killed by bud kill (%). Used only
        for Ponderosa/Jeffrey pine; selects the ``PK`` (bud-kill) equation over
        the ``PP`` (scorch) equation when provided. Scalar or np.ndarray of the
        same length as ``spp``. Default ``None`` (uses scorch-based equation).
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{201: 'PIPO'}``).

    :return: Mortality probability (float in [0, 1], or ``np.nan`` for
        unsupported species). Returns a scalar ``float`` when all primary inputs
        (``spp``, ``dbh``, ``ht``, ``crown_depth``, ``ckr``, ``scorch_ht``) are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = (_is_scalar(spp) and _is_scalar(dbh) and _is_scalar(ht)
                    and _is_scalar(crown_depth) and _is_scalar(ckr)
                    and _is_scalar(scorch_ht))

    # Verify tree_code_dict
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings. '
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Coerce all inputs to np.ndarray (at least 1-D)
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh))
    ht = np.atleast_1d(np.asarray(ht))
    crown_depth = np.atleast_1d(np.asarray(crown_depth))
    ckr = np.atleast_1d(np.asarray(ckr))
    scorch_ht = np.atleast_1d(np.asarray(scorch_ht))

    # Broadcast beetles to a per-tree boolean array
    beetles = np.broadcast_to(np.asarray(beetles, dtype=bool), spp.shape).copy()

    # Broadcast cvk to a per-tree float array (NaN where not provided)
    if cvk is None:
        cvk_arr = np.full(spp.shape, np.nan)
    else:
        cvk_arr = np.broadcast_to(np.asarray(cvk, dtype=float), spp.shape).copy()

    # Map numeric spp codes to FOFEM string codes if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Calculate crown volume scorched (cvs, %) and crown length scorched (cls, %)
    _, cvs, cls = calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    # Output array – NaN by default (unsupported species remain NaN)
    Pm = np.full(len(spp), np.nan)

    # --- Species masks ---
    mask_abco = np.isin(spp, ['ABCO', 'ABCOC'])
    mask_abgr = np.isin(spp, ['ABGR', 'ABGRI2', 'ABGRG', 'ABGRI', 'ABGRJ', 'ABLA', 'ABLAL'])
    mask_abma = spp == 'ABMA'
    mask_cade = np.isin(spp, ['CADE27', 'LIDE'])
    mask_pien = np.isin(spp, ['PIEN', 'PIENE', 'PIENM', 'PIENM2'])
    mask_laoc = spp == 'LAOC'
    mask_psme = np.isin(spp, ['PSME', 'PSMEF', 'PSMEM'])
    mask_pial = np.isin(spp, ['PIAL', 'PICO', 'PICOL', 'PICOL2'])
    mask_pila = spp == 'PILA'
    mask_pipo = np.isin(spp, [
        'PIPO', 'PIPOK', 'PIPOB', 'PIPOBK', 'PIPOB2', 'PIPOB3', 'PIPOB3K',
        'PIPOP', 'PIPOPK', 'PIPOP2', 'PIPOP2K', 'PIPOS', 'PIPOSK', 'PIPOS2', 'PIPOS2K', 'PIPO_BH',
        'PIJE', 'PIJEK'
    ])

    # FOFEM Eq WF - White Fir (ambrosia beetle; atk: attacked=1, unattacked=-1)
    if np.any(mask_abco):
        atk = np.where(beetles[mask_abco], 1, -1).astype(float)
        Pm[mask_abco] = 1 / (1 + np.exp(
            -(-3.5964 + (np.power(cls[mask_abco], 3) * 0.00000628) +
              (ckr[mask_abco] * 0.3019) + (dbh[mask_abco] * 0.019) + (atk * 0.5209))))

    # FOFEM Eq SF - Grand Fir and Subalpine Fir
    if np.any(mask_abgr):
        Pm[mask_abgr] = 1 / (1 + np.exp(
            -(-2.6036 + (np.power(cvs[mask_abgr], 3) * 0.000004587) + (ckr[mask_abgr] * 1.3554))))

    # FOFEM Eq RF - Red Fir
    if np.any(mask_abma):
        Pm[mask_abma] = 1 / (1 + np.exp(
            -(-4.7515 + (np.power(cls[mask_abma], 3) * 0.000005989) + (ckr[mask_abma] * 1.0668))))

    # FOFEM Eq IC - Incense Cedar
    if np.any(mask_cade):
        Pm[mask_cade] = 1 / (1 + np.exp(
            -(-5.6465 + (np.power(cls[mask_cade], 3) * 0.000007274) + (ckr[mask_cade] * 0.5428))))

    # FOFEM Eq ES - Engelmann Spruce
    if np.any(mask_pien):
        Pm[mask_pien] = 1 / (1 + np.exp(
            -(-2.9791 + (cvs[mask_pien] * 0.0405) + (ckr[mask_pien] * 1.1596))))

    # FOFEM Eq WL - Western Larch
    if np.any(mask_laoc):
        Pm[mask_laoc] = 1 / (1 + np.exp(
            -(-3.8458 + (np.power(cvs[mask_laoc], 2) * 0.0004) + (ckr[mask_laoc] * 0.6266))))

    # FOFEM Eq DF - Douglas-fir (Douglas-fir beetle; atk: attacked=1, unattacked=0)
    if np.any(mask_psme):
        atk = np.where(beetles[mask_psme], 1, 0).astype(float)
        Pm[mask_psme] = 1 / (1 + np.exp(
            -(-1.8912 + (cvs[mask_psme] * 0.07) -
              (np.power(cvs[mask_psme], 2) * 0.0019) +
              (np.power(cvs[mask_psme], 3) * 0.000018) +
              (ckr[mask_psme] * 0.5840) - (dbh[mask_psme] * 0.031) -
              (atk * 0.7959) + (dbh[mask_psme] * atk * 0.0492))))

    # FOFEM Eq WP - Whitebark Pine and Lodgepole Pine
    if np.any(mask_pial):
        Pm[mask_pial] = 1 / (1 + np.exp(
            -(-1.4059 + (np.power(cvs[mask_pial], 3) * 0.000004459) +
              (np.power(ckr[mask_pial], 2) * 0.2843) - (dbh[mask_pial] * 0.0485))))

    # FOFEM Eq SP - Sugar Pine (red turpentine / mountain pine beetle; atk: attacked=1, unattacked=-1)
    if np.any(mask_pila):
        atk = np.where(beetles[mask_pila], 1, -1).astype(float)
        Pm[mask_pila] = 1 / (1 + np.exp(
            -(-2.7598 + (np.power(cls[mask_pila], 2) * 0.000642) +
              (np.power(ckr[mask_pila], 3) * 0.0386) + (atk * 0.8485))))

    # FOFEM Eq PP / PK - Ponderosa / Jeffrey Pine
    # (mountain pine, red turpentine, or ips beetle; atk: attacked=1, unattacked=0)
    # Uses PK (cvk-based) equation where cvk is provided, otherwise PP (scorch-based).
    if np.any(mask_pipo):
        atk = np.where(beetles[mask_pipo], 1, 0).astype(float)
        cvk_sub = cvk_arr[mask_pipo]
        has_cvk = ~np.isnan(cvk_sub)
        _Pm = np.empty(int(np.sum(mask_pipo)))
        # PP equation (scorch-based)
        _Pm[~has_cvk] = 1 / (1 + np.exp(
            -(-4.1914 + (np.power(cvs[mask_pipo][~has_cvk], 2) * 0.000376) +
              (ckr[mask_pipo][~has_cvk] * 0.5130) + (atk[~has_cvk] * 1.5873))))
        # PK equation (cvk/bud-kill-based)
        _Pm[has_cvk] = 1 / (1 + np.exp(
            -(-3.5729 + (np.power(cvk_sub[has_cvk], 2) * 0.000567) +
              (ckr[mask_pipo][has_cvk] * 0.4573) + (atk[has_cvk] * 1.6075))))
        Pm[mask_pipo] = _Pm

    # Warn about any unsupported species
    mask_supported = (mask_abco | mask_abgr | mask_abma | mask_cade | mask_pien |
                      mask_laoc | mask_psme | mask_pial | mask_pila | mask_pipo)
    mask_unsupported = ~mask_supported
    if np.any(mask_unsupported):
        unsupported = np.unique(spp[mask_unsupported])
        print(f'Warning: CRCABE mortality model unavailable for species: {unsupported.tolist()}. '
              f'Mortality set to np.nan for those trees.')

    return float(Pm[0]) if scalar_input else Pm


def run_burnup(
    fuel_loadings: Dict[str, float],
    fuel_moistures: Dict[str, float],
    intensity: float = 50.0,
    ig_time: float = 60.0,
    windspeed: float = 0.0,
    depth: float = 0.3,
    ambient_temp: float = 27.0,
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
    heat_content: float = 1.867e7,
    density: float = 513.0,
    heat_capacity: float = 2750.0,
    conductivity: float = 0.133,
    ignition_temp: float = 300.0,
    char_temp: float = 350.0,
    ash_content: float = 0.05,
    duff_loading: float = 0.0,
    duff_moisture: float = 2.0,
    fint_switch: float = 15.0,
    validate: bool = True,
) -> Tuple[List[BurnResult], List[BurnSummaryRow]]:
    """
    Run the BURNUP post-frontal combustion model.

    This is a convenience wrapper around the lower-level
    :func:`~pyfofem.components.burnup.burnup` engine.  It accepts fuel
    loadings and moistures keyed by familiar FOFEM size-class names
    (matching the terminology in :func:`gen_burnup_in_file`) and builds the
    required :class:`~pyfofem.components.burnup.FuelParticle` list
    internally.

    Only size classes with a loading > 0 are passed to the simulation.

    :param fuel_loadings: Oven-dry mass loading (kg/m²) per size class.
        Recognised keys (all optional; omitted or zero-valued classes are
        skipped):

        - ``'litter'`` – litter
        - ``'dw1'`` – 1-hr down woody (0–0.64 cm)
        - ``'dw10'`` – 10-hr down woody (0.64–2.54 cm)
        - ``'dw100'`` – 100-hr down woody (2.54–7.62 cm)
        - ``'dwk_3_6'`` – 1000-hr sound, 3–6 in. diameter
        - ``'dwk_6_9'`` – 1000-hr sound, 6–9 in. diameter
        - ``'dwk_9_20'`` – 1000-hr sound, 9–20 in. diameter
        - ``'dwk_20'`` – 1000-hr sound, ≥ 20 in. diameter

    :param fuel_moistures: Moisture content (fraction of dry weight) per size
        class.  Uses the same keys as *fuel_loadings*.  Any size class
        present in *fuel_loadings* but missing from *fuel_moistures* will
        use a default of 0.10 (10 %).
    :param intensity: Intensity of the igniting surface fire (kW/m²).
        Default 50.
    :param ig_time: Residence time of the igniting surface fire (s).
        Default 60.
    :param windspeed: Windspeed at the top of the fuel bed (m/s).
        Default 0.
    :param depth: Fuel bed depth (m). Default 0.3.
    :param ambient_temp: Ambient air temperature (°C). Default 27.
    :param r0: Fire-environment minimum mixing parameter (dimensionless).
        Default 1.83.
    :param dr: Fire-environment mixing-parameter range (dimensionless).
        Default 0.4.
    :param timestep: Integration time step (s). Default 15.
    :param max_times: Maximum number of simulation time steps.
        Default 3000.
    :param surat_lit: Surface-area-to-volume ratio of litter (1/m).
        Default 8200.
    :param surat_dw1: Surface-area-to-volume ratio of 1-hr fuels (1/m).
        Default 1480.
    :param surat_dw10: Surface-area-to-volume ratio of 10-hr fuels (1/m).
        Default 394.
    :param surat_dw100: Surface-area-to-volume ratio of 100-hr fuels (1/m).
        Default 105.
    :param surat_dwk_3_6: Surface-area-to-volume ratio of 3–6 in. fuels
        (1/m). Default 39.4.
    :param surat_dwk_6_9: Surface-area-to-volume ratio of 6–9 in. fuels
        (1/m). Default 21.9.
    :param surat_dwk_9_20: Surface-area-to-volume ratio of 9–20 in. fuels
        (1/m). Default 12.7.
    :param surat_dwk_20: Surface-area-to-volume ratio of ≥ 20 in. fuels
        (1/m). Default 5.91.
    :param heat_content: Low heat of combustion (J/kg), applied to all
        size classes. Default 1.867e7.
    :param density: Oven-dry mass density (kg/m³), applied to all size
        classes. Default 513.
    :param heat_capacity: Specific heat capacity (J/kg·K), applied to all
        size classes. Default 2750.
    :param conductivity: Oven-dry thermal conductivity (W/m·K), applied to
        all size classes. Default 0.133.
    :param ignition_temp: Piloted-ignition temperature (°C), applied to
        all size classes. Default 300.
    :param char_temp: End-of-pyrolysis (char) temperature (°C), applied to
        all size classes. Default 350.
    :param ash_content: Mineral ash mass fraction, applied to all size
        classes. Default 0.05.
    :param duff_loading: Duff oven-dry loading (kg/m²). Default 0 (no
        duff).
    :param duff_moisture: Duff moisture content (fraction). Default 2.0
        (effectively suppresses duff burning).
    :param fint_switch: Flaming / smoldering intensity threshold (kW/m²).
        Default 15.
    :param validate: If True (default), run range checks on all inputs
        before simulation.
    :return: ``(results, summary)`` where *results* is a list of
        :class:`~pyfofem.components.burnup.BurnResult` (one per completed
        time step) and *summary* is a list of
        :class:`~pyfofem.components.burnup.BurnSummaryRow` (one per fuel
        component).
    :raises BurnupValidationError: If *validate* is True and any parameter
        is out of range.
    :raises ValueError: If no fuel size classes have loading > 0.
    """
    # Map of size-class key → surface-area-to-volume ratio
    _sigma_map: Dict[str, float] = {
        'litter':   surat_lit,
        'dw1':      surat_dw1,
        'dw10':     surat_dw10,
        'dw100':    surat_dw100,
        'dwk_3_6':  surat_dwk_3_6,
        'dwk_6_9':  surat_dwk_6_9,
        'dwk_9_20': surat_dwk_9_20,
        'dwk_20':   surat_dwk_20,
    }

    # Canonical ordering (finest → coarsest)
    _class_order = [
        'litter', 'dw1', 'dw10', 'dw100',
        'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    ]

    default_moisture = 0.10

    # Build FuelParticle list (only include classes with loading > 0)
    particles: List[FuelParticle] = []
    for key in _class_order:
        loading = fuel_loadings.get(key, 0.0)
        if loading <= 0.0:
            continue
        moisture = fuel_moistures.get(key, default_moisture)
        sigma = _sigma_map[key]
        particles.append(FuelParticle(
            wdry=loading,
            htval=heat_content,
            fmois=moisture,
            dendry=density,
            sigma=sigma,
            cheat=heat_capacity,
            condry=conductivity,
            tpig=ignition_temp,
            tchar=char_temp,
            ash=ash_content,
        ))

    if not particles:
        raise ValueError(
            'run_burnup requires at least one fuel size class with loading > 0. '
            f'Recognised keys: {_class_order}'
        )

    return _burnup(
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
        fint_switch=fint_switch,
        validate=validate,
    )
