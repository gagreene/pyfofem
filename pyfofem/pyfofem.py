# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13, 12:00:00 2025

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from numpy import ma as mask
from typing import Union, Optional


# Load species codes lookup table
import os
from pandas import read_csv
spp_codes = read_csv(os.path.join(os.path.dirname(__file__), 'species_codes_lut.csv'))


def _calc_scorch_ht(sfi, amb_t=None, instand_ws=None):
    """
    Van Wagner (1973) & Alexander (1982/85) lethal scorch height model.
    Accepts scalars or np.ndarray inputs.
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


def _calc_flame_length(fire_intensity=None, char_ht=None, fl_model='Byram'):
    """
    Flame length model (Byram, Butler, Thomas).
    Accepts scalars or np.ndarray inputs.
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
        else:
            return 0.026700 * np.power(fire_intensity, 2 / 3)

    if (fire_intensity is None) and (char_ht is not None):
        char_ht = np.asarray(char_ht)
        return char_ht * 1.8


def _calc_char_ht(flame_length):
    """
    Vectorized char height calculation.
    Accepts scalars or np.ndarray inputs.
    """
    flame_length = np.asarray(flame_length)
    return flame_length / 1.8


def _calc_bark_thickness(spp, dbh, tree_fuels_df):
    """
    Vectorized bark thickness calculation (cm).
    spp: array-like of species codes
    dbh: array-like of diameters (cm)
    tree_fuels_df: DataFrame with 'spp' and 'FOFEM_BrkThck_Vsp'
    """
    spp = np.asarray(spp)
    dbh = np.asarray(dbh)
    # Ensure spp is an array of strings for indexing
    spp_str = spp.astype(str)
    bark_thick_per_dbh = tree_fuels_df.set_index('spp').loc[spp_str, 'FOFEM_BrkThck_Vsp'].to_numpy()
    return bark_thick_per_dbh * dbh


def _calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth):
    """
    Vectorized calculation of crown length scorched, percent crown volume scorched, and percent crown length scorched.
    Accepts scalars or np.ndarray inputs.
    """
    scorch_ht = np.asarray(scorch_ht)
    ht = np.asarray(ht)
    crown_depth = np.asarray(crown_depth)
    crown_length_scorched = scorch_ht - (ht - crown_depth)
    crown_length_scorched = np.clip(crown_length_scorched, 0, crown_depth)
    cvs = 100 * (crown_length_scorched * ((2 * crown_depth) - crown_length_scorched) / np.power(crown_depth, 2))
    cls = 100 * (crown_length_scorched / crown_depth)
    return crown_length_scorched, cvs, cls


def _mort_crnsch(
    spp: np.ndarray,
    dbh: np.ndarray,
    ht: np.ndarray,
    crown_depth: np.ndarray,
    bark_thickness: np.ndarray = None,
    fire_intensity: np.ndarray = None,
    amb_t: np.ndarray = [25],
    flame_length: np.ndarray = None,
    char_ht: np.ndarray = None,
    scorch_ht: np.ndarray = None,
    instand_ws: np.ndarray = [1],
    aspen_sev: str = 'low',
    tree_params_df: pd.DataFrame = None
) -> np.ndarray:
    """
    FOFEM crown scorch mortality model.
    All inputs must be np.ndarrays of the same length.
    spp: array of species codes (str or int). If int, will be mapped to FOFEM_sppCD using tree_params_df.
    tree_params_df: DataFrame with columns 'sppNumCD' and 'FOFEM_sppCD' for code mapping.
    """
    n = len(spp)
    spp = np.array(spp)
    # Map numeric spp to FOFEM_sppCD if needed
    if np.issubdtype(spp.dtype, np.integer):
        spp_map = dict(zip(tree_params_df['sppNumCD'], tree_params_df['FOFEM_sppCD']))
        spp = np.vectorize(lambda x: spp_map.get(x, 'UNK'))(spp)
    else:
        spp = spp.astype(str)

    # Fill missing bark_thickness
    if bark_thickness is None:
        bark_thickness = _calc_bark_thickness(spp, dbh)
    # Fill missing flame_length/char_ht
    if (flame_length is None) and (char_ht is None) and (fire_intensity is None):
        raise Exception('The CRNSCH mortality model requires either flame length, char height or surface fire intensity (kW/m) as an input')
    if (fire_intensity is not None) and (flame_length is None) and (char_ht is None):
        flame_length = _calc_flame_length(fire_intensity, char_ht)
        char_ht = _calc_char_ht(flame_length)
    if ((fire_intensity is not None) or (char_ht is not None)) and (flame_length is None):
        flame_length = _calc_flame_length(fire_intensity, char_ht)
    if (flame_length is not None) and (char_ht is None):
        char_ht = _calc_char_ht(flame_length)
    if scorch_ht is None:
        scorch_ht = _calc_scorch_ht(fire_intensity, amb_t, instand_ws)

    # Calculate cvs, cls
    _, cvs, cls = _calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    # Output array
    Pm = np.zeros(n, dtype=float)

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
            sev == 'low',
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

    return Pm