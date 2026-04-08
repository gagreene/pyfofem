# -*- coding: utf-8 -*-
"""
tree_flame_calcs.py – Tree geometry, canopy, and fire-flame calculations.

Functions in this module compute intermediate fire-behaviour quantities
(flame length, char height, scorch height) and tree-level descriptors
(bark thickness, crown geometry, canopy cover) that feed into both the
mortality and consumption/emissions sub-models.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import os
import numpy as np
from pandas import read_csv, DataFrame
from typing import Dict, List, Optional, Tuple, Union

from ._component_helpers import _is_scalar, _maybe_scalar


# ---------------------------------------------------------------------------
# Canopy Cover
# ---------------------------------------------------------------------------

# Crown-width equation coefficients (Crookston & Stage 1999, RMRS-GTR-24).
# Keys are equation numbers; values are (A, B, R) where:
#   CW [ft] = A × DBH [in]^B   for trees > 4.5 ft (1.37 m) tall
#   CW [ft] = R × DBH [in]     for trees ≤ 4.5 ft (1.37 m) tall
_CANOPY_COEFFS: Dict[int, Tuple[float, float, float]] = {
     1: (3.9723, 0.5177, 0.473),   # Abies amabilis — Pacific silver fir
     2: (3.8166, 0.5229, 0.452),   # Abies balsamea/concolor/lowiana — Balsam/white/Sierra white fir
     3: (4.1870, 0.5341, 0.489),   # Abies grandis — Grand fir
     4: (3.2348, 0.5179, 0.385),   # Abies lasiocarpa — Subalpine fir
     5: (3.1146, 0.5780, 0.345),   # Abies magnifica — Red fir
     7: (3.0614, 0.6276, 0.320),   # Abies procera — Noble fir
     8: (3.5341, 0.5374, 0.331),   # Callitropsis nootkatensis — Alaska cedar
     9: (4.0920, 0.4912, 0.412),   # Chamaecyparis/Cupressus spp — Atlantic white/Port Orford cedar
    10: (3.6802, 0.4940, 0.412),   # Picea spp — Spruce spp
    11: (2.4132, 0.6403, 0.298),   # Pinus contorta group — Lodgepole/jack pine group
    12: (3.2367, 0.6247, 0.406),   # Pinus jeffreyi — Jeffrey pine
    13: (3.0610, 0.6201, 0.385),   # Pinus lambertiana — Sugar pine
    14: (3.4447, 0.5185, 0.476),   # Pinus monticola/strobus — Western/eastern white pine
    15: (2.8541, 0.6400, 0.407),   # Pinus ponderosa group — Ponderosa/slash/loblolly pine
    16: (4.4215, 0.5329, 0.517),   # Pseudotsuga menziesii — Douglas-fir
    17: (4.4215, 0.5329, 0.517),   # Sequoia gigantea/sempervirens — Giant sequoia/redwood
    18: (6.2318, 0.4259, 0.698),   # Thuja/Juniperus/Calocedrus spp — Redcedar/arborvitae/incense-cedar
    19: (5.4864, 0.5144, 0.533),   # Tsuga canadensis/heterophylla — Eastern/western hemlock
    20: (2.9372, 0.5878, 0.253),   # Tsuga mertensiana — Mountain hemlock
    21: (7.5183, 0.4461, 0.815),   # Acer spp — Maple spp
    22: (7.0806, 0.4771, 0.730),   # Alnus rubra — Red alder
    23: (7.0806, 0.4771, 0.730),   # Alnus rhombifolia — White alder
    24: (5.8980, 0.4841, 0.601),   # Betula/Celtis spp — Birch/hackberry spp
    25: (2.4922, 0.8544, 0.140),   # Castanopsis chrysophylla — Giant chinkapin
    26: (4.0910, 0.5907, 0.351),   # Populus tremuloides — Quaking aspen
    27: (7.5183, 0.4461, 0.815),   # Populus spp — Cottonwood/poplar spp
    28: (2.4922, 0.8544, 0.140),   # Quercus spp — Oak spp
    29: (4.5859, 0.4841, 0.468),   # Juniperus spp — Juniper spp
    30: (2.1039, 0.6758, 0.207),   # Larix lyallii — Subalpine larch
    31: (2.1606, 0.6897, 0.255),   # Pinus albicaulis/flexilis — Whitebark/limber pine
    32: (2.1451, 0.7132, 0.248),   # Pinus attenuata — Knobcone pine
    33: (4.5859, 0.4841, 0.468),   # Taxus brevifolia — Pacific yew
    34: (2.4922, 0.8544, 0.140),   # Cornus spp — Dogwood spp
    35: (4.5859, 0.4841, 0.468),   # Crataegus spp — Hawthorn spp
    36: (4.5859, 0.4841, 0.468),   # Prunus spp — Cherry spp
    37: (4.5859, 0.4841, 0.468),   # Salix spp — Willow spp
    39: (4.4215, 0.5329, 0.517),   # Other
}

_CANOPY_EQ_DEFAULT = 39   # fallback equation number


# Load species codes lookup table
SPP_CODES = read_csv(os.path.join(os.path.dirname(__file__), '..', 'supporting_data', 'species_codes_lut.csv'))



# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def calc_bark_thickness(
    spp: np.ndarray,
    dbh: np.ndarray,
) -> np.ndarray:
    """
    Vectorized bark thickness calculation (cm).

    :param spp: np.ndarray of species codes (str or int)
    :param dbh: np.ndarray of diameters (cm)
    :return: np.ndarray of bark thickness values (cm)
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
        raise ValueError(f'No bark thickness coefficient found for species code(s): {missing.tolist()}')

    return bark_thick_per_dbh * dbh


def calc_canopy_cover(
    spp: Union[np.ndarray, List],
    dbh: Union[np.ndarray, List[float]],
    ht: Union[np.ndarray, List[float]],
    tree_code_dict: Optional[Dict[str, int]] = None,
    units: str = 'SI',
) -> float:
    """
    Estimate percent canopy cover for a stand from a tree list.

    Uses species-specific crown width equations (Crookston & Stage 1999,
    RMRS-GTR-24) with a Poisson-process overlap correction.

    **Crown width equations:**

    * Trees > 1.37 m (4.5 ft) tall: ``CW [ft] = A × DBH [in] ^ B``
    * Trees ≤ 1.37 m (4.5 ft) tall: ``CW [ft] = R × DBH [in]``

    **Overlap correction (Poisson process):**

    .. code-block:: text

        CovProp = Cov / 43 560
        PctCov  = 100 × (1 − exp(−CovProp))

    where *Cov* is the accumulated crown area (ft²/ac) summed over all trees
    and 43 560 is the number of square feet in an acre.

    :param spp: Array-like of species codes (str or int).  Each code is
        looked up in *tree_code_dict* to obtain the equation number for
        :data:`_CANOPY_COEFFS`.  If a code is not found, equation 39
        (``Other``) is used.
    :param dbh: Diameter at breast height for each tree.  Units: **m** when
        ``units='SI'``; **in** when ``units='imperial'``.  Trees with
        ``dbh ≤ 0`` or ``NaN`` are excluded.
    :param ht: Height of each tree.  Units: **m** when ``units='SI'``; **ft**
        when ``units='imperial'``.
    :param tree_code_dict: Optional mapping of species code → crown-width
        equation number (see :data:`_CANOPY_COEFFS`).  When ``None``, a
        default mapping (keys = equation numbers as strings) is used.
    :param units: ``'SI'`` (m/m²) or ``'imperial'`` (in/ft).
    :returns: Percent canopy cover (%), adjusted for crown overlap.
    """
    spp_arr = np.asarray(spp)
    dbh_arr = np.asarray(dbh, dtype=float)
    ht_arr  = np.asarray(ht,  dtype=float)

    n = len(spp_arr)
    if len(dbh_arr) != n or len(ht_arr) != n:
        raise ValueError(
            "spp, dbh, and ht must have the same length."
        )

    # Convert to imperial internally (ft for height, in for DBH)
    if units.upper() == 'SI':
        dbh_in = dbh_arr * 39.3701   # m → in
        ht_ft  = ht_arr  *  3.28084  # m → ft
    else:
        dbh_in = dbh_arr.copy()
        ht_ft  = ht_arr.copy()

    # Build equation-number array via species lookup
    eq_arr = np.full(n, _CANOPY_EQ_DEFAULT, dtype=int)
    if tree_code_dict is not None:
        for i, code in enumerate(spp_arr):
            eq_no = tree_code_dict.get(code, _CANOPY_EQ_DEFAULT)
            if eq_no in _CANOPY_COEFFS:
                eq_arr[i] = eq_no
    # else: default eq 39 for all trees

    ht_thresh_ft = 4.5   # 1.37 m in feet

    # Accumulate crown area (ft²) per acre
    # FOFEM assumes each tree record represents trees/ac; crown area = π*(CW/2)²
    # per tree × trees/ac. Without a TPA column we assume 1 tree/ac per record.
    cov_total = 0.0
    for i in range(n):
        d = dbh_in[i]
        h = ht_ft[i]
        # Exclude trees with no valid DBH
        if np.isnan(d) or d <= 0.0:
            continue
        A, B, R = _CANOPY_COEFFS.get(int(eq_arr[i]), _CANOPY_COEFFS[_CANOPY_EQ_DEFAULT])
        if h > ht_thresh_ft:
            cw = A * (d ** B)
        else:
            cw = R * d
        # Crown area per tree (ft²) — circular crown
        cov_total += np.pi * (cw / 2.0) ** 2

    # Overlap correction (Poisson process)
    cov_prop = cov_total / 43560.0
    pct_cov  = 100.0 * (1.0 - np.exp(-cov_prop))
    return float(pct_cov)


def calc_char_ht(flame_length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorized char height calculation.

    :param flame_length: Flame length (m), scalar or np.ndarray
    :return: Char height (m), scalar or np.ndarray
    """
    flame_length = np.asarray(flame_length)
    return flame_length / 1.8


def calc_crown_length_vol_scorched(
    scorch_ht: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray]
) -> tuple:
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

