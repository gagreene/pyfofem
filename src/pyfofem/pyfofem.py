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


# ---------------------------------------------------------------------------
# Moisture Regime Lookup
# ---------------------------------------------------------------------------

_MOISTURE_REGIMES: Dict[str, Dict[str, float]] = {
    'wet':      {'duff': 130.0, '10hr': 22.0, '3plus': 40.0, 'soil': 25.0},
    'moderate': {'duff':  75.0, '10hr': 16.0, '3plus': 30.0, 'soil': 15.0},
    'dry':      {'duff':  40.0, '10hr': 10.0, '3plus': 15.0, 'soil': 10.0},
    'very dry': {'duff':  20.0, '10hr':  6.0, '3plus': 10.0, 'soil':  5.0},
}


# ---------------------------------------------------------------------------
# Carbon Calculations
# ---------------------------------------------------------------------------

# Fuel component groups and their carbon conversion factors (Penman 2003;
# Smith & Heath 2002)
_CARBON_FACTOR_WOODY = 0.50   # down woody, herb, shrub, foliage, branch
_CARBON_FACTOR_DUFF  = 0.37   # duff and litter

_CARBON_WOODY_KEYS = frozenset({
    'dw1', 'dw10', 'dw100',
    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    'herb', 'shrub', 'foliage', 'branch',
})
_CARBON_DUFF_KEYS = frozenset({'duff', 'litter'})


# ---------------------------------------------------------------------------
# Smoke Emissions
# ---------------------------------------------------------------------------

# Path to the bundled emissions-factors CSV (relative to this module).
_EF_CSV_DEFAULT = os.path.join(
    os.path.dirname(__file__), 'supporting_data', 'emissions_factors.csv',
)

# Emission_Factors.csv group indices (1-based row in the CSV data section)
_EF_GROUP_DEFAULT = 3   # Western Forest-Rx (FOFEM default)

# Module-level cache so the CSV is parsed at most once per session.
_ef_df_cache: Optional[DataFrame] = None


def _load_ef_csv(csv_path: Optional[str] = None) -> DataFrame:
    """Load and cache the emission-factors CSV.

    :param csv_path: Explicit path override; falls back to the bundled file.
    :returns: :class:`~pandas.DataFrame` with one row per emission-factor
        group (row 0 = group 1, etc.).
    :raises FileNotFoundError: If the CSV cannot be located.
    """
    global _ef_df_cache
    if csv_path is None:
        csv_path = _EF_CSV_DEFAULT
    if _ef_df_cache is not None and csv_path == _EF_CSV_DEFAULT:
        return _ef_df_cache
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"emissions_factors.csv not found at '{csv_path}'. "
            "Provide the correct path via the ef_csv_path argument."
        )
    # Row 0 = human-readable names (skip), Row 1 = chemical codes
    df = read_csv(csv_path, skiprows=1, header=0)
    if csv_path == _EF_CSV_DEFAULT:
        _ef_df_cache = df
    return df


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
SOIL_HEAT_VARS = ['Lay0', 'Lay2', 'Lay4', 'Lay6', 'Lay60d', 'Lay275d']


def _is_scalar(x) -> bool:
    """Return True if *x* is a Python scalar or 0-d array (i.e. not a sequence/array)."""
    if isinstance(x, np.ndarray):
        return x.ndim == 0
    return np.ndim(x) == 0


def _maybe_scalar(arr, scalar_input: bool):
    """Return ``float(arr[0])`` when *scalar_input* is True, otherwise return *arr* unchanged.

    ``None`` is passed through as-is regardless of *scalar_input*.
    """
    if arr is None:
        return None
    return float(arr[0]) if scalar_input else arr


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
        raise ValueError(f'No bark thickness coefficient found for species code\(s\): {missing.tolist()}')

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


def calc_carbon(
    loadings: Dict[str, Union[float, np.ndarray]],
    units: str = 'SI',
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Convert fuel loadings to carbon loadings using FOFEM conversion factors.

    Two fixed factors are applied (Lutes 2020, p. 79):

    * Down woody, herbaceous, shrub, foliage, branch → Carbon = loading × 0.50
      (Penman et al. 2003)
    * Duff and litter → Carbon = loading × 0.37
      (Smith & Heath 2002)

    :param loadings: Dict of fuel-component loadings. Recognised keys:

        * ``'dw1'``, ``'dw10'``, ``'dw100'`` – timelag woody fuels
        * ``'dwk_3_6'``, ``'dwk_6_9'``, ``'dwk_9_20'``, ``'dwk_20'`` – 1000-hr fuels
        * ``'herb'``, ``'shrub'``, ``'foliage'``, ``'branch'``
        * ``'duff'``, ``'litter'``

        Values may be Python scalars or :class:`numpy.ndarray`.  Units are
        not converted — values are returned in the same units as supplied.
    :param units: ``'SI'`` (kg/m²) or ``'imperial'`` (T/ac).  Currently
        informational only; no unit conversion is performed.
    :returns: Dict with the same keys as *loadings* and carbon loading values
        in the same units.
    :raises ValueError: If *loadings* contains a key that is not recognised.
    """
    result: Dict[str, Union[float, np.ndarray]] = {}
    all_known = _CARBON_WOODY_KEYS | _CARBON_DUFF_KEYS
    for key, val in loadings.items():
        if key in _CARBON_WOODY_KEYS:
            result[key] = val * _CARBON_FACTOR_WOODY
        elif key in _CARBON_DUFF_KEYS:
            result[key] = val * _CARBON_FACTOR_DUFF
        else:
            raise ValueError(
                f"Unrecognised fuel component key '{key}'. "
                f"Known keys: {sorted(all_known)}"
            )
    return result


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


_EF_SMOLDERING_GROUP_DEFAULT = 7  # CWDRSC — coarse woody smoldering
_EF_DUFF_GROUP_DEFAULT = 8        # DuffRSC — duff smoldering


def calc_smoke_emissions(
    flaming_load: Union[float, Dict[str, float]],
    smoldering_load: Union[float, Dict[str, float]],
    mode: str = 'default',
    ef_group: int = _EF_GROUP_DEFAULT,
    ef_smoldering_group: int = _EF_SMOLDERING_GROUP_DEFAULT,
    ef_duff_group: int = _EF_DUFF_GROUP_DEFAULT,
    duff_load: float = 0.0,
    ef_csv_path: Optional[str] = None,
    units: str = 'SI',
) -> Dict[str, float]:
    """
    Compute smoke-emission mass per unit area from fuel consumption totals.

    Both modes read emission factors from ``emissions_factors.csv`` (bundled
    with the package under ``supporting_data/``).

    Two emission-factor modes are available:

    **default** – uses a single emission-factor group (``ef_group``, default 3
    = Western Forest-Rx) applied identically to both flaming and smoldering
    consumption.

    **expanded** – uses **three separate factor groups** matching the C++
    architecture:

    * *STFS* group (default 3) → applied to **flaming** consumption
    * *CWDRSC* group (default 7) → applied to **coarse-wood smoldering**
    * *DuffRSC* group (default 8) → applied to **duff smoldering**

    When ``duff_load`` is provided (> 0) in expanded mode, duff is subtracted
    from *smoldering_load* and processed with the duff-specific factors.

    :param flaming_load: Total flaming fuel consumption.  May be a scalar
        (all components summed) or a dict keyed by fuel component name.
        Units: kg/m² when ``units='SI'``, T/acre when ``units='imperial'``.
    :param smoldering_load: Total smoldering fuel consumption; same format as
        *flaming_load*.  When ``duff_load`` is non-zero in expanded mode,
        this should include the duff portion — it will be separated internally.
    :param mode: ``'default'`` or ``'expanded'``.
    :param ef_group: Emission-factor group (1–8).  In default mode this group
        is applied to both flaming and smoldering.  In expanded mode it is the
        STFS flaming group.  Default 3 (Western Forest-Rx).
    :param ef_smoldering_group: CWDRSC coarse-wood smoldering group (1–8)
        for expanded mode.  Default 7 (Woody RSC).
    :param ef_duff_group: DuffRSC duff smoldering group (1–8) for expanded
        mode.  Default 8 (Duff RSC).
    :param duff_load: Duff consumption included in *smoldering_load*.  In
        expanded mode, this portion is separated and multiplied by the duff-
        specific emission factors.  Default 0 (all smoldering treated with
        the same factors, backward-compatible).
    :param ef_csv_path: Path to ``emissions_factors.csv``.  When ``None``
        (default) the bundled file under ``supporting_data/`` is used.
    :param units: ``'SI'`` (kg/m²) or ``'imperial'`` (T/acre).  Outputs are
        returned in **g/m²** (SI) or **lb/acre** (imperial).
    :returns: Dict with keys matching the FOFEM emission slots:
        ``'PM10F'``, ``'PM10S'``, ``'PM25F'``, ``'PM25S'``,
        ``'CH4F'``, ``'CH4S'``, ``'COF'``, ``'COS'``,
        ``'CO2F'``, ``'CO2S'``, ``'NOXF'``, ``'NOXS'``,
        ``'SO2F'``, ``'SO2S'``.
        In expanded mode, additional duff-only keys are included:
        ``'PM10S_Duff'``, ``'PM25S_Duff'``, ``'CH4S_Duff'``, ``'COS_Duff'``,
        ``'CO2S_Duff'``, ``'NOXS_Duff'``, ``'SO2S_Duff'``.
        Values are in g/m² (SI) or lb/acre (imperial).
    :raises ValueError: If *mode* is not ``'default'`` or ``'expanded'``.
    :raises FileNotFoundError: If the CSV cannot be found.
    """
    # Resolve total loads (scalar sums if dict supplied)
    def _total(load):
        if isinstance(load, dict):
            return float(sum(load.values()))
        return float(load)

    f_kg = _total(flaming_load)    # kg/m² or T/ac
    s_kg = _total(smoldering_load)
    d_kg = float(duff_load)

    # Load emission factors from the bundled CSV
    ef_df = _load_ef_csv(ef_csv_path)

    if mode not in ('default', 'expanded'):
        raise ValueError(
            f"Unknown emissions mode '{mode}'. "
            "Valid options: 'default', 'expanded'."
        )

    if mode == 'default':
        # EF in g/kg → result in g/m² (SI) or g/T * T/ac = g/ac
        #   For SI: emission [g/m²] = load [kg/m²] × EF [g/kg]
        #   For imperial: emission [lb/ac] = load [T/ac] × EF [g/kg]
        #                  × 2000 lb/T / 1000 g/kg = load × EF × 2.0
        if units.upper() == 'SI':
            unit_conv = 1.0  # → g/m²
        else:
            unit_conv = 2.0  # → lb/ac

        lc = {c.strip().lower(): c for c in ef_df.columns}

        def _ef_val(col: str) -> float:
            """Look up emission factor for the default group (case-insensitive)."""
            row = ef_df.iloc[ef_group - 1]
            return float(row[lc[col.lower()]]) if col.lower() in lc else 0.0

        pm25_f = _ef_val('PM2.5')
        pm10_f = _ef_val('PM10')
        ch4_f  = _ef_val('CH4')
        co_f   = _ef_val('CO')
        co2_f  = _ef_val('CO2')
        nox_f  = _ef_val('NOx as NO')
        so2_f  = _ef_val('SO2')

        return {
            'PM10F': pm10_f * f_kg * unit_conv,
            'PM10S': pm10_f * s_kg * unit_conv,
            'PM25F': pm25_f * f_kg * unit_conv,
            'PM25S': pm25_f * s_kg * unit_conv,
            'CH4F':  ch4_f  * f_kg * unit_conv,
            'CH4S':  ch4_f  * s_kg * unit_conv,
            'COF':   co_f   * f_kg * unit_conv,
            'COS':   co_f   * s_kg * unit_conv,
            'CO2F':  co2_f  * f_kg * unit_conv,
            'CO2S':  co2_f  * s_kg * unit_conv,
            'NOXF':  nox_f  * f_kg * unit_conv,
            'NOXS':  nox_f  * s_kg * unit_conv,
            'SO2F':  so2_f  * f_kg * unit_conv,
            'SO2S':  so2_f  * s_kg * unit_conv,
        }

    elif mode == 'expanded':

        def _validate_group(grp, name):
            if grp < 1 or grp > len(ef_df):
                raise ValueError(
                    f"{name} must be between 1 and {len(ef_df)}; got {grp}."
                )

        _validate_group(ef_group, 'ef_group')
        _validate_group(ef_smoldering_group, 'ef_smoldering_group')
        _validate_group(ef_duff_group, 'ef_duff_group')

        lc = {c.strip().lower(): c for c in ef_df.columns}

        def _get_ef_row(row_idx: int, col: str) -> float:
            """Case-insensitive column lookup for a given row; return 0 if missing."""
            r = ef_df.iloc[row_idx]
            return float(r[lc[col.lower()]]) if col.lower() in lc else 0.0

        _EF_COLS = ('CO2', 'CO', 'CH4', 'NOx as NO', 'SO2', 'PM2.5', 'PM10')

        # Three sets of emission factors matching C++ BRN_SetEmis architecture
        row_f = ef_group - 1              # STFS — flaming
        row_s = ef_smoldering_group - 1   # CWDRSC — coarse-wood smoldering
        row_d = ef_duff_group - 1         # DuffRSC — duff smoldering

        # Separate duff from coarse smoldering
        coarse_smo = max(s_kg - d_kg, 0.0)

        if units.upper() == 'SI':
            unit_conv = 1.0
        else:
            unit_conv = 2.0

        def _emit(row_idx, load):
            """Return {species: emission} for a given factor row and load."""
            return {
                'PM10': _get_ef_row(row_idx, 'PM10')      * load * unit_conv,
                'PM25': _get_ef_row(row_idx, 'PM2.5')     * load * unit_conv,
                'CH4':  _get_ef_row(row_idx, 'CH4')        * load * unit_conv,
                'CO':   _get_ef_row(row_idx, 'CO')         * load * unit_conv,
                'CO2':  _get_ef_row(row_idx, 'CO2')        * load * unit_conv,
                'NOX':  _get_ef_row(row_idx, 'NOx as NO')  * load * unit_conv,
                'SO2':  _get_ef_row(row_idx, 'SO2')        * load * unit_conv,
            }

        e_fla = _emit(row_f, f_kg)          # flaming
        e_smo = _emit(row_s, coarse_smo)    # coarse-wood smoldering
        e_duf = _emit(row_d, d_kg)          # duff smoldering

        # Total smoldering = coarse-wood smoldering + duff smoldering
        result = {
            'PM10F': e_fla['PM10'],
            'PM10S': e_smo['PM10'] + e_duf['PM10'],
            'PM25F': e_fla['PM25'],
            'PM25S': e_smo['PM25'] + e_duf['PM25'],
            'CH4F':  e_fla['CH4'],
            'CH4S':  e_smo['CH4']  + e_duf['CH4'],
            'COF':   e_fla['CO'],
            'COS':   e_smo['CO']   + e_duf['CO'],
            'CO2F':  e_fla['CO2'],
            'CO2S':  e_smo['CO2']  + e_duf['CO2'],
            'NOXF':  e_fla['NOX'],
            'NOXS':  e_smo['NOX']  + e_duf['NOX'],
            'SO2F':  e_fla['SO2'],
            'SO2S':  e_smo['SO2']  + e_duf['SO2'],
            # Duff-only emission outputs (§10 from EMISSIONS_COMPARISON)
            'PM10S_Duff': e_duf['PM10'],
            'PM25S_Duff': e_duf['PM25'],
            'CH4S_Duff':  e_duf['CH4'],
            'COS_Duff':   e_duf['CO'],
            'CO2S_Duff':  e_duf['CO2'],
            'NOXS_Duff':  e_duf['NOX'],
            'SO2S_Duff':  e_duf['SO2'],
        }
        return result


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

    return {
        'flc': _maybe_scalar(flc, scalar_input),
        'blc': _maybe_scalar(blc, scalar_input),
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
    dw1000_moist: Optional[Union[float, np.ndarray]] = None,
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
    :param pre_dl110: Pre-fire duff + 1-hr + 10-hr fuel load (kg/m² if
        ``units='SI'``, T/acre if ``units='Imperial'``). Scalar or np.ndarray.
        Required for the Southeast non-Pocosin equation (Eq 16).
    :param pre_l110: Pre-fire litter + 1-hr + 10-hr fuel load (same units as
        ``pre_dl110``). Scalar or np.ndarray. Required for Eq 16.
    :param dw1000_moist: 1000-hr fuel moisture content (%). Required for
        Eq 3 (``duff_moist_cat='nfdth'``), which uses 1000-hr moisture rather
        than duff moisture (matching C++ ``Equ_3_Per`` / ``Equ_7_Red``).
        When ``None``, falls back to ``duff_moist`` for backward compatibility.
    :param pile: ``True`` for pile burning (applies Eq 17, 10% consumed).
        Default ``False``.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``. The
        FOFEM equations operate in T/acre and inches; ``'SI'`` inputs/outputs
        are converted automatically.

    :return: Dict with keys:

        - ``'pdc'`` – percent duff consumed (%) or np.ndarray; ``np.nan``
          where the condition could not be determined.
        - ``'ddc'`` – duff depth consumed (cm if ``units='SI'``, inches if
          ``units='Imperial'``) or np.ndarray; ``None`` if ``d_pre`` is not
          provided.
        - ``'rdd'`` – residual duff depth (cm if ``units='SI'``, inches if
          ``units='Imperial'``) or np.ndarray; ``None`` if
          ``duff_moist_cat != 'edm'`` or ``d_pre`` is not provided.
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

    # Fix A: Eq 3 / Eq 7 use 1000-hr moisture, not duff moisture (C++ Equ_3_Per /
    # Equ_7_Red both use f_MoistDW1000). Fall back to duff_moist if not supplied.
    if dw1000_moist is not None:
        dw1000_moist_arr = np.atleast_1d(np.asarray(dw1000_moist, dtype=float))
    else:
        dw1000_moist_arr = duff_moist  # backward-compatible fallback

    if units == 'SI':
        pre_dl = pre_dl * 4.4609                            # Mg/ha → T/acre
        if d_pre is not None:
            d_pre = d_pre / 2.54                            # cm → in
        if isinstance(rm_depth, list):
            rm_depth = [x / 2.54 for x in rm_depth]        # cm → in
        elif rm_depth is not None:
            rm_depth = rm_depth / 2.54
        if pre_dl110 is not None:
            pre_dl110 = pre_dl110 * 4.4609                  # kg/m² → T/acre
        if pre_l110 is not None:
            pre_l110 = pre_l110 * 4.4609

    # --- pdc: PERCENT DUFF CONSUMED ---
    pdc = np.full_like(duff_moist, np.nan)

    # Fix D: C++ DUF_Mngr forces 100 % consumed when duff_moist ≤ 10
    # (lowest allowable moisture — all duff burns at this extreme)
    low_moist_mask = duff_moist <= 10.0

    if (duff_moist_cat == 'ldm') and (reg in ['InteriorWest', 'PacificWest']):
        if cvr_grp not in ['Ponderosa pine', 'PN', 'Ponderosa', 'Pocosin', 'PC']:
            # Equation 1
            pdc = np.where(duff_moist <= 160, 97.1 - 0.519 * duff_moist, 13.6)
        else:
            # Equation 4
            pdc = 89.9 - 0.55 * duff_moist
    elif (duff_moist_cat == 'nfdth') and (reg in ['InteriorWest', 'PacificWest', 'NorthEast']):
        # Equation 3 — Fix A: uses 1000-hr moisture (f_MoistDW1000 in C++)
        pdc = 114.7 - 4.20 * dw1000_moist_arr
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
        # Equation 17 – pile burning: 10% consumed (C++ Equ_17_Per)
        pdc = np.full_like(duff_moist, 10.0)
    elif cvr_grp in ['Chaparral', 'Shrub-Chaparral', 'SGC', 'ShrubGroupChaparral']:
        # Equation 19 – Chaparral: 100% consumed
        pdc = np.full_like(duff_moist, 100.0)
    else:
        # Equation 2 – default
        pdc = 83.7 - 0.426 * duff_moist

    # Fix D: C++ DUF_Mngr overrides pdc to 100% when duff_moist ≤ 10
    pdc = np.where(low_moist_mask, 100.0, pdc)

    # --- ddc: DUFF DEPTH CONSUMED ---
    ddc = None
    if d_pre is not None:
        if (duff_moist_cat == 'ldm') and (reg in ['InteriorWest', 'PacificWest']):
            # Equation 5
            ddc = 1.028 - 0.0089 * duff_moist + 0.417 * d_pre
        elif (duff_moist_cat == 'nfdth') and (reg in ['InteriorWest', 'PacificWest', 'NorthEast']):
            # Equation 7 — Fix A: uses 1000-hr moisture (f_MoistDW1000 in C++)
            ddc = 1.773 - 0.1051 * dw1000_moist_arr + 0.399 * d_pre
        else:
            # Equation 6 – default
            ddc = 0.8811 - 0.0096 * duff_moist + 0.439 * d_pre

    # --- rdd: RESIDUAL DUFF DEPTH ---
    rdd = None
    if (duff_moist_cat == 'edm') and (d_pre is not None):
        pine = 1 if cvr_grp in ['Red Jack Pine', 'Red, Jack Pine', 'RedJacPin', 'RJP'] else 0
        # Equation 15
        rdd = -0.791 + 0.004 * duff_moist + 0.8 * d_pre + 0.56 * pine

    # Clamp pdc to valid range [0, 100] (matching C++ DUF_Mngr Note-1)
    pdc = np.clip(pdc, 0.0, 100.0)

    # Convert depth outputs back to cm when inputs were SI
    if units == 'SI':
        if ddc is not None:
            ddc = ddc * 2.54  # in → cm
        if rdd is not None:
            rdd = rdd * 2.54  # in → cm

    return {
        'pdc': _maybe_scalar(pdc, scalar_input),
        'ddc': _maybe_scalar(ddc, scalar_input),
        'rdd': _maybe_scalar(rdd, scalar_input),
    }


def consm_herb(
    reg: str,
    cvr_grp: str,
    pre_ll: Union[float, np.ndarray],
    pre_hl: Union[float, np.ndarray],
    season: Optional[str] = None,
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
          **only in Spring** (matching C++ ``Calc_Herb``); all other seasons
          fall back to Eq 22 (100% consumed).
        - ``'Flatwood'`` / ``'Pine Flatwoods'`` / ``'PFL'`` / ``'PinFltwd'``
          → Eq 223
        - All others → Eq 22 (100% consumed)

    :param pre_ll: Pre-fire litter fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_hl: Pre-fire herbaceous fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param season: Burn season (``'Spring'``, ``'Summer'``, ``'Fall'``,
        ``'Winter'``). Only relevant for GrassGroup: Eq 221 (10%) applies in
        Spring; all other seasons use Eq 22 (100%). Optional; defaults to
        non-Spring behaviour when not provided.
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

    # Normalise season for comparison
    season_cap = season.capitalize() if season is not None else ''

    if reg in ['SouthEast']:
        # Equation 222 — C++: -0.059 + 0.004*litter + 0.917*herb
        hlc = -0.059 + (0.004 * pre_ll) + (0.917 * pre_hl)
    elif cvr_grp in ['Grass', 'GG', 'GrassGroup'] and season_cap == 'Spring':
        # Equation 221 — Fix B: C++ Calc_Herb applies 10% only in Spring
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
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``. The
        FOFEM equations operate in T/acre; ``'SI'`` inputs are converted to
        T/acre internally and the result is converted back to kg/m² before
        returning. ``'Imperial'`` inputs (T/acre) are passed directly.

    :return: Litter load consumed (kg/m² for ``'SI'``, T/acre for
        ``'Imperial'``). Scalar ``float`` when all numeric inputs are scalars,
        otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(l_moist)

    pre_ll = np.atleast_1d(np.asarray(pre_ll, dtype=float))
    l_moist = np.atleast_1d(np.asarray(l_moist, dtype=float))

    if units == 'SI':
        pre_ll = pre_ll * 4.4609  # kg/m² → T/acre

    if cvr_grp in ['Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd']:
        # FOFEM litter consumption equation 997
        llc = np.power(0.2871 + (0.9140 * np.sqrt(pre_ll)) - (0.0101 * l_moist), 2)
    elif reg == 'SouthEast':
        # FOFEM litter consumption equation 998
        llc = pre_ll * 0.8
    else:
        # FOFEM litter consumption equation 999 – calculated with Burnup (generally 100%)
        llc = pre_ll.copy()

    if units == 'SI':
        llc = llc / 4.4609  # T/acre → kg/m²

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

    if season is not None:
        season = season.capitalize()

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

    if reg in ['SouthEast']:
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


def get_moisture_regime(regime: str) -> Dict[str, float]:
    """
    Return default fuel moisture values (%) for a named FOFEM moisture regime.

    Four western moisture regimes are defined (Lutes 2020, p. 79):

    +----------+---------+---------+----------+---------+
    | Regime   | Duff    | 10-hr   | 3+ in.   | Soil    |
    +==========+=========+=========+==========+=========+
    | Wet      | 130 %   | 22 %    | 40 %     | 25 %    |
    | Moderate |  75 %   | 16 %    | 30 %     | 15 %    |
    | Dry      |  40 %   | 10 %    | 15 %     | 10 %    |
    | Very Dry |  20 %   |  6 %    | 10 %     |  5 %    |
    +----------+---------+---------+----------+---------+

    The 1-hr moisture is 2 % lower than 10-hr; the 100-hr moisture is 2 %
    higher than 10-hr.

    :param regime: One of ``'Wet'``, ``'Moderate'``, ``'Dry'``, or
        ``'Very Dry'`` (case-insensitive).
    :returns: Dict with keys ``'duff'``, ``'10hr'``, ``'3plus'``, ``'soil'``,
        ``'1hr'``, and ``'100hr'`` — all values in percent.
    :raises ValueError: If *regime* is not one of the four recognised names.
    """
    key = regime.strip().lower()
    if key not in _MOISTURE_REGIMES:
        valid = [k.title() for k in _MOISTURE_REGIMES]
        raise ValueError(
            f"Unrecognised moisture regime '{regime}'. "
            f"Valid options: {valid}"
        )
    base = _MOISTURE_REGIMES[key].copy()
    base['1hr']   = base['10hr'] - 2.0
    base['100hr'] = base['10hr'] + 2.0
    return base


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
        Pm[mask_cade] = 1 / (1 + np.exp(-(-4.2466 + (np.power(cls[mask_cade], 3) * 0.000007172))))
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
    densities: Optional[Dict[str, float]] = None,
    fint_switch: float = 15.0,
    validate: bool = True,
) -> Tuple[List[BurnResult], List[BurnSummaryRow], List[str]]:
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
    :param densities: Optional dict mapping size-class key to per-particle
        oven-dry density (kg/m³).  Any key not present falls back to
        *density*.  Useful for rotten wood (lower density, e.g. 300 kg/m³).
    :param fint_switch: Flaming / smoldering intensity threshold (kW/m²).
        Default 15.
    :param validate: If True (default), run range checks on all inputs
        before simulation.
    :return: ``(results, summary, class_order)`` where *results* is a list of
        :class:`~pyfofem.components.burnup.BurnResult` (one per completed
        time step), *summary* is a list of
        :class:`~pyfofem.components.burnup.BurnSummaryRow` (one per fuel
        component), and *class_order* is the list of size-class keys
        in the order they were passed to the engine (corresponding to the
        sorted-component indices in *summary* and the per-component arrays
        in *results*).
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
    class_order: List[str] = []
    for key in _class_order:
        loading = fuel_loadings.get(key, 0.0)
        if loading <= 0.0:
            continue
        moisture = fuel_moistures.get(key, default_moisture)
        sigma = _sigma_map[key]
        d = densities.get(key, density) if densities else density
        particles.append(FuelParticle(
            wdry=loading,
            htval=heat_content,
            fmois=moisture,
            dendry=d,
            sigma=sigma,
            cheat=heat_capacity,
            condry=conductivity,
            tpig=ignition_temp,
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
        fint_switch=fint_switch,
        validate=validate,
    )

    return results, summary, class_order


# ---------------------------------------------------------------------------
# Emissions / Consumption Facade
# ---------------------------------------------------------------------------

# Default densities for sound and rotten wood (kg/m³)
_DENSITY_SOUND: float = 513.0
_DENSITY_ROTTEN: float = 224.0   # C++ e_Rot_dendry (Albini & Reinhardt)

# C++ physical constants for burnup fuel particles
# tpig/tchar are in °C — burnup.py adds +273 internally
_SOUND_TPIG: float = 327.0      # C++ e_Snd_tpig  (ignition temp, °C)
_ROTTEN_TPIG: float = 302.0     # C++ e_Rot_tpig  (ignition temp, °C)
_TCHAR: float = 377.0           # C++ e_tchar      (char temp, °C)
_HTVAL: float = 1.86e7           # C++ e_htval=18600 kJ/kg × 1000 = J/kg

# Unit conversion factor: T/acre → kg/m²
_TPAC_TO_KGPM2: float = 1.0 / 4.4609
_KGPM2_TO_TPAC: float = 4.4609
# inches → cm
_IN_TO_CM: float = 2.54


def _extract_burnup_consumption(
    results: List[BurnResult],
    summary: List[BurnSummaryRow],
    class_order: List[str],
    dt: float,
) -> Dict[str, Dict[str, float]]:
    """Extract per-fuel-class consumption and flaming/smoldering partition from burnup.

    :param results: Time-step outputs from the burnup simulation.
    :param summary: Per-component summary rows.
    :param class_order: Fuel-class keys in the order passed to ``run_burnup``
        (matches the sorted component indices in *summary*).
    :param dt: Integration time step (s) used by the simulation.
    :return: Dict keyed by fuel-class name with sub-dict
        ``{'consumed', 'flaming', 'smoldering', 'frac_remaining'}``.
        Mass values are in kg/m².
    """
    n_comp = len(class_order)
    out: Dict[str, Dict[str, float]] = {}

    # Accumulate per-component flaming / smoldering masses across time steps
    comp_fla = [0.0] * n_comp
    comp_smo = [0.0] * n_comp
    for r in results:
        if r.comp_flaming is not None:
            for i in range(n_comp):
                # comp_flaming stores the mass-loss rate (kg/m²/s) that was
                # partitioned as flaming during this step.  Multiply by dt
                # to get mass.
                comp_fla[i] += r.comp_flaming[i] * dt
        if r.comp_smoldering is not None:
            for i in range(n_comp):
                comp_smo[i] += r.comp_smoldering[i] * dt

    for i, key in enumerate(class_order):
        row = summary[i]
        consumed = row.wdry * (1.0 - row.frac_remaining)
        # Use accumulated flaming/smoldering if available; otherwise
        # fall back to total consumed assigned to smoldering (conservative)
        total_partitioned = comp_fla[i] + comp_smo[i]
        if total_partitioned > 1e-12:
            # Scale partitioned totals to match the consumed mass (avoids drift)
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


def _burnup_durations(
    results: List[BurnResult],
    ff_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Derive flaming and smoldering durations from burnup time-series.

    :param results: List of :class:`BurnResult` from the simulation.
    :param ff_threshold: Flaming-fraction threshold; steps with ``ff ≥``
        this value are classified as flaming.
    :return: ``(fla_dur, smo_dur)`` in **seconds** (matching C++ ``d_CO``).
    """
    if not results:
        return float('nan'), float('nan')

    fla_start = None
    fla_end = None
    smo_end = None

    for r in results:
        if r.ff >= ff_threshold:
            if fla_start is None:
                fla_start = r.time
            fla_end = r.time
        if r.wdf < 1.0:
            smo_end = r.time

    t0 = results[0].time

    if fla_start is not None and fla_end is not None:
        fla_dur = fla_end - fla_start
    else:
        fla_dur = 0.0

    if smo_end is not None:
        smo_start = fla_end if fla_end is not None else t0
        smo_dur = max(smo_end - smo_start, 0.0)
    else:
        smo_dur = 0.0

    return fla_dur, smo_dur


def run_fofem_emissions(
    litter: Union[float, np.ndarray],
    duff: Union[float, np.ndarray],
    duff_moist: Union[float, np.ndarray],
    duff_depth: Union[float, np.ndarray],
    herb: Union[float, np.ndarray],
    shrub: Union[float, np.ndarray],
    crown_foliage: Union[float, np.ndarray],
    crown_branch: Union[float, np.ndarray],
    pct_crown_burned: Union[float, np.ndarray],
    l_moist: Union[float, np.ndarray],
    dw10_moist: Union[float, np.ndarray],
    dw1000_moist: Union[float, np.ndarray],
    region: str,
    cvr_grp: str = '',
    season: str = 'Summer',
    fuel_category: str = 'Natural',
    dw1: Union[float, np.ndarray] = 0.0,
    dw10: Union[float, np.ndarray] = 0.0,
    dw100: Union[float, np.ndarray] = 0.0,
    dw1000s: Union[float, np.ndarray] = 0.0,
    dw1000r: Union[float, np.ndarray] = 0.0,
    dw3_6s: Union[float, np.ndarray] = 0.0,
    dw6_9s: Union[float, np.ndarray] = 0.0,
    dw9_20s: Union[float, np.ndarray] = 0.0,
    dw20s: Union[float, np.ndarray] = 0.0,
    dw3_6r: Union[float, np.ndarray] = 0.0,
    dw6_9r: Union[float, np.ndarray] = 0.0,
    dw9_20r: Union[float, np.ndarray] = 0.0,
    dw20r: Union[float, np.ndarray] = 0.0,
    hfi: Optional[Union[float, np.ndarray]] = None,
    flame_res_time: Optional[Union[float, np.ndarray]] = None,
    fuel_bed_depth: Optional[float] = None,
    ambient_temp: Optional[float] = None,
    windspeed: Optional[float] = None,
    use_burnup: bool = True,
    burnup_kwargs: Optional[dict] = None,
    em_mode: str = 'default',
    ef_group: int = _EF_GROUP_DEFAULT,
    ef_csv_path: Optional[str] = None,
    units: str = 'Imperial',
) -> dict:
    """
    Facade driver for FOFEM fuel consumption, carbon, and smoke emissions.

    Orchestrates calls to the individual ``consm_*`` functions and (when
    ``use_burnup=True``) the :func:`run_burnup` post-frontal combustion
    model.  Burnup physics-based consumption replaces the simplified
    percentage defaults for woody fuels and duff, partitioning mass loss
    into flaming and smoldering phases.  Results are then passed to
    :func:`calc_smoke_emissions` for pollutant estimates.

    When burnup cannot run (missing fire-environment parameters, or the
    simulation raises an error), the function falls back to the original
    simplified percentage defaults and prints a warning.

    **Fuel consumption sub-models:**

    - Litter → :func:`consm_litter` (Eqs 997–999)
    - Duff → :func:`consm_duff` (Eqs 1–20, ``duff_moist_cat='edm'``);
      duff consumption via burnup when ``use_burnup=True``
    - Herbaceous → :func:`consm_herb` (Eqs 22, 221–223)
    - Shrub → :func:`consm_shrub` (Eqs 23, 231–236)
    - Crown foliage/branch → :func:`consm_canopy`
    - Mineral soil exposure → :func:`consm_mineral_soil`
    - Woody fuels (1-hr through 20"+ diameter) → :func:`run_burnup`
      (Albini & Reinhardt post-frontal combustion model) when
      ``use_burnup=True``; otherwise simplified percentage defaults.

    **Simplified defaults (fallback when Burnup is not used):**

    - 1-hr, 10-hr woody: 100 % consumed (flaming)
    - 100-hr woody: ``max(50 %, 95 % − 0.8 %/% moisture above 10 %)``
    - 1000-hr sound: ``max(2 %, 15 % − 0.1 %/% moisture)``
    - 1000-hr rotten: ``max(5 %, 45 % − 0.3 %/% moisture)``

    **1000-hr size-class inputs:**

    The function accepts per-size-class 1000-hr loads (``dw3_6s``,
    ``dw6_9s``, ``dw9_20s``, ``dw20s`` for sound; ``dw3_6r`` … ``dw20r``
    for rotten).  If these are all zero but ``dw1000s`` / ``dw1000r`` are
    non-zero, the aggregate loads are used as a single size class (mapped
    to the 3–6" SAV) for backward compatibility.

    :param litter: Pre-fire litter load (T/ac if ``units='Imperial'``, kg/m²
        if ``units='SI'``).
    :param duff: Pre-fire duff load (T/ac or kg/m²).
    :param duff_moist: Duff gravimetric moisture content (%).
    :param duff_depth: Pre-fire duff depth (in. if ``units='Imperial'``,
        cm if ``units='SI'``).
    :param herb: Pre-fire herbaceous load (same units as *litter*).
    :param shrub: Pre-fire shrub load (same units as *litter*).
    :param crown_foliage: Pre-fire crown foliage load (same units as *litter*).
    :param crown_branch: Pre-fire crown branch load (same units as *litter*).
    :param pct_crown_burned: Proportion of stand area affected by crown fire
        (%).
    :param l_moist: Litter (≈ 1-hr) moisture content (%).
    :param dw10_moist: 10-hr woody fuel moisture content (%).
    :param dw1000_moist: 1000-hr woody fuel moisture content (%).
    :param region: FOFEM region.  One of ``'InteriorWest'``,
        ``'PacificWest'``, ``'NorthEast'``, ``'SouthEast'``.
    :param cvr_grp: FOFEM cover group name (e.g. ``'ShrubGroup'``,
        ``'GrassGroup'``, ``'Ponderosa pine'``).  Empty string uses default
        equations.
    :param season: Burn season: ``'Spring'``, ``'Summer'``, ``'Fall'``, or
        ``'Winter'``.  Default ``'Summer'``.
    :param fuel_category: ``'Natural'`` or ``'Slash'``.  Used for the mineral
        soil exposure equation.  Default ``'Natural'``.
    :param dw1: Pre-fire 1-hr down woody load (same units as *litter*).
    :param dw10: Pre-fire 10-hr down woody load (same units as *litter*).
    :param dw100: Pre-fire 100-hr down woody load (same units as *litter*).
    :param dw1000s: Pre-fire 1000-hr sound down woody load (aggregate, same
        units).  Used as fallback when per-size-class sound loads are all 0.
    :param dw1000r: Pre-fire 1000-hr rotten down woody load (aggregate, same
        units).  Used as fallback when per-size-class rotten loads are all 0.
    :param dw3_6s: 1000-hr sound, 3–6 in. diameter (same units as *litter*).
    :param dw6_9s: 1000-hr sound, 6–9 in. diameter.
    :param dw9_20s: 1000-hr sound, 9–20 in. diameter.
    :param dw20s: 1000-hr sound, ≥ 20 in. diameter.
    :param dw3_6r: 1000-hr rotten, 3–6 in. diameter.
    :param dw6_9r: 1000-hr rotten, 6–9 in. diameter.
    :param dw9_20r: 1000-hr rotten, 9–20 in. diameter.
    :param dw20r: 1000-hr rotten, ≥ 20 in. diameter.
    :param hfi: Head fire intensity (kW/m).  Used to compute burnup
        igniting intensity when *flame_res_time* is not supplied, and/or
        to set ``FlaDur`` in fallback mode.
    :param flame_res_time: Flame residence time (sec), used for burnup (``ig_time``).
    :param fuel_bed_depth: Fuel bed depth (m).  Required for burnup;
        defaults to 0.3 m when not provided.
    :param ambient_temp: Ambient air temperature (°C).  Required for burnup;
        defaults to 27 °C when not provided.
    :param windspeed: Windspeed at fuelbed top (m/s).  Required for burnup;
        defaults to 0 m/s when not provided.
    :param use_burnup: If True (default), run the burnup model for woody
        and duff consumption.  Set to False to use simplified defaults only.
    :param burnup_kwargs: Optional dict of advanced parameters forwarded to
        :func:`run_burnup` (e.g. ``r0``, ``dr``, ``timestep``, ``max_times``,
        ``fint_switch``).
    :param em_mode: Emission factor mode: ``'default'`` or ``'expanded'``.
        Both read from ``emissions_factors.csv``.
    :param ef_group: Emission factor group (1–8; default 3 = Western
        Forest-Rx).
    :param ef_csv_path: Path to ``emissions_factors.csv`` when overriding the
        bundled file.
    :param units: Unit system.  ``'Imperial'`` (T/ac, in) or ``'SI'``
        (kg/m², cm).  Input and output values use the same units.
    :returns: Dict with all :data:`CONSUMPTION_VARS` keys.  Load values are
        in the same units as the inputs.  Emission values are in g/m² (SI)
        or lb/ac (Imperial).  ``Lay*`` (soil temperature) keys are always
        ``NaN`` — use :func:`~pyfofem.components.soil_heating.soil_heat_campbell`
        separately to populate those.
    """
    import warnings

    season = season.capitalize()
    is_imperial = units.strip().lower() in ('imperial', 'english')

    def _s(x) -> float:
        """Collapse any numpy scalar / 0-d / 1-d array to a Python float."""
        if x is None:
            return float('nan')
        a = np.asarray(x).ravel()
        return float(a[0]) if a.size > 0 else float('nan')

    # ── 1. Litter ─────────────────────────────────────────────────────────
    lit_pre = _s(litter)
    lit_con = _s(consm_litter(litter, l_moist, cvr_grp=cvr_grp, reg=region, units=units))
    lit_pos = lit_pre - lit_con

    if cvr_grp in ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd'):
        lit_eq = 997
    elif region == 'SouthEast':
        lit_eq = 998
    else:
        lit_eq = 999

    # ── 2. Resolve 1000-hr size-class loads ───────────────────────────────
    # Per-size-class sound
    s3_6  = _s(dw3_6s)
    s6_9  = _s(dw6_9s)
    s9_20 = _s(dw9_20s)
    s20   = _s(dw20s)
    # Per-size-class rotten
    r3_6  = _s(dw3_6r)
    r6_9  = _s(dw6_9r)
    r9_20 = _s(dw9_20r)
    r20   = _s(dw20r)

    # Aggregate fallback: if all per-size-class loads are zero but aggregates
    # are supplied, put the aggregate into the 3-6" class for backward compat.
    dw1ks_total_cls = s3_6 + s6_9 + s9_20 + s20
    dw1kr_total_cls = r3_6 + r6_9 + r9_20 + r20
    if dw1ks_total_cls == 0.0 and _s(dw1000s) > 0.0:
        s3_6 = _s(dw1000s)
        dw1ks_total_cls = s3_6
    if dw1kr_total_cls == 0.0 and _s(dw1000r) > 0.0:
        r3_6 = _s(dw1000r)
        dw1kr_total_cls = r3_6

    dw1ks_pre = s3_6 + s6_9 + s9_20 + s20
    dw1kr_pre = r3_6 + r6_9 + r9_20 + r20

    # ── 3. Pre-fire scalars for other woody classes ───────────────────────
    dw10m  = _s(dw10_moist)
    dw1km  = _s(dw1000_moist)
    dw1_pre  = _s(dw1)
    dw10_pre = _s(dw10)
    dw100_pre = _s(dw100)

    # ── 4. Attempt burnup for woody + duff consumption ────────────────────
    burnup_ran = False
    duf_loading_si = 0.0
    from_si = 1.0

    if use_burnup:
        # Determine fire-environment parameters
        frt_s = _s(flame_res_time) if flame_res_time is not None else None
        hfi_val = _s(hfi) if hfi is not None else None
        if frt_s is None and hfi_val is not None and hfi_val > 0:
            # Estimate residence time from fire intensity (very rough)
            frt_s = 60.0
        fb_depth = fuel_bed_depth if fuel_bed_depth is not None else 0.3
        amb_temp = ambient_temp if ambient_temp is not None else 21.0
        ws = windspeed if windspeed is not None else 0.0
        intensity_kw = hfi_val if hfi_val is not None else 50.0

        # Convert loads to kg/m² for burnup (burnup works in SI)
        if is_imperial:
            to_si = _TPAC_TO_KGPM2
        else:
            to_si = 1.0

        # Build fuel loadings dict for burnup — sound classes
        fuel_loadings: Dict[str, float] = {}
        if lit_pre * to_si > 0:
            fuel_loadings['litter'] = lit_pre * to_si
        if dw1_pre * to_si > 0:
            fuel_loadings['dw1'] = dw1_pre * to_si
        if dw10_pre * to_si > 0:
            fuel_loadings['dw10'] = dw10_pre * to_si
        if dw100_pre * to_si > 0:
            fuel_loadings['dw100'] = dw100_pre * to_si
        if s3_6 * to_si > 0:
            fuel_loadings['dwk_3_6'] = s3_6 * to_si
        if s6_9 * to_si > 0:
            fuel_loadings['dwk_6_9'] = s6_9 * to_si
        if s9_20 * to_si > 0:
            fuel_loadings['dwk_9_20'] = s9_20 * to_si
        if s20 * to_si > 0:
            fuel_loadings['dwk_20'] = s20 * to_si

        # Gotcha #2: C++ BCM_SetInputs injects a tiny load into 1-hr wood
        # if all down-wood loads are zero, so burnup can still run with
        # litter/duff only.  The tiny amount won't show up in results.
        _dw_keys = ('dw1', 'dw10', 'dw100', 'dwk_3_6', 'dwk_6_9',
                     'dwk_9_20', 'dwk_20')
        if not any(fuel_loadings.get(k, 0.0) > 0.0 for k in _dw_keys):
            fuel_loadings['dw1'] = 1.1e-6  # just above _SMALLX validation threshold

        # Rotten classes: use same SAV keys but suffix with '_r' internally
        # to differentiate from sound; will be treated as additional
        # FuelParticle entries with lower density.
        rotten_keys: Dict[str, str] = {}  # internal_key → sav key
        if r3_6 * to_si > 0:
            fuel_loadings['dwk_3_6_r'] = r3_6 * to_si
            rotten_keys['dwk_3_6_r'] = 'dwk_3_6'
        if r6_9 * to_si > 0:
            fuel_loadings['dwk_6_9_r'] = r6_9 * to_si
            rotten_keys['dwk_6_9_r'] = 'dwk_6_9'
        if r9_20 * to_si > 0:
            fuel_loadings['dwk_9_20_r'] = r9_20 * to_si
            rotten_keys['dwk_9_20_r'] = 'dwk_9_20'
        if r20 * to_si > 0:
            fuel_loadings['dwk_20_r'] = r20 * to_si
            rotten_keys['dwk_20_r'] = 'dwk_20'

        # Build fuel moistures dict (percent → fraction)
        # Apply C++ BCM_DW10M_Adj moisture adjustments (Gotcha #1):
        #   1-hr:   DW10_moisture − 0.02
        #   10-hr:  DW10_moisture (as-is)
        #   100-hr: DW10_moisture + 0.02
        # Rotten 1000-hr: moisture × 2.5, capped at burnup upper limit (3.0)
        _DW1HR_ADJ = 0.02
        _DW100HR_ADJ = 0.02
        _DW1000HR_ADJ_ROT = 2.5
        _BURNUP_MOIST_UPPER = 3.0   # e_fms2 in C++

        l_moist_frac = _s(l_moist) / 100.0
        dw10_moist_frac = dw10m / 100.0
        dw1000_moist_frac = dw1km / 100.0
        dw1000_rot_moist_frac = min(
            dw1000_moist_frac * _DW1000HR_ADJ_ROT,
            _BURNUP_MOIST_UPPER,
        )
        fuel_moistures: Dict[str, float] = {}
        for k in fuel_loadings:
            if k == 'litter':
                fuel_moistures[k] = max(dw10_moist_frac - _DW1HR_ADJ, 0.02)
            elif k == 'dw1':
                fuel_moistures[k] = max(dw10_moist_frac - _DW1HR_ADJ, 0.02)
            elif k in ('dw10',):
                fuel_moistures[k] = max(dw10_moist_frac, 0.02)
            elif k in ('dw100',):
                fuel_moistures[k] = max(dw10_moist_frac + _DW100HR_ADJ, 0.02)
            elif k.endswith('_r'):
                # Rotten 1000-hr classes: moisture × 2.5, capped
                fuel_moistures[k] = max(dw1000_rot_moist_frac, 0.02)
            else:
                # Sound 1000-hr classes
                fuel_moistures[k] = max(dw1000_moist_frac, 0.02)

        # Build per-particle density dict (rotten gets lower density)
        density_map: Dict[str, float] = {}
        for k in rotten_keys:
            density_map[k] = _DENSITY_ROTTEN

        # Duff for burnup (kg/m²)
        duf_loading_si = _s(duff) * to_si
        duf_moist_frac = _s(duff_moist) / 100.0

        # Merge burnup_kwargs
        bkw = {
            'r0': 1.83, 'dr': 0.4, 'timestep': 15.0, 'max_times': 3000,
            'fint_switch': 15.0, 'validate': True,
        }
        if burnup_kwargs:
            bkw.update(burnup_kwargs)

        # We need to add rotten SAV entries to the sigma map inside run_burnup.
        # Rather than modify run_burnup's internal sigma map, we'll build
        # particles directly when rotten classes are present.
        if rotten_keys and fuel_loadings:
            try:
                # Build particles manually to support both sound + rotten
                _sav_defaults = {
                    'litter': 8200.0, 'dw1': 1480.0, 'dw10': 394.0,
                    'dw100': 105.0, 'dwk_3_6': 39.4, 'dwk_6_9': 21.9,
                    'dwk_9_20': 12.7, 'dwk_20': 5.91,
                }
                _class_order_all = [
                    'litter', 'dw1', 'dw10', 'dw100',
                    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
                    'dwk_3_6_r', 'dwk_6_9_r', 'dwk_9_20_r', 'dwk_20_r',
                ]
                particles_list: List[FuelParticle] = []
                burnup_class_order: List[str] = []
                for key in _class_order_all:
                    loading = fuel_loadings.get(key, 0.0)
                    if loading <= 0.0:
                        continue
                    moisture = fuel_moistures.get(key, 0.10)
                    # Determine SAV: rotten keys map back to their sound SAV
                    if key in rotten_keys:
                        sav = _sav_defaults[rotten_keys[key]]
                    else:
                        sav = _sav_defaults.get(key, 39.4)
                    d = density_map.get(key, _DENSITY_SOUND)
                    is_rotten = key in rotten_keys
                    particles_list.append(FuelParticle(
                        wdry=loading,
                        htval=_HTVAL,
                        fmois=moisture,
                        dendry=d,
                        sigma=sav,
                        cheat=2750.0,
                        condry=0.133,
                        tpig=_ROTTEN_TPIG if is_rotten else _SOUND_TPIG,
                        tchar=_TCHAR,
                        ash=0.05,
                    ))
                    burnup_class_order.append(key)

                if particles_list:
                    burnup_results, burnup_summary = _burnup(
                        particles=particles_list,
                        fi=intensity_kw,
                        ti=frt_s if frt_s else 60.0,
                        u=ws,
                        d=fb_depth,
                        tamb=amb_temp,
                        r0=bkw['r0'],
                        dr=bkw['dr'],
                        dt=bkw['timestep'],
                        ntimes=bkw['max_times'],
                        wdf=duf_loading_si,
                        dfm=max(duf_moist_frac, 0.02) if duf_loading_si > 0 else 2.0,
                        fint_switch=bkw['fint_switch'],
                        validate=bkw['validate'],
                    )
                    burnup_ran = True
                    burnup_dt = bkw['timestep']
            except Exception as e:
                warnings.warn(
                    f'Burnup simulation failed ({e}); falling back to '
                    f'simplified consumption defaults.',
                    stacklevel=2,
                )
                burnup_ran = False

        elif fuel_loadings:
            try:
                burnup_results, burnup_summary, burnup_class_order = run_burnup(
                    fuel_loadings=fuel_loadings,
                    fuel_moistures=fuel_moistures,
                    intensity=intensity_kw,
                    ig_time=frt_s if frt_s else 60.0,
                    windspeed=ws,
                    depth=fb_depth,
                    ambient_temp=amb_temp,
                    duff_loading=duf_loading_si,
                    duff_moisture=max(duf_moist_frac, 0.02) if duf_loading_si > 0 else 2.0,
                    densities=density_map if density_map else None,
                    **bkw,
                )
                burnup_ran = True
                burnup_dt = bkw['timestep']
            except Exception as e:
                warnings.warn(
                    f'Burnup simulation failed ({e}); falling back to '
                    f'simplified consumption defaults.',
                    stacklevel=2,
                )
                burnup_ran = False

    # ── 5. Extract woody consumption from burnup or apply defaults ────────
    if burnup_ran:
        # Extract per-class consumption
        bcon = _extract_burnup_consumption(
            burnup_results, burnup_summary, burnup_class_order, burnup_dt,
        )
        from_si = 1.0 / to_si  # convert kg/m² back to user units

        # Litter consumption from burnup (overrides consm_litter if available)
        if 'litter' in bcon:
            lit_con = bcon['litter']['consumed'] * from_si
            lit_pos = lit_pre - lit_con

        # 1-hr
        if 'dw1' in bcon:
            dw1_con = bcon['dw1']['consumed'] * from_si
        else:
            dw1_con = dw1_pre  # 100% default
        dw1_pos = dw1_pre - dw1_con

        # 10-hr
        if 'dw10' in bcon:
            dw10_con = bcon['dw10']['consumed'] * from_si
        else:
            dw10_con = dw10_pre
        dw10_pos = dw10_pre - dw10_con

        # 100-hr
        if 'dw100' in bcon:
            dw100_con = bcon['dw100']['consumed'] * from_si
        else:
            pct_100 = float(np.clip(0.95 - 0.008 * max(dw10m - 10.0, 0.0), 0.50, 1.00))
            dw100_con = dw100_pre * pct_100
        dw100_pos = dw100_pre - dw100_con

        # 1000-hr sound: sum all sound size classes
        snd_consumed = 0.0
        snd_flaming = 0.0
        snd_smoldering = 0.0
        for k in ('dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20'):
            if k in bcon:
                snd_consumed += bcon[k]['consumed'] * from_si
                snd_flaming += bcon[k]['flaming'] * from_si
                snd_smoldering += bcon[k]['smoldering'] * from_si
        dw1ks_con = snd_consumed
        dw1ks_pos = dw1ks_pre - dw1ks_con

        # 1000-hr rotten: sum all rotten size classes
        rot_consumed = 0.0
        rot_flaming = 0.0
        rot_smoldering = 0.0
        for k in ('dwk_3_6_r', 'dwk_6_9_r', 'dwk_9_20_r', 'dwk_20_r'):
            if k in bcon:
                rot_consumed += bcon[k]['consumed'] * from_si
                rot_flaming += bcon[k]['flaming'] * from_si
                rot_smoldering += bcon[k]['smoldering'] * from_si
        dw1kr_con = rot_consumed
        dw1kr_pos = dw1kr_pre - dw1kr_con

        # Collect per-class flaming/smoldering for fine fuels from burnup
        fine_fla = 0.0
        fine_smo = 0.0
        for k in ('litter', 'dw1', 'dw10', 'dw100'):
            if k in bcon:
                fine_fla += bcon[k]['flaming'] * from_si
                fine_smo += bcon[k]['smoldering'] * from_si

        # Flaming/smoldering durations from burnup time-series
        fla_dur, smo_dur = _burnup_durations(burnup_results)
    else:
        # Simplified percentage defaults (no burnup)
        # 1-hr: 100 % consumed, flaming
        dw1_con = dw1_pre
        dw1_pos = 0.0
        # 10-hr: 100 % consumed, flaming
        dw10_con = dw10_pre
        dw10_pos = 0.0
        # 100-hr: moisture-based, flaming
        pct_100 = float(np.clip(0.95 - 0.008 * max(dw10m - 10.0, 0.0), 0.50, 1.00))
        dw100_con = dw100_pre * pct_100
        dw100_pos = dw100_pre - dw100_con
        # 1000-hr sound: moisture-based, smoldering
        pct_1ks = float(np.clip(0.15 - 0.001 * dw1km, 0.02, 0.20))
        dw1ks_con = dw1ks_pre * pct_1ks
        dw1ks_pos = dw1ks_pre - dw1ks_con
        # 1000-hr rotten: moisture-based, smoldering
        pct_1kr = float(np.clip(0.45 - 0.003 * dw1km, 0.05, 0.50))
        dw1kr_con = dw1kr_pre * pct_1kr
        dw1kr_pos = dw1kr_pre - dw1kr_con

        snd_flaming = 0.0
        snd_smoldering = dw1ks_con
        rot_flaming = 0.0
        rot_smoldering = dw1kr_con
        fine_fla = dw1_con + dw10_con + dw100_con
        fine_smo = 0.0

        fla_dur = _s(flame_res_time) if flame_res_time is not None else float('nan')
        smo_dur = float('nan')

    # ── 6. Duff ────────────────────────────────────────────────────────────
    duf_pre = _s(duff)
    duf_dep_pre = _s(duff_depth)

    # When burnup ran with duff, extract duff consumption from burnup's
    # DuffBurn output (matching C++ §7), otherwise use consm_duff regression.
    burnup_duf_con = None
    if burnup_ran and duf_loading_si > 0:
        # Burnup tracks duff smoldering in comp_smoldering[number] (the
        # extra slot beyond the fuel components).  Sum across time steps.
        n_comp = len(burnup_class_order)
        duff_smo_total_si = 0.0
        for r in burnup_results:
            if r.comp_smoldering is not None and len(r.comp_smoldering) > n_comp:
                duff_smo_total_si += r.comp_smoldering[n_comp] * burnup_dt
        # Convert back to user units
        burnup_duf_con = duff_smo_total_si * from_si if burnup_ran else None

    if burnup_duf_con is not None and burnup_duf_con > 0:
        duf_con = min(burnup_duf_con, duf_pre)
    else:
        # Fallback: use consm_duff regression
        duff_res = consm_duff(
            duff, duff_moist,
            reg=region,
            cvr_grp=cvr_grp if cvr_grp else None,
            duff_moist_cat='edm',
            d_pre=duff_depth,
            units=units,
        )
        pdc = float(np.clip(_s(duff_res['pdc']), 0.0, 100.0))
        duf_con = duf_pre * pdc / 100.0

    duf_pos = duf_pre - duf_con

    # Duff depth: always use regression for depth outputs (burnup doesn't
    # model depth directly)
    duff_res_depth = consm_duff(
        duff, duff_moist,
        reg=region,
        cvr_grp=cvr_grp if cvr_grp else None,
        duff_moist_cat='edm',
        d_pre=duff_depth,
        units=units,
    )
    rdd = duff_res_depth['rdd']
    ddc_val = duff_res_depth['ddc']
    if rdd is not None:
        duf_dep_pos = float(np.clip(_s(rdd), 0.0, duf_dep_pre))
        duf_dep_con = duf_dep_pre - duf_dep_pos
    elif ddc_val is not None:
        duf_dep_con = float(np.clip(_s(ddc_val), 0.0, duf_dep_pre))
        duf_dep_pos = duf_dep_pre - duf_dep_con
    else:
        duf_dep_con = float('nan')
        duf_dep_pos = float('nan')

    if region in ('InteriorWest', 'PacificWest'):
        if cvr_grp in ('Ponderosa pine', 'PN', 'Ponderosa'):
            duf_eq = 4
        else:
            duf_eq = 2
    elif region == 'SouthEast':
        duf_eq = 20 if cvr_grp in ('Pocosin', 'PC') else 16
    elif region in ('NorthEast',):
        duf_eq = 3
    else:
        duf_eq = 2

    # ── 7. Mineral soil exposure ──────────────────────────────────────────
    mse = float(np.clip(
        _s(consm_mineral_soil(
            region, cvr_grp if cvr_grp else '', fuel_category.lower(),
            duff_moist, 'edm',
        )),
        0.0, 100.0,
    ))
    mse_eq = 10

    # ── 8. Herbaceous ─────────────────────────────────────────────────────
    her_pre = _s(herb)
    her_con = _s(consm_herb(region, cvr_grp if cvr_grp else '', litter, herb, season=season, units=units))
    her_pos = her_pre - her_con

    if region == 'SouthEast':
        herb_eq = 222
    elif cvr_grp in ('Grass', 'GG', 'GrassGroup'):
        herb_eq = 221
    elif cvr_grp in ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd'):
        herb_eq = 223
    else:
        herb_eq = 22

    # ── 9. Shrub ──────────────────────────────────────────────────────────
    shr_pre = _s(shrub)
    slc_pct = _s(consm_shrub(region, cvr_grp if cvr_grp else '', shrub, season=season, units=units))
    shr_con = shr_pre * float(np.clip(slc_pct, 0.0, 100.0)) / 100.0
    shr_pos = shr_pre - shr_con

    if region == 'Southeast':
        shrub_eq = 234 if cvr_grp not in ('Pocosin', 'PC') else 233
    elif cvr_grp in ('Sagebrush', 'SB'):
        shrub_eq = 232
    elif cvr_grp in ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd'):
        shrub_eq = 236
    elif cvr_grp in ('Shrub', 'SG', 'ShrubGroup'):
        shrub_eq = 231
    else:
        shrub_eq = 23

    # ── 10. Canopy (crown fire) ───────────────────────────────────────────
    crown_res = consm_canopy(pct_crown_burned, crown_foliage, crown_branch, units=units)
    fol_pre = _s(crown_foliage)
    fol_con = _s(crown_res['flc'])
    fol_pos = fol_pre - fol_con

    bra_pre = _s(crown_branch)
    bra_con = _s(crown_res['blc'])
    bra_pos = bra_pre - bra_con

    # ── 11. Flaming / smoldering totals ───────────────────────────────────
    # Flaming: litter, fine woody (from burnup or defaults), herb, shrub,
    #          crown foliage, crown branch, sound/rotten flaming portion
    # Smoldering: duff, sound/rotten smoldering portion
    # §8: Use burnup's flaming/smoldering partition for litter when available
    if burnup_ran and 'litter' in bcon:
        lit_fla = bcon['litter']['flaming'] * from_si
        lit_smo = bcon['litter']['smoldering'] * from_si
    else:
        lit_fla = lit_con  # litter is all flaming (fallback)
        lit_smo = 0.0
    fla_con = (lit_fla + fine_fla + her_con + shr_con
               + fol_con + bra_con + snd_flaming + rot_flaming)
    smo_con = duf_con + lit_smo + fine_smo + snd_smoldering + rot_smoldering

    # Smoldering duration fallback from duff burning rate
    if not burnup_ran or smo_dur == 0.0:
        if not np.isnan(duf_dep_con) and duf_dep_con > 0.0:
            dep_cm = (duf_dep_con * _IN_TO_CM if is_imperial else duf_dep_con)
            burn_rate = max(0.05 * np.exp(-0.025 * _s(duff_moist)), 1e-6)  # cm/min
            smo_dur = (dep_cm / burn_rate) * 60.0  # convert min → sec

    # Flaming duration fallback
    if not burnup_ran:
        fla_dur = _s(flame_res_time) if flame_res_time is not None else float('nan')

    # ── 12. Smoke emissions ───────────────────────────────────────────────
    emissions = calc_smoke_emissions(
        fla_con, smo_con,
        mode=em_mode,
        ef_group=ef_group,
        duff_load=duf_con,
        ef_csv_path=ef_csv_path,
        units=units,
    )

    # ── 13. Assemble and return ───────────────────────────────────────────
    return {
        # Litter
        'LitPre':     lit_pre,  'LitCon':     lit_con,  'LitPos':     lit_pos,
        # 1-hr
        'DW1Pre':     dw1_pre,  'DW1Con':     dw1_con,  'DW1Pos':     dw1_pos,
        # 10-hr
        'DW10Pre':    dw10_pre, 'DW10Con':    dw10_con, 'DW10Pos':    dw10_pos,
        # 100-hr
        'DW100Pre':   dw100_pre, 'DW100Con':  dw100_con, 'DW100Pos':  dw100_pos,
        # 1000-hr sound
        'DW1kSndPre': dw1ks_pre, 'DW1kSndCon': dw1ks_con, 'DW1kSndPos': dw1ks_pos,
        # 1000-hr rotten
        'DW1kRotPre': dw1kr_pre, 'DW1kRotCon': dw1kr_con, 'DW1kRotPos': dw1kr_pos,
        # Duff
        'DufPre':     duf_pre,  'DufCon':     duf_con,  'DufPos':     duf_pos,
        # Herbaceous
        'HerPre':     her_pre,  'HerCon':     her_con,  'HerPos':     her_pos,
        # Shrub
        'ShrPre':     shr_pre,  'ShrCon':     shr_con,  'ShrPos':     shr_pos,
        # Crown foliage
        'FolPre':     fol_pre,  'FolCon':     fol_con,  'FolPos':     fol_pos,
        # Crown branch
        'BraPre':     bra_pre,  'BraCon':     bra_con,  'BraPos':     bra_pos,
        # Mineral soil
        'MSE':        mse,
        # Duff depth
        'DufDepPre':  duf_dep_pre, 'DufDepCon': duf_dep_con, 'DufDepPos': duf_dep_pos,
        # Emissions (14 pollutant × 2 phase slots)
        **emissions,
        # Flaming/smoldering aggregates
        'FlaDur':  fla_dur, 'SmoDur':  smo_dur,
        'FlaCon':  fla_con, 'SmoCon':  smo_con,
        # Soil temperature layers — populated by soil_heat_campbell/massman
        'Lay0':   float('nan'), 'Lay2':   float('nan'),
        'Lay4':   float('nan'), 'Lay6':   float('nan'),
        'Lay60d': float('nan'), 'Lay275d': float('nan'),
        # Equation identifiers
        'Lit-Equ':    lit_eq,
        'DufCon-Equ': duf_eq,
        'DufRed-Equ': duf_eq,
        'MSE-Equ':    mse_eq,
        'Herb-Equ':   herb_eq,
        'Shurb-Equ':  shrub_eq,
    }
