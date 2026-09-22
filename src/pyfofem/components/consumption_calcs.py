# -*- coding: utf-8 -*-
"""
consumption_calcs.py – FOFEM fuel consumption and related calculations.

Provides fuel consumption models for each fuel component, as well as helpers
for running the Burnup post-frontal combustion model and calculating carbon
loadings.

Functions:
    calc_carbon            – Convert fuel loadings to carbon loadings.
    consm_canopy           – Crown/canopy fuel consumption.
    consm_duff             – Duff fuel consumption.
    consm_herb             – Herbaceous fuel consumption.
    consm_litter           – Litter fuel consumption.
    consm_mineral_soil     – Mineral soil exposure.
    consm_shrub            – Shrub fuel consumption.
    _extract_burnup_consumption     – Internal helper to extract per-class consumption.
    _burnup_durations               – Internal helper to extract flaming/smoldering durations.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .burnup import (
    FuelParticle,
    BurnResult,
    BurnSummaryRow,
    BurnupValidationError,
    burnup as _burnup,
    _FIRE_BOUNDS,
    _FUEL_BOUNDS,
    _BURNUP_LIMIT_ADJUST,
    _BURNUP_LIMIT_ERROR,
)
from ._component_helpers import _is_scalar, _maybe_scalar, _to_str_arr


# ---------------------------------------------------------------------------
# Output variable lists
# ---------------------------------------------------------------------------

DEFAULT_CONSUMPTION_VARS = [
    'LitPre', 'LitCon', 'LitPos', 'DW1Pre', 'DW1Con', 'DW1Pos', 'DW10Pre', 'DW10Con', 'DW10Pos',
    'DW100Pre', 'DW100Con', 'DW100Pos', 'DW1kSndPre', 'DW1kSndCon', 'DW1kSndPos',
    'DW1kRotPre', 'DW1kRotCon', 'DW1kRotPos', 'DufPre', 'DufCon', 'DufPos',
    'HerPre', 'HerCon', 'HerPos', 'ShrPre', 'ShrCon', 'ShrPos',
    'FolPre', 'FolCon', 'FolPos', 'BraPre', 'BraCon', 'BraPos',
    'MSE', 'DufDepPre', 'DufDepCon', 'DufDepPos',
    'PM10F', 'PM10S', 'PM25F', 'PM25S', 'CH4F', 'CH4S', 'COF', 'COS', 'CO2F', 'CO2S',
    'NOXF', 'NOXS', 'SO2F', 'SO2S',
]
EXPANDED_CONSUMPTION_VARS = ['PM10S_Duff', 'PM25S_Duff', 'CH4S_Duff',
                             'COS_Duff', 'CO2S_Duff', 'NOXS_Duff', 'SO2S_Duff']
TOTAL_DURATION_CONSUMED_VARS = ['FlaDur', 'SmoDur', 'FlaCon', 'SmoCon',]
SOIL_HEAT_VARS = ['Lay0', 'Lay2', 'Lay4', 'Lay6', 'Lay60d', 'Lay275d']
EQUATION_VARS = ['Lit-Equ', 'DufCon-Equ', 'DufRed-Equ', 'MSE-Equ', 'Herb-Equ', 'Shrub-Equ']
ERROR_VARS = ['BurnupLimitAdj', 'BurnupError']

#: FOFEM's Pine Flatwoods equations work internally in Mg/ha while the
#: surrounding consumption API uses T/ac for Imperial inputs. The pinned C++
#: uses this rounded factor in ``Ton_To_Mega``/``Mega_To_Ton``
#: (``fof_hsf.cpp:828-839``).
T_ACRE_PER_MG_HECTARE = 0.446

#: T/ac -> Mg/ha (metric tons/hectare) conversion, used ONLY by the Coastal
#: Plain forest-floor equation (``Equ_CP_Per``). This is a DIFFERENT, more
#: precise factor than :data:`T_ACRE_PER_MG_HECTARE` above (0.446, a
#: Pine-Flatwoods-specific rounded constant from ``Ton_To_Mega``/
#: ``Mega_To_Ton``, ``fof_hsf.cpp:828-839``) -- the two must not be
#: interchanged. Matches C++'s ``TPA_To_MTPH`` exactly
#: (``fof_util.cpp:657-662``: ``f = 0.907184 * TPA``).
MG_HECTARE_PER_TON_ACRE = 0.907184

#: Inclusive litter-moisture bounds for the Coastal Plain forest-floor
#: equation, matching C++'s ``e_LitMoiMin``/``e_LitMoiMax``
#: (``fof_ci.h:61-62``) and enforced the same way C++'s ``_ChkLitMoist()``
#: does (``fof_hsf.cpp:614-626``) -- only for Coastal Plain cover types.
_CP_LIT_MOIST_MIN = 1.0
_CP_LIT_MOIST_MAX = 100.0

#: Case-insensitive Coastal Plain cover-group aliases, matching C++'s
#: ``CI_isCoastPlain()`` (``fof_ci.cpp:185``), which accepts either
#: ``e_CoastPlain`` ("CoastPlain") or ``e_CVT_CoastPlain`` ("CP")
#: case-insensitively (``xstrcmpi``). Deliberately excludes "Coastal Plain"
#: (two words) and any integer code -- not part of this pass's scope.
_COASTPLAIN_ALIASES = frozenset({'cp', 'coastplain'})


def _check_cp_litter_moisture(l_moist: np.ndarray) -> None:
    """
    Enforce C++'s inclusive Coastal Plain litter-moisture range, matching
    ``_ChkLitMoist()`` (``fof_hsf.cpp:614-626``).

    :param l_moist: Litter moisture content (%). np.ndarray.
    :raises ValueError: If any value is missing (``NaN``) or outside
        ``[1.0, 100.0]`` inclusive.
    """
    bad = ~np.isfinite(l_moist) | (l_moist < _CP_LIT_MOIST_MIN) | (l_moist > _CP_LIT_MOIST_MAX)
    if np.any(bad):
        bad_val = float(np.asarray(l_moist)[bad][0])
        raise ValueError(
            f"Litter Moisture {bad_val:.2f} is out of limits "
            f"({_CP_LIT_MOIST_MIN:.2f} -> {_CP_LIT_MOIST_MAX:.2f}). "
            "Litter Moisture is required for Coastal Plain cover types "
            "(matches C++ _ChkLitMoist(), fof_hsf.cpp:614-626)."
        )

def _coastal_plain_forest_floor(
        pre_ll: np.ndarray, pre_dl: np.ndarray, l_moist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared core of C++ ``Equ_CP_Per`` (Coastal Plain combined litter/duff
    percent consumed, equation ID 30, ``fof_duf.cpp:1171-1225``).

    Litter is consumed first; duff only receives whatever remains of the
    total-consumed budget beyond the full pre-fire litter load, exactly
    matching the C++ allocation (``Dc = Tc - Ll``, ``Lc = Tc - Dc``).

    :param pre_ll: Pre-fire litter load (T/ac). np.ndarray.
    :param pre_dl: Pre-fire duff load (T/ac). np.ndarray, same shape.
    :param l_moist: Litter moisture content (%). np.ndarray, same shape.
    :return: ``(litter_pct, duff_pct)``, each clamped to ``[0, 100]`` and
        ``0`` wherever the combined pre-fire litter+duff load is not
        strictly positive (matching C++'s ``if ((Lit+Duff)<=0) return;``
        early exit, which leaves both percents at their zero defaults).
    """
    ll_mgha = pre_ll * MG_HECTARE_PER_TON_ACRE
    dl_mgha = pre_dl * MG_HECTARE_PER_TON_ACRE
    x1_load = ll_mgha + dl_mgha
    has_load = x1_load > 0

    total_consumed = -3.893 + (0.944 * x1_load) - (0.078 * l_moist)
    total_consumed = np.minimum(total_consumed, x1_load)
    total_consumed = np.maximum(total_consumed, 0.0)

    duff_consumed = np.maximum(total_consumed - ll_mgha, 0.0)
    litter_consumed = total_consumed - duff_consumed

    dl_safe = np.where(dl_mgha > 0, dl_mgha, 1.0)
    ll_safe = np.where(ll_mgha > 0, ll_mgha, 1.0)
    duff_pct = np.where(dl_mgha > 0, 100.0 * duff_consumed / dl_safe, 0.0)
    litter_pct = np.where(ll_mgha > 0, 100.0 * litter_consumed / ll_safe, 0.0)

    duff_pct = np.where(has_load, np.clip(duff_pct, 0.0, 100.0), 0.0)
    litter_pct = np.where(has_load, np.clip(litter_pct, 0.0, 100.0), 0.0)
    return litter_pct, duff_pct


def _is_coastal_plain(cvr_grp: Optional[str]) -> bool:
    """
    Case-insensitive Coastal Plain cover-group check, matching C++'s
    ``CI_isCoastPlain()`` (``fof_ci.cpp:185``).

    :param cvr_grp: Cover group value, or ``None``.
    :return: ``True`` when *cvr_grp* is ``'CP'`` or ``'CoastPlain'``
        (any case), ``False`` otherwise (including ``None`` or a
        non-string value).
    """
    if not isinstance(cvr_grp, str):
        return False
    return cvr_grp.strip().lower() in _COASTPLAIN_ALIASES


# ---------------------------------------------------------------------------
# Categorical parameter lookup tables (int code → string label)
# ---------------------------------------------------------------------------

#: Integer codes for FOFEM regions.
REGION_CODES: Dict[int, str] = {
    1: 'InteriorWest',
    2: 'PacificWest',
    3: 'NorthEast',
    4: 'SouthEast',
}

#: Integer codes for FOFEM cover groups.
CVR_GRP_CODES: Dict[int, str] = {
    0:  '',
    1:  'Ponderosa pine',
    2:  'Pocosin',
    3:  'Chaparral',
    4:  'Shrub-Chaparral',
    5:  'Sagebrush',
    6:  'Flatwood',
    7:  'Pine Flatwoods',
    8:  'Red Jack Pine',
    9:  'Red, Jack Pine',
    10: 'Grass',
    11: 'Shrub',
    # Short aliases
    12: 'PN',
    13: 'PC',
    14: 'SGC',
    15: 'ShrubGroupChaparral',
    16: 'SB',
    17: 'PFL',
    18: 'PinFltwd',
    19: 'RJP',
    20: 'RedJacPin',
    21: 'GG',
    22: 'GrassGroup',
    23: 'SG',
    24: 'ShrubGroup',
}

#: Integer codes for burn seasons.
SEASON_CODES: Dict[int, str] = {
    1: 'Spring',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter',
}

#: Integer codes for fuel categories.
FUEL_CATEGORY_CODES: Dict[int, str] = {
    1: 'Natural',
    2: 'Slash',
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# Fuel component groups and their carbon conversion factors (Penman 2003;
# Smith & Heath 2002), used only by calc_carbon().
_CARBON_FACTOR_WOODY = 0.50   # down woody, herb, shrub, foliage, branch
_CARBON_FACTOR_DUFF  = 0.37   # duff and litter

_CARBON_WOODY_KEYS = frozenset({
    'dw1', 'dw10', 'dw100',
    'dwk_3_6', 'dwk_6_9', 'dwk_9_20', 'dwk_20',
    'herb', 'shrub', 'foliage', 'branch',
})
_CARBON_DUFF_KEYS = frozenset({'duff', 'litter'})


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
    :return: Dict with the same keys as *loadings* and carbon loading values
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

    crown_burn = np.ravel(np.asarray(crown_burn, dtype=float))
    pre_fl = np.ravel(np.asarray(pre_fl, dtype=float))
    pre_bl = np.ravel(np.asarray(pre_bl, dtype=float))

    if units.upper() == 'SI':
        pre_fl = pre_fl * 4.4609  # kg/m² → T/acre
        pre_bl = pre_bl * 4.4609

    flc = (crown_burn / 100) * pre_fl
    blc = (crown_burn / 100) * pre_bl * 0.5

    if units.upper() == 'SI':
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
        mc_lyr1: Optional[Union[float, np.ndarray]] = None,
        pre_dl110: Optional[Union[float, np.ndarray]] = None,
        pre_l110: Optional[Union[float, np.ndarray]] = None,
        dw1000_moist: Optional[Union[float, np.ndarray]] = None,
        pile: bool = False,
        units: str = 'SI',
        pre_ll: Optional[Union[float, np.ndarray]] = None,
        l_moist: Optional[Union[float, np.ndarray]] = None,
) -> dict:
    """
    FOFEM duff consumption model.

    Mirrors the logic of ``DUF_Mngr`` / region sub-functions in the C++
    source (``fof_duf.cpp``).  Computes:

    * ``'pdc'`` – percent of duff load consumed (%).
    * ``'ddc'`` – duff depth consumed (same depth units as *d_pre*).
    * ``'rdd'`` – residual (post-fire) duff depth (same units as *d_pre*).

    **Depth reduction approach** – Per C++ Note-5 (``DUF_Mngr``, 2016), the
    raw regression-based depth-reduction equations (Eqs 5, 6, 7) are no
    longer used for normal runs.  Instead, depth consumed is derived directly
    from the percent consumed::

        ddc = d_pre × (pdc / 100)
        rdd = d_pre − ddc

    This matches the final line of ``DUF_Mngr``:
    ``a_DUF->f_Red = a_CI->f_DufDep * (a_DUF->f_Per / 100.0)``.

    **Percent consumed routing** (matches ``DUF_Mngr``/``DUF_NorthEast``
    priority order, ``fof_duf.cpp:306-483``):

    +---------------------+---------------+------------+--------+
    | Region              | Cover type    | Moist cat  | Eq     |
    +=====================+===============+============+========+
    | Any                 | Piles         | —          | 17     |
    +---------------------+---------------+------------+--------+
    | Any                 | Chaparral/SGC | —          | 19     |
    +---------------------+---------------+------------+--------+
    | InteriorWest /      | Ponderosa     | ldm        | 4      |
    | PacificWest         | Ponderosa     | edm        | 2      |
    |                     | Ponderosa     | nfdth      | 3      |
    |                     | Other         | ldm        | 1      |
    |                     | Other         | edm        | 2      |
    |                     | Other         | nfdth      | 3      |
    +---------------------+---------------+------------+--------+
    | NorthEast           | WhiPinHem     | any        | (delegates to |
    |                     |               |            | InteriorWest, |
    |                     |               |            | see above)    |
    |                     | RedJacPine    | edm        | 15 (pine=1)   |
    |                     | RedJacPine    | ldm        | 2 (Duf_Default)|
    |                     | RedJacPine    | nfdth      | 3             |
    |                     | BalsamSpruce  | ldm        | 5→pct         |
    |                     | BalsamSpruce  | edm        | 15 (pine=0)   |
    |                     | BalsamSpruce  | nfdth      | 3             |
    |                     | Other         | any        | 2 (Duf_Default,|
    |                     |               |            | UNCONDITIONAL —|
    |                     |               |            | never branches |
    |                     |               |            | on moist cat)  |
    +---------------------+---------------+------------+--------+
    | SouthEast           | Pocosin       | —          | 20     |
    |                     | CP/CoastPlain | —          | 30     |
    |                     | Other         | —          | 16     |
    +---------------------+---------------+------------+--------+
    | Any / fallback      | —             | —          | 2      |
    +---------------------+---------------+------------+--------+

    **Depth (``ddc``/``rdd``) is ALWAYS percent-derived**, never a
    per-region regression equation, for every region and fuel category —
    see the "Depth reduction approach" note above and the implementation's
    own comment at ``fof_duf.cpp:290-300`` (Note-5) / ``:395``. The
    per-region depth-reduction equations (Eq 5/6/7/15/etc.) referenced
    implicitly above (percent-only equations like 1-4/17/19 have no depth
    equation of their own) are still computed by the pinned C++ but
    discarded before being returned to the caller.

    **Coastal Plain** (``cvr_grp`` ``'CP'`` or ``'CoastPlain'``, matching
    C++'s ``CI_isCoastPlain()`` case-insensitively) is a SouthEast COVER
    GROUP, not a region — C++'s special route lives entirely inside
    ``DUF_SouthEast()`` (``fof_duf.cpp:409-426``). Supplying a Coastal Plain
    ``cvr_grp`` with any ``reg`` other than ``'SouthEast'`` raises
    ``ValueError`` rather than silently falling through to an unrelated
    regional equation — this is PyFOFEM's own supported contract; the C++
    special branch itself is likewise SouthEast-only by construction
    (``DUF_Mngr`` only calls ``DUF_SouthEast()`` when ``CI_isSouthEast()``).
    Requires *pre_ll* and *l_moist* (raises ``ValueError`` naming the
    missing/out-of-range input otherwise); litter moisture is enforced to
    C++'s inclusive ``[1.0, 100.0]`` range (``fof_ci.h:61-62``,
    ``_ChkLitMoist()``, ``fof_hsf.cpp:614-626``). The pre-existing global
    ``duff_moist <= 10`` override (100% duff consumed) and the final
    ``ddc = d_pre * (pdc/100)`` depth derivation below both still apply
    unchanged on top of Eq 30's ``pdc`` — this mirrors the CURRENT C++
    production result exactly: ``DUF_Mngr`` overwrites ``Equ_CP_Red``'s own
    historical regression value with this same percent-derived depth after
    recording equation ID 31 (``fof_duf.cpp:388-391``). Litter percent
    consumed (equation ID 30, shared with duff) is available from
    :func:`consm_litter` (pass the same *pre_dl* as this call's *pre_dl* via
    its own *pre_dl* parameter) — both derive from the same
    :func:`_coastal_plain_forest_floor` helper so the two are never
    inconsistent for identical inputs.

    :param pre_dl: Pre-fire duff load (Mg/ha if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param reg: Region name. One of ``'InteriorWest'``, ``'PacificWest'``,
        ``'NorthEast'``, ``'SouthEast'``.
    :param cvr_grp: Cover group name (e.g. ``'Ponderosa pine'``,
        ``'Pocosin'``, ``'Chaparral'``).
    :param duff_moist_cat: Duff moisture category. One of:

        - ``'ldm'`` – lower duff moisture
        - ``'edm'`` – entire / average duff moisture
        - ``'nfdth'`` – NFDR 1,000-hour moisture content

    :param d_pre: Pre-fire duff depth (cm if ``units='SI'``, inches if
        ``units='Imperial'``). Required for ``'ddc'`` and ``'rdd'`` outputs.
    :param mc_lyr1: Surface-layer moisture content (%). Required for Eq 20
        (SouthEast Pocosin).
    :param pre_dl110: Pre-fire duff + litter + 10-hr load (same mass units as
        *pre_dl*). Required for Eq 16 (SouthEast non-Pocosin).
    :param pre_l110: Pre-fire litter + 10-hr load (same units). Required for
        Eq 16.
    :param dw1000_moist: 1000-hr fuel moisture content (%). Used by Eq 3 and
        the NorthEast Balsam/Spruce nfdth path.  Falls back to *duff_moist*
        when ``None``.
    :param pile: ``True`` for pile burning (Eq 17 – 10 % consumed).
    :param units: ``'SI'`` (default, kg/m² / cm) or ``'Imperial'`` (T/ac /
        in).
    :param pre_ll: Pre-fire litter load (same mass units as *pre_dl*).
        Required for Coastal Plain (Eq 30).
    :param l_moist: Litter moisture content (%). Required for Coastal Plain
        (Eq 30); enforced to the inclusive ``[1.0, 100.0]`` range.

    :return: Dict with keys ``'pdc'``, ``'ddc'``, ``'rdd'``.
        ``'ddc'`` and ``'rdd'`` are ``None`` when *d_pre* is not supplied.
    :raises ValueError: If *cvr_grp* is Coastal Plain and *reg* is not
        ``'SouthEast'``; if *pre_ll* or *l_moist* is missing for Coastal
        Plain; or if *l_moist* is outside ``[1.0, 100.0]``.
    """
    scalar_input = _is_scalar(pre_dl) and _is_scalar(duff_moist)

    pre_dl     = np.ravel(np.asarray(pre_dl,     dtype=float))
    duff_moist = np.ravel(np.asarray(duff_moist, dtype=float))
    if d_pre is not None:
        d_pre = np.ravel(np.asarray(d_pre, dtype=float))
    if mc_lyr1 is not None:
        mc_lyr1 = np.ravel(np.asarray(mc_lyr1, dtype=float))
    if pre_dl110 is not None:
        pre_dl110 = np.ravel(np.asarray(pre_dl110, dtype=float))
    if pre_l110 is not None:
        pre_l110 = np.ravel(np.asarray(pre_l110, dtype=float))
    if pre_ll is not None:
        pre_ll = np.ravel(np.asarray(pre_ll, dtype=float))
    if l_moist is not None:
        l_moist = np.ravel(np.asarray(l_moist, dtype=float))

    # Eq 3 / NE-Balsam-nfdth use 1000-hr moisture (C++ f_MoistDW1000).
    # Fall back to duff_moist when not supplied.
    if dw1000_moist is not None:
        dw1k = np.ravel(np.asarray(dw1000_moist, dtype=float))
    else:
        dw1k = duff_moist

    if units.upper() == 'SI':
        pre_dl = pre_dl * 4.4609                 # Mg/ha → T/acre
        if d_pre is not None:
            d_pre = d_pre / 2.54                 # cm → in
        if pre_dl110 is not None:
            pre_dl110 = pre_dl110 * 4.4609
        if pre_l110 is not None:
            pre_l110  = pre_l110  * 4.4609
        if pre_ll is not None:
            pre_ll = pre_ll * 4.4609

    # ------------------------------------------------------------------
    # Convenience flag sets (matching C++ CI_is* predicates)
    # ------------------------------------------------------------------
    _IW_PW     = {'InteriorWest', 'PacificWest'}
    _PONDEROSA = {'Ponderosa pine', 'PN', 'Ponderosa'}
    _POCOSIN   = {'Pocosin', 'PC'}
    _WHITE_PINE_HEMLOCK = {'White Pine Hemlock', 'WhiPinHem', 'WPH'}
    _CHAPARRAL = {'Chaparral', 'Shrub-Chaparral', 'SGC', 'ShrubGroupChaparral'}
    _REDJAC    = {'Red Jack Pine', 'Red, Jack Pine', 'RedJacPin', 'RJP'}
    _BALSAM    = {'Balsam', 'Black Spruce', 'Red Spruce', 'White Spruce',
                  'BalBRWSpr', 'Balsam Fir', 'BFS'}

    is_iw_pw    = reg in _IW_PW
    is_ne       = reg == 'NorthEast'
    is_se       = reg == 'SouthEast'
    is_ponderosa = cvr_grp in _PONDEROSA
    is_pocosin  = cvr_grp in _POCOSIN
    is_white_pine_hemlock = cvr_grp in _WHITE_PINE_HEMLOCK
    is_chaparral = cvr_grp in _CHAPARRAL
    is_redjac   = cvr_grp in _REDJAC
    is_balsam   = cvr_grp in _BALSAM
    is_coastplain = _is_coastal_plain(cvr_grp)

    # PyFOFEM's own supported contract (stricter than raw C++, which would
    # silently fall through to whatever OTHER regional equation "reg"
    # happens to select): Coastal Plain is a SouthEast cover group, not a
    # region, and its special route lives only inside DUF_SouthEast()
    # (fof_duf.cpp:409-426). A non-SouthEast region with a Coastal Plain
    # cvr_grp fails loudly instead of silently applying an unrelated
    # regional equation.
    if is_coastplain and not is_se:
        raise ValueError(
            f"consm_duff(): Coastal Plain (cvr_grp={cvr_grp!r}) is only "
            "supported for reg='SouthEast' (C++'s DUF_SouthEast-only "
            f"routing, fof_duf.cpp:409-426); got reg={reg!r}."
        )

    # ------------------------------------------------------------------
    # pdc – percent consumed
    # Priority mirrors DUF_Mngr: Piles → Chaparral → region branches
    # ------------------------------------------------------------------
    pdc = np.full_like(duff_moist, np.nan)

    if pile:
        # Eq 17 – pile burning: 10 %
        pdc = np.full_like(duff_moist, 10.0)

    elif is_chaparral:
        # Eq 19 – Chaparral/SGC: 100 %
        pdc = np.full_like(duff_moist, 100.0)

    elif is_iw_pw or (is_ne and is_white_pine_hemlock):
        # PacificWest Slash → same as InteriorWest (C++ DUF_PacificWest Note-1)
        if is_ponderosa:
            if duff_moist_cat == 'ldm':
                pdc = 89.9 - 0.55 * duff_moist                  # Eq 4
            elif duff_moist_cat == 'edm':
                pdc = 83.7 - 0.426 * duff_moist                 # Eq 2
            else:                                                 # nfdth
                pdc = 114.7 - 4.2 * dw1k                        # Eq 3
        else:
            if duff_moist_cat == 'ldm':
                pdc = np.where(duff_moist <= 160,
                               97.1 - 0.519 * duff_moist, 13.6) # Eq 1
            elif duff_moist_cat == 'edm':
                pdc = 83.7 - 0.426 * duff_moist                 # Eq 2
            else:                                                 # nfdth
                pdc = 114.7 - 4.2 * dw1k                        # Eq 3

    elif is_ne:
        if is_redjac:
            if duff_moist_cat == 'edm':
                # C++ Jack_Red_Pine + Duf Entire -> Equ_15_RedPer(..., "JACK")
                # (fof_duf.cpp:458-461): Eq 15 with pine=1, pdc derived from
                # residual depth.
                f_rdd = (-0.791 + 0.004 * duff_moist
                         + 0.8 * (d_pre if d_pre is not None
                                  else np.zeros_like(duff_moist))
                         + 0.56)
                f_red = np.clip(
                    (d_pre if d_pre is not None
                     else np.zeros_like(duff_moist)) - f_rdd,
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            elif duff_moist_cat == 'ldm':
                # C++ Jack_Red_Pine + Duf Lower -> Duf_Default() (fof_duf.cpp:
                # 462-464) -- NOT a RedJacPine-specific equation; C++ falls
                # through to the SAME Eq 2 the generic NorthEast case uses.
                pdc = 83.7 - 0.426 * duff_moist
            else:
                # C++ Jack_Red_Pine + NFDR/Adj-NFDR -> Equ_3_Red (fof_duf.cpp:
                # 465-467), whose own Equ_3_Per is Eq 3 on 1000-hr moisture --
                # matches Balsam's and IW/PW's own nfdth routing exactly.
                pdc = 114.7 - 4.2 * dw1k
        elif is_balsam:
            if duff_moist_cat == 'ldm':
                # Eq 5 → derive pdc from depth reduction
                f_red_5 = np.clip(
                    1.028 - 0.0089 * duff_moist
                    + 0.417 * (d_pre if d_pre is not None
                               else np.zeros_like(duff_moist)),
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red_5 / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            elif duff_moist_cat == 'edm':
                # Eq 15 with pine=0
                f_rdd = (-0.791 + 0.004 * duff_moist
                         + 0.8 * (d_pre if d_pre is not None
                                  else np.zeros_like(duff_moist)))
                f_red = np.clip(
                    (d_pre if d_pre is not None
                     else np.zeros_like(duff_moist)) - f_rdd,
                    0.0, None,
                )
                pdc = np.where(
                    (d_pre is not None) and (d_pre > 0),
                    np.clip((f_red / np.maximum(
                        d_pre if d_pre is not None
                        else np.ones_like(duff_moist), 1e-12)) * 100, 0, 100),
                    0.0,
                )
            else:
                # nfdth → Eq 3
                pdc = 114.7 - 4.2 * dw1k
        else:
            # C++ DUF_NorthEast's generic fallback (non-WhiPinHem,
            # non-RedJacPin, non-BalBRWSpr) calls Duf_Default()
            # UNCONDITIONALLY (fof_duf.cpp:454-455) -- Equ_2_Per, with NO
            # branching on duff_moist_method at all. F-23's case-6 defect
            # was exactly this: the prior code below used Eq 15 whenever
            # duff_moist_cat == 'edm', but C++ never reaches Equ_15_RedPer
            # for this generic cover-group bucket -- only for RedJacPine/
            # Balsam specifically (handled in the branches above).
            pdc = 83.7 - 0.426 * duff_moist

    elif is_se:
        if is_pocosin:
            # Eq 20 – Pocosin per-layer load-based algorithm (C++ Equ_20_PerRed_Pocosin)
            # Works on duff load per 4-inch layer; mc_lyr1 is the top-layer moisture.
            mc0    = float(mc_lyr1[0]) if mc_lyr1 is not None else float(duff_moist[0])
            dl_val = float(pre_dl[0])
            dp_val = float(d_pre[0]) if d_pre is not None else 0.0

            if dp_val > 0 and dl_val > 0:
                f_10th        = dl_val / (dp_val * 10.0)  # load per 0.1-inch slice
                layer_load    = f_10th * 40.0 if dp_val >= 4.0 else dl_val  # per 4-in layer
                dep_rem       = dp_val
                duf_rem       = dl_val
                mc_layer      = mc0
                moi_inc       = 0.0
                tot_consumed  = 0.0
                _E_MINERAL    = 5.0

                while True:
                    cur_layer_load = layer_load if dep_rem >= 4.0 else duf_rem
                    if mc_layer < 10.0:
                        f_per = 1.0
                    elif mc_layer < 30.0:
                        f_per = 1.0 - (mc_layer * 0.00167)
                    elif mc_layer < 140.0:
                        f_per = 1.0 / (1.0 + np.exp(-1.0 * (
                            2.033 - (0.043 * mc_layer) + (0.44 * _E_MINERAL))))
                    elif mc_layer < 170.0:
                        f_per = 0.143441 - ((mc_layer - 140.0) * 0.0049)
                    else:
                        f_per = 0.0
                    f_per = max(f_per, 0.0)
                    tot_consumed += cur_layer_load * f_per

                    if dep_rem < 4.0:
                        break
                    dep_rem -= 4.0
                    duf_rem -= layer_load
                    if dep_rem < 4.0:
                        layer_load = duf_rem
                    moi_inc = min(moi_inc + 3.0, 12.0)
                    mc_layer += moi_inc

                pdc_val = np.clip((tot_consumed / dl_val) * 100.0, 0.0, 100.0)
            else:
                pdc_val = 0.0
            pdc = np.ravel(np.full_like(duff_moist, pdc_val))

        elif is_coastplain:
            # Eq 30 – Coastal Plain combined litter/duff percent consumed
            # (C++ Equ_CP_Per, fof_duf.cpp:1171-1225). Only the DUFF percent
            # is used here; consm_litter() derives the litter percent from
            # the same shared helper for consistency.
            if pre_ll is None or l_moist is None:
                raise ValueError(
                    "consm_duff(): Coastal Plain (cvr_grp="
                    f"{cvr_grp!r}) requires both pre_ll (litter load) and "
                    "l_moist (litter moisture %) to compute Eq 30 "
                    "(fof_duf.cpp:1171-1225)."
                )
            _check_cp_litter_moisture(l_moist)
            _lit_pct_cp, duff_pct_cp = _coastal_plain_forest_floor(
                pre_ll, pre_dl, l_moist,
            )
            pdc = np.ravel(duff_pct_cp)

        else:
            # SE non-Pocosin – Eq 16
            # f_WPRE = lit + duff + dw10 + dw1  (here approximated by pre_dl110)
            # f_L    = lit + dw10 + dw1          (here approximated by pre_l110)
            if pre_dl110 is not None and pre_l110 is not None:
                f_wpre = np.where(pre_dl110 > 0, pre_dl110, 0.0)
                f_w = np.where(
                    f_wpre > 0,
                    3.4958 + (0.3833 * f_wpre) - (0.0237 * duff_moist) - (5.6075 / np.maximum(f_wpre, 1e-12)),
                    0.0,
                )
                f_l = pre_l110
                duff_only = f_wpre - f_l
                pdc = np.where(
                    f_w <= f_l, 0.0,
                    np.where(
                        duff_only > 0,
                        np.clip(100.0 * (f_w - f_l) / duff_only, 0.0, 100.0),
                        0.0,
                    ),
                )
            else:
                # Fall back to Eq 2 if required inputs missing
                pdc = 83.7 - 0.426 * duff_moist

    else:
        # Global fallback → Duf_Default → Eq 2
        pdc = 83.7 - 0.426 * duff_moist

    # C++ DUF_Mngr: duff_moist ≤ 10 forces 100 % consumed (Note, 2012)
    low_moist_mask = duff_moist <= 10.0
    pdc = np.where(low_moist_mask, 100.0, pdc)

    # There is nothing to consume when the duff load is zero.  Preserve the
    # moisture override above for ordinary loads, then apply this independent
    # physical boundary condition.
    pdc = np.where(pre_dl <= 0.0, 0.0, pdc)

    # Clamp to [0, 100] (C++ DUF_Mngr Note-1)
    pdc = np.clip(pdc, 0.0, 100.0)

    # ------------------------------------------------------------------
    # ddc / rdd – depth outputs.
    #
    # C++ DUF_Mngr's own Note-5 (fof_duf.cpp:290-300) states plainly: "The
    # Duff Reduction Equations are no longer used to calculate duff depth
    # reduction... just base the reduction based on the amounts consumed."
    # Concretely, every region/cover-group branch's own depth-reduction
    # equation (Eq 5/6/7/15/etc., dispatched alongside its percent
    # equation) IS still computed, but DUF_Mngr unconditionally DISCARDS
    # it for every non-batch run with one final line (fof_duf.cpp:395):
    #
    #     a_DUF->f_Red = a_CI->f_DufDep * (a_DUF->f_Per / 100.0);
    #
    # This applies identically to EVERY region and fuel category (Piles,
    # Chaparral, InteriorWest, PacificWest, NorthEast, SouthEast) -- there
    # is no per-region depth formula left in real production output. The
    # previous per-region Eq5/Eq6/Eq7/Eq15 branches here were F-23's
    # depth-side defect: they reproduced equations C++ itself no longer
    # uses for this output, diverging for every region whose depth
    # equation differs from the percent-derived value (confirmed for
    # InteriorWest/PacificWest AND NorthEast, including cover groups
    # -- Chaparral, Piles -- that route through the InteriorWest region
    # branch above without being InteriorWest cover types themselves).
    # pdc above is already the FINAL, fully-clamped percent (including the
    # duff_moist<=10 override and the [0,100] clamp), matching what
    # DUF_Mngr's f_Per holds at line 395.
    # ------------------------------------------------------------------
    ddc = None
    rdd = None
    if d_pre is not None:
        d_pre_in = d_pre  # already in inches if units=='SI' was converted above
        ddc = np.clip(d_pre_in * (pdc / 100.0), 0.0, d_pre_in)
        rdd = d_pre_in - ddc

        # Convert back to cm for SI callers
        if units.upper() == 'SI':
            ddc = ddc * 2.54
            rdd = rdd * 2.54

    return {
        'pdc': _maybe_scalar(pdc, scalar_input),
        'ddc': _maybe_scalar(ddc, scalar_input),
        'rdd': _maybe_scalar(rdd, scalar_input),
    }


def consm_herb(
        reg: Union[str, int, np.ndarray],
        cvr_grp: Union[str, int, np.ndarray],
        pre_ll: Union[float, np.ndarray],
        pre_hl: Union[float, np.ndarray],
        season: Union[str, int, np.ndarray, None] = None,
        units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM herbaceous fuel consumption model.

    Accepts scalar or array inputs for all parameters, including *reg*,
    *cvr_grp*, and *season* (strings, integer codes, or arrays thereof).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see
        :data:`CVR_GRP_CODES`).
    :param pre_ll: Pre-fire litter fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param pre_hl: Pre-fire herbaceous fuel load (kg/m² if ``units='SI'``,
        T/acre if ``units='Imperial'``). Scalar or np.ndarray.
    :param season: Burn season (``'Spring'``, ``'Summer'``, ``'Fall'``,
        ``'Winter'``) or integer code (see :data:`SEASON_CODES`). Only
        relevant for GrassGroup: Eq 221 (90%) applies in Spring; all other
        seasons use Eq 22 (100%). Optional; defaults to non-Spring behaviour.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.

    :return: Herbaceous load consumed (same units as input). Scalar ``float``
        when all numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(pre_hl)

    pre_ll = np.ravel(np.asarray(pre_ll, dtype=float))
    pre_hl = np.ravel(np.asarray(pre_hl, dtype=float))
    n = max(len(pre_ll), len(pre_hl))

    if units.upper() == 'SI':
        pre_ll = pre_ll * 4.4609
        pre_hl = pre_hl * 4.4609

    reg_arr  = _to_str_arr(reg, REGION_CODES)
    cvr_arr  = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    sea_arr  = _to_str_arr(season if season is not None else '', SEASON_CODES)
    reg_arr  = np.broadcast_to(reg_arr,  (n,)) if reg_arr.size  == 1 else reg_arr
    cvr_arr  = np.broadcast_to(cvr_arr,  (n,)) if cvr_arr.size  == 1 else cvr_arr
    sea_arr  = np.broadcast_to(sea_arr,  (n,)) if sea_arr.size  == 1 else sea_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd', 'PinFlaWoo')
    _grass_vals    = ('Grass', 'GG', 'GrassGroup')

    is_se         = reg_arr == 'SouthEast'
    is_grass_spr  = np.isin(cvr_arr, _grass_vals) & (sea_arr == 'Spring')
    is_flatwood   = np.isin(cvr_arr, _flatwood_vals)

    hlc = np.select(
        [is_flatwood, is_se, is_grass_spr],
        [
            # Eq 223
            ((pre_hl * 2.24) * 0.9944) / 2.24,
            # Eq 222
            -0.059 + (0.004 * pre_ll) + (0.917 * pre_hl),
            # Eq 221 - 90% in Spring only (fof_hsf.cpp:352-358)
            pre_hl * 0.9,
        ],
        default=pre_hl.copy(),  # Eq 22 – 100%
    )

    hlc = np.clip(hlc, 0.0, pre_hl)

    if units.upper() == 'SI':
        hlc = hlc / 4.4609

    return float(hlc[0]) if scalar_input else hlc


def consm_litter(
        pre_ll: Union[float, np.ndarray],
        l_moist: Union[float, np.ndarray],
        cvr_grp: Union[str, int, np.ndarray, None] = None,
        reg: Union[str, int, np.ndarray, None] = None,
        units: str = 'SI',
        pre_dl: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM litter consumption model (Eqs 997–999, 30).

    Accepts scalar or array inputs for all parameters, including *cvr_grp*
    and *reg* (which may be strings, integer codes, or arrays thereof).

    .. note::
        Most fuel consumption is simulated using Burnup. This function covers
        litter-specific override equations for Flatwoods, Southeast, and
        Coastal Plain (a SouthEast cover group, not a region).

    **Coastal Plain** (``cvr_grp`` ``'CP'`` or ``'CoastPlain'``,
    case-insensitive, matching C++'s ``CI_isCoastPlain()``) bypasses the
    ordinary SouthEast Eq 998 (``* 0.8``) route entirely and uses the same
    litter-first-allocation Eq 30 (C++ ``Equ_CP_Per``,
    ``fof_duf.cpp:1171-1225``) that :func:`consm_duff` uses for its duff
    percent — both derive from the shared :func:`_coastal_plain_forest_floor`
    helper, so calling this function and :func:`consm_duff` with the SAME
    *pre_ll*/*pre_dl*/*l_moist* for a Coastal Plain cell always yields a
    consistent, non-contradictory litter/duff split. Requires *pre_dl*
    (raises ``ValueError`` otherwise); *l_moist* is enforced to C++'s
    inclusive ``[1.0, 100.0]`` range (``fof_ci.h:61-62``, ``_ChkLitMoist()``,
    ``fof_hsf.cpp:614-626``). Coastal Plain is a SouthEast cover group, not a
    region — supplying it with any other *reg* raises ``ValueError`` rather
    than silently falling through to Flatwoods/Eq999 (PyFOFEM's own
    supported contract; C++'s special route is likewise SouthEast-only,
    living entirely inside ``DUF_SouthEast()``, ``fof_duf.cpp:409-426``).

    This litter-consumed value is the SAME one used by the full
    ``run_fofem_emissions()`` facade for Coastal Plain cells — it does not
    additionally run through Burnup's own independent litter handling, so
    litter is never double-counted (mirrors C++'s ``BCM_Mngr``:
    ``if (i_LitEqu == e_CP_PerEq) goto OneHr;``, ``fof_bcm.cpp:99-101``).

    :param pre_ll: Pre-fire litter load (kg/m² if ``units='SI'``, T/acre if
        ``units='Imperial'``). Scalar or np.ndarray.
    :param l_moist: Litter moisture content (%). Scalar or np.ndarray.
    :param cvr_grp: Cover group name or integer code (see
        :data:`CVR_GRP_CODES`). Scalar or np.ndarray. Optional.
    :param reg: Region name or integer code (see :data:`REGION_CODES`).
        Scalar or np.ndarray. Optional.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``.
    :param pre_dl: Pre-fire duff load (same units as *pre_ll*). Required for
        Coastal Plain (Eq 30); ignored otherwise.

    :return: Litter load consumed (kg/m² for ``'SI'``, T/acre for
        ``'Imperial'``). Scalar ``float`` when all numeric inputs are scalars,
        otherwise 1D ``np.ndarray``.
    :raises ValueError: If *cvr_grp* is Coastal Plain and *reg* is not
        ``'SouthEast'``; if *pre_dl* is missing for Coastal Plain; or if
        *l_moist* is outside ``[1.0, 100.0]`` for a Coastal Plain cell.
    """
    scalar_input = _is_scalar(pre_ll) and _is_scalar(l_moist)

    pre_ll = np.ravel(np.asarray(pre_ll, dtype=float))
    l_moist = np.ravel(np.asarray(l_moist, dtype=float))
    n = max(len(pre_ll), len(l_moist))
    pre_ll = np.broadcast_to(pre_ll, (n,)) if pre_ll.size == 1 else pre_ll
    l_moist = np.broadcast_to(l_moist, (n,)) if l_moist.size == 1 else l_moist

    if units.upper() == 'SI':
        pre_ll = pre_ll * 4.4609  # kg/m² → T/acre
        if pre_dl is not None:
            pre_dl = np.ravel(np.asarray(pre_dl, dtype=float)) * 4.4609
    elif pre_dl is not None:
        pre_dl = np.ravel(np.asarray(pre_dl, dtype=float))

    # Resolve categorical strings to broadcast-compatible arrays
    cvr_arr = _to_str_arr(cvr_grp if cvr_grp is not None else '', CVR_GRP_CODES)
    reg_arr = _to_str_arr(reg if reg is not None else '', REGION_CODES)
    # Broadcast to n
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd', 'PinFlaWoo')
    is_flatwood = np.isin(cvr_arr, _flatwood_vals)
    is_southeast = reg_arr == 'SouthEast'
    cvr_lower = np.array([str(v).strip().lower() for v in cvr_arr], dtype=object)
    is_coastplain = np.isin(cvr_lower, tuple(_COASTPLAIN_ALIASES))

    if np.any(is_coastplain & ~is_southeast):
        bad_reg = str(reg_arr[is_coastplain & ~is_southeast][0])
        raise ValueError(
            "consm_litter(): Coastal Plain is only supported for "
            f"reg='SouthEast' (C++'s DUF_SouthEast-only routing, "
            f"fof_duf.cpp:409-426); got reg={bad_reg!r}."
        )

    if np.any(is_coastplain):
        if pre_dl is None:
            raise ValueError(
                "consm_litter(): Coastal Plain requires pre_dl (duff load) "
                "to compute Eq 30 (fof_duf.cpp:1171-1225)."
            )
        pre_dl_bc = np.broadcast_to(pre_dl, (n,)) if np.asarray(pre_dl).size == 1 else pre_dl
        _check_cp_litter_moisture(l_moist[is_coastplain])
        litter_pct_cp, _duff_pct_cp = _coastal_plain_forest_floor(
            pre_ll, pre_dl_bc, l_moist,
        )
        coastplain_consumed_tac = pre_ll * (litter_pct_cp / 100.0)
    else:
        coastplain_consumed_tac = np.zeros(n)

    pre_ll_mgha = pre_ll / T_ACRE_PER_MG_HECTARE
    flatwood_consumed_tac = T_ACRE_PER_MG_HECTARE * np.square(
        0.2871 + (0.9140 * np.sqrt(pre_ll_mgha)) - (0.0101 * l_moist)
    )
    flatwood_consumed_tac = np.minimum(flatwood_consumed_tac, pre_ll)

    llc = np.select(
        [is_flatwood, is_coastplain, is_southeast],
        [
            # Eq 997 operates in Mg/ha before converting back to T/ac.
            flatwood_consumed_tac,
            # Eq 30 (Coastal Plain) — litter-first allocation.
            coastplain_consumed_tac,
            # Eq 998
            pre_ll * 0.8,
        ],
        default=pre_ll.copy(),  # Eq 999
    )

    if units.upper() == 'SI':
        llc = llc / 4.4609  # T/acre → kg/m²

    return float(llc[0]) if scalar_input else llc


def consm_mineral_soil(
        reg: Union[str, int, np.ndarray],
        cvr_grp: Union[str, int, np.ndarray],
        fuel_type: Union[str, int, np.ndarray],
        duff_moist: Union[float, np.ndarray],
        duff_moist_cat: str,
        pile: bool = False,
        pdr: Optional[Union[float, np.ndarray]] = None,
        duff_load: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM mineral soil exposure model.

    Accepts scalar or array inputs, including *reg*, *cvr_grp*, and
    *fuel_type* (strings, integer codes, or arrays thereof).

    **Coastal Plain** (``cvr_grp`` ``'CP'`` or ``'CoastPlain'``,
    case-insensitive) reports equation ID 32 (C++ ``Equ_CP_MSE``,
    ``fof_duf.cpp:1232-1236``): a flat 5% whenever *duff_load* is positive,
    or 100% whenever *duff_load* is ``<= 0`` (C++'s subsequent global
    ``if (f_Duff <= 0) f_MSEPer = 100.0`` override, ``fof_duf.cpp:377-378``,
    which applies regardless of cover group but is exercised here only for
    Coastal Plain). Coastal Plain is a SouthEast cover group, not a region;
    supplying it with any other *reg* raises ``ValueError`` (PyFOFEM's own
    supported contract — see :func:`consm_duff`).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see :data:`CVR_GRP_CODES`).
    :param fuel_type: Fuel type — ``'natural'`` / ``1`` or ``'slash'`` / ``2``
        (see :data:`FUEL_CATEGORY_CODES`).
    :param duff_moist: Duff moisture content (%). Scalar or np.ndarray.
    :param duff_moist_cat: Duff moisture category (``'ldm'``, ``'edm'``,
        ``'nfdth'``, or ``'%dr'``).
    :param pile: ``True`` for pile burning (returns 10%). Default ``False``.
    :param pdr: Percent duff reduction (%), required when
        ``duff_moist_cat='%dr'``. Scalar or np.ndarray.
    :param duff_load: Pre-fire duff load (any consistent mass-load unit).
        Required for Coastal Plain (Eq 32); ignored otherwise.

    :return: Mineral soil exposure (%). Scalar ``float`` when all numeric
        inputs are scalars, otherwise 1D ``np.ndarray``.
    :raises ValueError: If *cvr_grp* is Coastal Plain and *reg* is not
        ``'SouthEast'``, or if *duff_load* is missing for Coastal Plain.
    """
    scalar_input = _is_scalar(duff_moist)

    duff_moist = np.ravel(np.asarray(duff_moist, dtype=float))
    n = len(duff_moist)
    if pdr is not None:
        pdr = np.ravel(np.asarray(pdr, dtype=float))
        pdr = np.broadcast_to(pdr, (n,)) if pdr.size == 1 else pdr
    if duff_load is not None:
        duff_load = np.ravel(np.asarray(duff_load, dtype=float))
        duff_load = np.broadcast_to(duff_load, (n,)) if duff_load.size == 1 else duff_load

    reg_arr = _to_str_arr(reg, REGION_CODES)
    cvr_arr = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    ft_arr  = _to_str_arr(fuel_type, FUEL_CATEGORY_CODES)
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    ft_arr  = np.broadcast_to(ft_arr,  (n,)) if ft_arr.size  == 1 else ft_arr

    ft_lower = np.array([v.lower() for v in ft_arr], dtype=object)
    cvr_lower = np.array([str(v).strip().lower() for v in cvr_arr], dtype=object)
    is_coastplain = np.isin(cvr_lower, tuple(_COASTPLAIN_ALIASES))
    is_southeast = reg_arr == 'SouthEast'

    if np.any(is_coastplain & ~is_southeast):
        bad_reg = str(reg_arr[is_coastplain & ~is_southeast][0])
        raise ValueError(
            "consm_mineral_soil(): Coastal Plain is only supported for "
            f"reg='SouthEast' (C++'s DUF_SouthEast-only routing, "
            f"fof_duf.cpp:409-426); got reg={bad_reg!r}."
        )
    if np.any(is_coastplain) and duff_load is None:
        raise ValueError(
            "consm_mineral_soil(): Coastal Plain requires duff_load to "
            "compute Eq 32 (fof_duf.cpp:1232-1236)."
        )

    mse = np.full(n, np.nan)

    if pile:
        mse[:] = 10.0
    else:
        is_iw_pw    = np.isin(reg_arr, ('InteriorWest', 'PacificWest'))
        is_ne       = reg_arr == 'NorthEast'
        is_pocosin  = np.isin(cvr_arr, ('Pocosin', 'PC'))
        is_white_pine_hemlock = np.isin(cvr_arr, ('White Pine Hemlock', 'WhiPinHem', 'WPH'))
        is_chaparral = np.isin(cvr_arr, ('Chaparral', 'Shrub-Chaparral', 'SGC', 'ShrubGroupChaparral'))
        is_iw_effective = is_iw_pw | (is_ne & is_white_pine_hemlock)
        is_slash    = ft_lower == 'slash'
        is_natural  = ft_lower == 'natural'

        pdr_vals = pdr if pdr is not None else np.zeros(n)
        duff_load_vals = duff_load if duff_load is not None else np.zeros(n)
        moisture_for_nfd = (
            duff_moist / 1.4 if duff_moist_cat == 'adjnfdr' else duff_moist
        )
        is_red_jack = np.isin(cvr_arr, ('Red Jack Pine', 'Red, Jack Pine', 'RedJacPin', 'RJP'))
        is_balsam_spruce = np.isin(
            cvr_arr,
            ('Balsam', 'Black Spruce', 'Red Spruce', 'White Spruce', 'BalBRWSpr', 'Balsam Fir', 'BFS'),
        )
        is_ne_eq14 = is_ne & (
            is_balsam_spruce | (is_red_jack & (duff_moist_cat != 'ldm'))
        )
        is_se_eq14 = (reg_arr == 'SouthEast') & ~is_pocosin & ~is_coastplain
        is_eq14 = is_ne_eq14 | is_se_eq14

        mse = np.select(
            [
                is_chaparral,                                         # Eq 19
                is_iw_effective & is_slash   & (duff_moist_cat == 'ldm'),   # Eq 9
                is_iw_effective & is_natural & (duff_moist_cat == 'ldm'),   # Eq 13
                is_iw_effective & is_slash & np.isin(duff_moist_cat, ('nfdth', 'adjnfdr')), # Eq 11
                is_iw_effective & is_natural & np.isin(duff_moist_cat, ('nfdth', 'adjnfdr')), # Eq 12
                is_iw_effective              & (duff_moist_cat == 'edm'),   # Eq 10
                is_eq14,                                             # Eq 14
                is_ne & is_red_jack & (duff_moist_cat == 'ldm'),    # Eq 10
                is_pocosin,                                           # Eq 202
                is_coastplain,                                        # Eq 32
                ~is_iw_pw & ~is_pocosin & (duff_moist_cat == 'edm'),         # Eq 10
                ~is_iw_pw & ~is_pocosin & (duff_moist_cat == '%dr'),         # Eq 14
            ],
            [
                np.full(n, 100.0),
                np.where(duff_moist <= 135,
                         80 - 0.507 * duff_moist,
                         23.5 - 0.0914 * duff_moist),
                60.4 - 0.440 * duff_moist,
                93.3 - 3.55  * moisture_for_nfd,
                94.3 - 4.96  * moisture_for_nfd,
                167.4 - 31.6 * np.log(duff_moist),
                -8.98 + 0.44 * pdr_vals,
                167.4 - 31.6 * np.log(duff_moist),
                np.zeros(n),
                np.where(duff_load_vals > 0, 5.0, 100.0),
                167.4 - 31.6 * np.log(duff_moist),
                -8.98 + 0.899 * pdr_vals,
            ],
            default=np.full(n, np.nan),
        )

        mse = np.clip(mse, 0.0, 100.0)
        if duff_load is not None:
            mse = np.where(duff_load_vals <= 0.0, 100.0, mse)

    return float(mse[0]) if scalar_input else mse


def consm_shrub(
        reg: Union[str, int, np.ndarray],
        cvr_grp: Union[str, int, np.ndarray],
        pre_sl: Union[float, np.ndarray],
        season: Union[str, int, np.ndarray, None] = None,
        pre_ll: Optional[Union[float, np.ndarray]] = None,
        pre_dl: Optional[Union[float, np.ndarray]] = None,
        pre_rl: Optional[Union[float, np.ndarray]] = None,
        pre_dw1: Optional[Union[float, np.ndarray]] = None,
        pre_dw10: Optional[Union[float, np.ndarray]] = None,
        duff_moist: Optional[Union[float, np.ndarray]] = None,
        llc: Optional[Union[float, np.ndarray]] = None,
        ddc: Optional[Union[float, np.ndarray]] = None,
        units: str = 'SI',
) -> Union[float, np.ndarray]:
    """
    FOFEM shrub fuel consumption model.

    Accepts scalar or array inputs, including *reg*, *cvr_grp*, and *season*
    (strings, integer codes, or arrays thereof).

    :param reg: Region name or integer code (see :data:`REGION_CODES`).
    :param cvr_grp: Cover group name or integer code (see :data:`CVR_GRP_CODES`).
    :param pre_sl: Pre-fire shrub fuel load. Scalar or np.ndarray.
    :param season: Burn season or integer code (see :data:`SEASON_CODES`).
    :param pre_ll: Pre-fire litter load (SE non-Pocosin Eq 16/234). Optional.
    :param pre_dl: Pre-fire duff load (SE non-Pocosin Eq 16/234). Optional.
    :param pre_rl: Pre-fire regeneration load (SE non-Pocosin Eq 234). Optional.
    :param pre_dw1: Pre-fire 1-hr dead woody load (SE non-Pocosin Eq 16/234).
        Optional; omitting it is equivalent to passing 0, matching the
        function's behavior before this parameter existed.
    :param pre_dw10: Pre-fire 10-hr dead woody load (SE non-Pocosin Eq 16/234).
        Optional; omitting it is equivalent to passing 0, matching the
        function's behavior before this parameter existed.
    :param duff_moist: Duff moisture content (%). Optional.
    :param llc: Litter load consumed (SE non-Pocosin Eq 234). Optional.
    :param ddc: Duff depth consumed (SE non-Pocosin Eq 234). Optional.
    :param units: Unit system. ``'SI'`` (default) or ``'Imperial'``. All of
        *pre_sl*, *pre_ll*, *pre_dl*, *pre_rl*, *pre_dw1*, and *pre_dw10* are
        read in this same system: kg/m2 for ``'SI'`` (converted internally to
        tons/acre), or tons/acre directly for ``'Imperial'``. The returned
        percent-consumed value is never unit-converted.

    :return: Percent shrub load consumed (%). Scalar ``float`` when all
        numeric inputs are scalars, otherwise 1D ``np.ndarray``.
    """
    scalar_input = _is_scalar(pre_sl)

    pre_sl = np.ravel(np.asarray(pre_sl, dtype=float))
    n = len(pre_sl)

    if units.upper() == 'SI':
        pre_sl = pre_sl * 4.4609
        if pre_ll is not None:
            pre_ll = np.ravel(np.asarray(pre_ll, dtype=float)) * 4.4609
        if pre_dl is not None:
            pre_dl = np.ravel(np.asarray(pre_dl, dtype=float)) * 4.4609
        if pre_rl is not None:
            pre_rl = np.ravel(np.asarray(pre_rl, dtype=float)) * 4.4609
        if pre_dw1 is not None:
            pre_dw1 = np.ravel(np.asarray(pre_dw1, dtype=float)) * 4.4609
        if pre_dw10 is not None:
            pre_dw10 = np.ravel(np.asarray(pre_dw10, dtype=float)) * 4.4609
    else:
        if pre_ll is not None:
            pre_ll = np.ravel(np.asarray(pre_ll, dtype=float))
        if pre_dl is not None:
            pre_dl = np.ravel(np.asarray(pre_dl, dtype=float))
        if pre_rl is not None:
            pre_rl = np.ravel(np.asarray(pre_rl, dtype=float))
        if pre_dw1 is not None:
            pre_dw1 = np.ravel(np.asarray(pre_dw1, dtype=float))
        if pre_dw10 is not None:
            pre_dw10 = np.ravel(np.asarray(pre_dw10, dtype=float))

    if duff_moist is not None:
        duff_moist = np.ravel(np.asarray(duff_moist, dtype=float))
    if llc is not None:
        llc = np.ravel(np.asarray(llc, dtype=float))
    if ddc is not None:
        ddc = np.ravel(np.asarray(ddc, dtype=float))

    reg_arr = _to_str_arr(reg, REGION_CODES)
    cvr_arr = _to_str_arr(cvr_grp, CVR_GRP_CODES)
    sea_arr = _to_str_arr(season if season is not None else '', SEASON_CODES)
    reg_arr = np.broadcast_to(reg_arr, (n,)) if reg_arr.size == 1 else reg_arr
    cvr_arr = np.broadcast_to(cvr_arr, (n,)) if cvr_arr.size == 1 else cvr_arr
    sea_arr = np.broadcast_to(sea_arr, (n,)) if sea_arr.size == 1 else sea_arr

    _flatwood_vals = ('Flatwood', 'Pine Flatwoods', 'PFL', 'PinFltwd', 'PinFlaWoo')

    is_se       = reg_arr == 'SouthEast'
    is_pocosin  = np.isin(cvr_arr, ('Pocosin', 'PC'))
    is_sage     = np.isin(cvr_arr, ('Sagebrush', 'SB'))
    is_flatwood = np.isin(cvr_arr, _flatwood_vals)
    is_shrubgrp = np.isin(cvr_arr, ('Shrub', 'SG', 'ShrubGroup'))

    # SE Pocosin seasonal
    sea_spr_win = np.isin(sea_arr, ('Spring', 'Winter'))
    sea_sum_fal = np.isin(sea_arr, ('Summer', 'Fall'))
    sea_fall    = sea_arr == 'Fall'
    sea_spr_sum = np.isin(sea_arr, ('Spring', 'Summer'))

    # Eq 234 – SE non-Pocosin.  The direct C++ path (fof_hsf.cpp:229-274,
    # Equation_16 / Equ_234_Per) first derives f_W = Equation_16(a_CI) using
    # f_WPRE = f_Lit + f_Duff + f_DW10 + f_DW1 (litter + duff + 10-hr + 1-hr
    # dead woody fuel), then Equ_234_Per reuses the same 4-term f_WPRE to
    # compute a fraction.  Calc_Shrub (fof_hsf.cpp:166-209) multiplies that
    # fraction directly by f_Shrub to get the consumed load -- the fraction
    # is not itself a 0-100 percent despite the `if (f > 100) f = 100;`
    # clamp bound (fof_hsf.cpp:253-255); this quirk is preserved unchanged.
    # C++ returns 0 when f_W == 0 (fof_hsf.cpp:234-235), f_WPRE == 0
    # (fof_hsf.cpp:238,271), and again when f_ShrReg (= f_Shrub) == 0
    # (fof_hsf.cpp:243); Calc_Shrub's own
    # `if (f_Shrub != 0) {...} else *af_Percent = 0;` (fof_hsf.cpp:182-186)
    # applies the same zero-shrub guard a second time. *pre_dw1*/*pre_dw10*
    # default to 0 when omitted, so omitting them is exactly equivalent to
    # passing 0 and existing callers are unaffected.
    if all(x is not None for x in (pre_ll, pre_dl, duff_moist)):
        pre_dw1_term = pre_dw1 if pre_dw1 is not None else 0.0
        pre_dw10_term = pre_dw10 if pre_dw10 is not None else 0.0
        woody_pre = pre_ll + pre_dl + pre_dw10_term + pre_dw1_term
        woody_pre_zero = woody_pre == 0
        woody_pre_safe = np.where(woody_pre > 0, woody_pre, np.nan)
        fire_weight = (
            3.4958 + (0.3833 * woody_pre) - (0.0237 * duff_moist) -
            (5.6075 / woody_pre_safe)
        )
        fire_weight_zero = fire_weight == 0
        shrub_zero = pre_sl == 0
        shrub_safe = np.where(pre_sl > 0, pre_sl, np.nan)
        eq234_fraction = (
            (3.2484 + (0.4322 * woody_pre) + (0.6765 * pre_sl) -
             (0.0276 * duff_moist) - (5.0796 / woody_pre_safe) - fire_weight) /
            shrub_safe
        )
        # Exact-zero fuel-load guards, matching C++'s `== 0` checks. Any
        # other invalid (e.g. negative) input is left to propagate as NaN
        # rather than being silently zeroed.
        eq234_fraction = np.where(
            woody_pre_zero | fire_weight_zero | shrub_zero, 0.0,
            eq234_fraction,
        )
        eq234_load = pre_sl * np.clip(eq234_fraction, 0.0, 100.0)
    else:
        eq234_load = np.full(n, np.nan)

    # Eq 236 – Flatwood
    season_flag = np.where(sea_spr_sum, 1.0, 0.0)
    pre_sl_mgha = pre_sl / T_ACRE_PER_MG_HECTARE
    eq236_load_tac = T_ACRE_PER_MG_HECTARE * np.exp(
        -0.1889 + (0.9049 * np.log(np.maximum(pre_sl_mgha, 1e-12))) +
        (0.0676 * season_flag)
    )
    eq236_load_tac = np.minimum(eq236_load_tac, pre_sl)
    consumed_load = np.select(
        [
            is_sage & sea_fall,                      # Eq 233
            is_sage & ~sea_fall,                     # Eq 232
            is_flatwood,                             # Eq 236
            is_shrubgrp,                             # Eq 231
            is_se & is_pocosin & sea_spr_win,        # Eq 233
            is_se & is_pocosin & sea_sum_fal,        # Eq 235
            is_se & ~is_pocosin,                     # Eq 234
        ],
        [
            pre_sl * 0.9,
            pre_sl * 0.5,
            eq236_load_tac,
            pre_sl * 0.8,
            pre_sl * 0.9,
            pre_sl * 0.8,
            eq234_load,
        ],
        default=pre_sl * 0.6,  # Eq 23
    )

    consumed_load = np.clip(consumed_load, 0.0, pre_sl)
    slc = np.zeros_like(pre_sl)
    np.divide(100.0 * consumed_load, pre_sl, out=slc, where=pre_sl > 0)

    return float(slc[0]) if scalar_input else slc


# Named moisture regimes, used only by get_moisture_regime().
_MOISTURE_REGIMES: Dict[str, Dict[str, float]] = {
    'wet':      {'duff': 130.0, '10hr': 22.0, '3plus': 40.0, 'soil': 25.0},
    'moderate': {'duff':  75.0, '10hr': 16.0, '3plus': 30.0, 'soil': 15.0},
    'dry':      {'duff':  40.0, '10hr': 10.0, '3plus': 15.0, 'soil': 10.0},
    'very dry': {'duff':  20.0, '10hr':  6.0, '3plus': 10.0, 'soil':  5.0},
}


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
    | Very dry |  20 %   |  6 %    | 10 %     |  5 %    |
    +----------+---------+---------+----------+---------+

    :param regime: One of ``'wet'``, ``'moderate'``, ``'dry'``, or ``'very dry'``
        (case-insensitive).
    :return: Dict with keys ``'duff'``, ``'10hr'``, ``'3plus'``, ``'soil'``
        and float values in percent.
    :raises KeyError: If *regime* is not one of the four recognised values.
    """
    key = regime.strip().lower()
    if key not in _MOISTURE_REGIMES:
        raise KeyError(
            f"Unknown moisture regime '{regime}'. "
            f"Valid options: {list(_MOISTURE_REGIMES.keys())}"
        )
    return dict(_MOISTURE_REGIMES[key])


# Removed burnup-related constants and functions (now in burnup_calcs.py)
