# -*- coding: utf-8 -*-
"""
emission_calcs.py – FOFEM smoke emissions calculations.

Provides:

* ``calc_smoke_emissions`` – compute smoke-emission mass per unit area from
  fuel consumption totals using bundled emission-factor CSV data.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import os
import numpy as np
from pandas import read_csv, DataFrame
from typing import Dict, Optional, Union


# ---------------------------------------------------------------------------
# Smoke Emissions
# ---------------------------------------------------------------------------

# Path to the bundled emissions-factors CSV (relative to this module).
_EF_CSV_DEFAULT = os.path.join(
    os.path.dirname(__file__), '..', 'supporting_data', 'emissions_factors.csv',
)

# Emission_Factors.csv group indices (1-based row in the CSV data section)
_EF_GROUP_DEFAULT = 3          # Western Forest-Rx (FOFEM default)
_EF_SMOLDERING_GROUP_DEFAULT = 7  # CWDRSC — coarse woody smoldering
_EF_DUFF_GROUP_DEFAULT = 8        # DuffRSC — duff smoldering

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
    # Resolve total loads — supports scalars, arrays, or dicts
    def _total(load):
        if isinstance(load, dict):
            vals = list(load.values())
            return np.sum(np.stack([np.asarray(v, dtype=float) for v in vals], axis=0), axis=0)
        return np.asarray(load, dtype=float)

    f_kg = _total(flaming_load)
    s_kg = _total(smoldering_load)
    d_kg = np.asarray(duff_load, dtype=float)

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

        # Three sets of emission factors matching C++ BRN_SetEmis architecture
        row_f = ef_group - 1              # STFS — flaming
        row_s = ef_smoldering_group - 1   # CWDRSC — coarse-wood smoldering
        row_d = ef_duff_group - 1         # DuffRSC — duff smoldering

        # Separate duff from coarse smoldering
        coarse_smo = np.maximum(s_kg - d_kg, 0.0)

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
            # Duff-only emission outputs
            'PM10S_Duff': e_duf['PM10'],
            'PM25S_Duff': e_duf['PM25'],
            'CH4S_Duff':  e_duf['CH4'],
            'COS_Duff':   e_duf['CO'],
            'CO2S_Duff':  e_duf['CO2'],
            'NOXS_Duff':  e_duf['NOX'],
            'SO2S_Duff':  e_duf['SO2'],
        }
        return result

