#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_public_constants.py - Phase 3 coverage for all **11** non-function
exports of the supported public API, enumerated by the authoritative
Gate 0 inventory (``development/plans/gate0/02-api-inventory.md`` §2):
``SPP_CODES``, ``DEFAULT_CONSUMPTION_VARS``,
``EXPANDED_CONSUMPTION_VARS``, ``TOTAL_DURATION_CONSUMED_VARS``,
``SOIL_HEAT_VARS``, ``EQUATION_VARS``, ``ERROR_VARS``,
``REGION_CODES``, ``CVR_GRP_CODES``, ``SEASON_CODES`` and
``FUEL_CATEGORY_CODES``.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):

(a) *Python contract/equation test* - **every test in this module.**
(b) *Source-relation cross-check* - **none here.** Gate 0
    ``06-runtime-tables.md`` §4 records the ``SPP_CODES`` vs C++
    ``sr_MSMT[]`` row-coverage comparison as an explicitly open
    **Phase 4** work item; it is not attempted or claimed here.
(c) *Executable C++ parity* - **none here.**

**What each export is covered for**, per the phase brief: availability
from the supported public namespace, runtime type and structural shape,
stable keys/categories, scientifically meaningful sentinel content
(real values checkable against the shipped table or the documented
code domain), and aliasing/mutation behaviour.

**Aliasing, stated precisely.** None of the 11 exports is defensively
copied. Each is the *same object* reached through all supported import
paths (``pyfofem`` -> ``pyfofem.pyfofem`` -> ``pyfofem.components`` ->
the defining module), so a caller that mutates one corrupts the table
for every later caller in the process. The tests below pin that
identity rather than mutating shared state to demonstrate it, and this
docstring records the consequence. The one place where the API *does*
promise a defensive copy is ``get_moisture_regime``, whose copy is
verified directly in ``tests/unit/test_utility_contracts.py``.

Full-object snapshot assertions are used **only** for the six output
variable lists and the four small code dictionaries, where exact
membership and order *are* the contract (an output-key list whose order
changed would silently reorder every consumer's columns). ``SPP_CODES``
is deliberately **not** snapshotted: it is a 121-row scientific table
whose per-row content is covered structurally and by spot-checked
sentinels in this module, and by schema/provenance tests in
``tests/unit/test_runtime_data_resources.py``.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pyfofem
import pyfofem.components as components
import pyfofem.pyfofem as pyfofem_module
from pyfofem.components import consumption_calcs, tree_flame_calcs

#: The 11 non-function exports, from ``gate0/02-api-inventory.md`` §2.
_CONSTANT_EXPORT_NAMES = [
    "CVR_GRP_CODES",
    "DEFAULT_CONSUMPTION_VARS",
    "EQUATION_VARS",
    "ERROR_VARS",
    "EXPANDED_CONSUMPTION_VARS",
    "FUEL_CATEGORY_CODES",
    "REGION_CODES",
    "SEASON_CODES",
    "SOIL_HEAT_VARS",
    "SPP_CODES",
    "TOTAL_DURATION_CONSUMED_VARS",
]

#: The 11 fuel components that carry a pre/consumed/post triplet in
#: ``DEFAULT_CONSUMPTION_VARS``, in their documented output order.
_COMPONENT_PREFIXES = [
    "Lit",
    "DW1",
    "DW10",
    "DW100",
    "DW1kSnd",
    "DW1kRot",
    "Duf",
    "Her",
    "Shr",
    "Fol",
    "Bra",
]

#: The seven emission species carried by ``DEFAULT_CONSUMPTION_VARS``
#: (flaming ``F`` / smouldering ``S``) and, with an ``S_Duff`` suffix, by
#: ``EXPANDED_CONSUMPTION_VARS``.
_EMISSION_SPECIES = ["PM10", "PM25", "CH4", "CO", "CO2", "NOX", "SO2"]

#: Exact expected membership and order of the six output variable lists.
_EXPECTED_LISTS = {
    "DEFAULT_CONSUMPTION_VARS": [
        "LitPre", "LitCon", "LitPos",
        "DW1Pre", "DW1Con", "DW1Pos",
        "DW10Pre", "DW10Con", "DW10Pos",
        "DW100Pre", "DW100Con", "DW100Pos",
        "DW1kSndPre", "DW1kSndCon", "DW1kSndPos",
        "DW1kRotPre", "DW1kRotCon", "DW1kRotPos",
        "DufPre", "DufCon", "DufPos",
        "HerPre", "HerCon", "HerPos",
        "ShrPre", "ShrCon", "ShrPos",
        "FolPre", "FolCon", "FolPos",
        "BraPre", "BraCon", "BraPos",
        "MSE", "DufDepPre", "DufDepCon", "DufDepPos",
        "PM10F", "PM10S", "PM25F", "PM25S", "CH4F", "CH4S",
        "COF", "COS", "CO2F", "CO2S", "NOXF", "NOXS", "SO2F", "SO2S",
    ],
    "EXPANDED_CONSUMPTION_VARS": [
        "PM10S_Duff", "PM25S_Duff", "CH4S_Duff",
        "COS_Duff", "CO2S_Duff", "NOXS_Duff", "SO2S_Duff",
    ],
    "TOTAL_DURATION_CONSUMED_VARS": ["FlaDur", "SmoDur", "FlaCon", "SmoCon"],
    "SOIL_HEAT_VARS": ["Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"],
    "EQUATION_VARS": [
        "Lit-Equ", "DufCon-Equ", "DufRed-Equ", "MSE-Equ", "Herb-Equ", "Shrub-Equ",
    ],
    "ERROR_VARS": ["BurnupLimitAdj", "BurnupError"],
}

#: Exact expected contents of the four categorical code dictionaries.
_EXPECTED_MAPS = {
    "REGION_CODES": {
        1: "InteriorWest",
        2: "PacificWest",
        3: "NorthEast",
        4: "SouthEast",
    },
    "SEASON_CODES": {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"},
    "FUEL_CATEGORY_CODES": {1: "Natural", 2: "Slash"},
    "CVR_GRP_CODES": {
        0: "",
        1: "Ponderosa pine",
        2: "Pocosin",
        3: "Chaparral",
        4: "Shrub-Chaparral",
        5: "Sagebrush",
        6: "Flatwood",
        7: "Pine Flatwoods",
        8: "Red Jack Pine",
        9: "Red, Jack Pine",
        10: "Grass",
        11: "Shrub",
        12: "PN",
        13: "PC",
        14: "SGC",
        15: "ShrubGroupChaparral",
        16: "SB",
        17: "PFL",
        18: "PinFltwd",
        19: "RJP",
        20: "RedJacPin",
        21: "GG",
        22: "GrassGroup",
        23: "SG",
        24: "ShrubGroup",
    },
}

#: Spot-checked ``SPP_CODES`` rows: ``species_cd -> (fofem_cd,
#: tree_type, num_cd)``. Chosen for scientific checkability - three
#: Douglas-fir codes that must all resolve to the FOFEM code ``PSME``,
#: a ponderosa pine, a lodgepole pine, a quaking aspen, and the three
#: catch-all sentinels (997 other-softwood, 998 other-hardwood, 999
#: unknown) that bound the numeric code domain.
_SPP_SENTINELS = {
    "FD": ("PSME", "conifer", 137),
    "FDC": ("PSME", "conifer", 138),
    "FDI": ("PSME", "conifer", 139),
    "PY": ("PIPO", "conifer", 178),
    "PL": ("PICO", "conifer", 169),
    "AT": ("POTRA", "hardwood", 104),
    "ZC": ("OTHSOFT", "conifer", 997),
    "ZH": ("OTHHARD", "hardwood", 998),
    "UNK": ("UNK", "conifer", 999),
}


def _export_aliases(name: str):
    """
    Collect the same-named export from every supported import path.

    :param name: Export name from :data:`_CONSTANT_EXPORT_NAMES`.
    :return: List of the objects reached through ``pyfofem``,
        ``pyfofem.pyfofem``, ``pyfofem.components`` and the defining
        module, skipping any namespace that does not re-export the name.
    """
    defining = tree_flame_calcs if name == "SPP_CODES" else consumption_calcs
    namespaces = (pyfofem, pyfofem_module, components, defining)
    return [getattr(ns, name) for ns in namespaces if hasattr(ns, name)]


def test_all_eleven_constants_are_exported_and_declared():
    """
    All 11 non-function exports must be importable from the top-level
    ``pyfofem`` namespace and declared in its ``__all__``, and
    ``__all__`` must contain exactly 24 functions plus these 11 entries
    (the 24/11 split Gate 0 confirmed).

    :return: None. Raises via ``assert`` on mismatch.
    """
    declared = set(pyfofem.__all__)
    assert len(pyfofem.__all__) == len(declared) == 35

    for name in _CONSTANT_EXPORT_NAMES:
        assert name in declared, f"{name} missing from pyfofem.__all__"
        assert hasattr(pyfofem, name), f"{name} not importable from pyfofem"

    non_functions = {
        name for name in declared if not callable(getattr(pyfofem, name))
    }
    assert non_functions == set(_CONSTANT_EXPORT_NAMES)
    assert len(declared - non_functions) == 24


@pytest.mark.parametrize("name", sorted(_EXPECTED_MAPS))
def test_code_maps_have_exact_contents(name):
    """
    Each categorical code map must be a ``dict`` with exactly the
    documented integer-code -> label pairs, in insertion order.

    Exact snapshots are used deliberately here: these maps define the
    integer codes callers pass to every ``consm_*`` function, so a
    changed or reordered entry is a breaking API change, not an
    implementation detail.

    :param name: One of the four code-map export names.
    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = getattr(pyfofem, name)
    expected = _EXPECTED_MAPS[name]
    assert isinstance(actual, dict)
    assert actual == expected
    assert list(actual) == list(expected)
    assert all(isinstance(code, int) for code in actual)
    assert all(isinstance(label, str) for label in actual.values())


def test_cover_group_codes_carry_the_documented_alias_block():
    """
    ``CVR_GRP_CODES`` holds 25 entries but far fewer distinct cover
    groups: codes 12-24 are short-form alias spellings of codes 1-11
    (``gate0/02-api-inventory.md`` §2, note on ``CVR_GRP_CODES``).

    Pin the split explicitly - codes 0-11 are the canonical block and
    12-24 the alias block - so a future edit cannot quietly move an
    alias into the canonical range or add a 26th entry unnoticed. Every
    label must also be unique, since the consumption functions match on
    the string value.

    :return: None. Raises via ``assert`` on mismatch.
    """
    codes = pyfofem.CVR_GRP_CODES
    assert len(codes) == 25
    assert sorted(codes) == list(range(25))

    canonical = {code: label for code, label in codes.items() if code <= 11}
    aliases = {code: label for code, label in codes.items() if code >= 12}
    assert len(canonical) == 12
    assert len(aliases) == 13

    labels = list(codes.values())
    assert len(set(labels)) == len(labels)

    assert codes[0] == ""
    assert codes[12] == "PN"
    assert codes[24] == "ShrubGroup"


def test_default_consumption_vars_structure_is_derivable_from_its_parts():
    """
    ``DEFAULT_CONSUMPTION_VARS``'s 51 entries decompose exactly into
    three documented blocks, which is stronger evidence of correctness
    than the flat snapshot alone:

    * 33 = 11 fuel components x ``Pre``/``Con``/``Pos``
    * 4 = ``MSE`` plus the duff-depth triplet
    * 14 = 7 emission species x flaming (``F``) / smouldering (``S``)

    :return: None. Raises via ``assert`` on mismatch.
    """
    variables = pyfofem.DEFAULT_CONSUMPTION_VARS
    assert len(variables) == 51

    component_block = [
        f"{prefix}{suffix}"
        for prefix in _COMPONENT_PREFIXES
        for suffix in ("Pre", "Con", "Pos")
    ]
    depth_block = ["MSE", "DufDepPre", "DufDepCon", "DufDepPos"]
    emission_block = [
        f"{species}{phase}" for species in _EMISSION_SPECIES for phase in ("F", "S")
    ]

    assert len(component_block) == 33
    assert len(depth_block) == 4
    assert len(emission_block) == 14
    assert variables == component_block + depth_block + emission_block


def test_expanded_consumption_vars_are_the_duff_smouldering_species():
    """
    ``EXPANDED_CONSUMPTION_VARS`` must be exactly the seven emission
    species carrying the smouldering-duff suffix, in the same species
    order used by ``DEFAULT_CONSUMPTION_VARS``, and must not overlap it.

    :return: None. Raises via ``assert`` on mismatch.
    """
    expanded = pyfofem.EXPANDED_CONSUMPTION_VARS
    assert expanded == [f"{species}S_Duff" for species in _EMISSION_SPECIES]
    assert not set(expanded) & set(pyfofem.DEFAULT_CONSUMPTION_VARS)


@pytest.mark.parametrize("name", sorted(_CONSTANT_EXPORT_NAMES))
def test_exports_are_the_same_object_through_every_import_path(name):
    """
    Each export must be the *same object* through ``pyfofem``,
    ``pyfofem.pyfofem``, ``pyfofem.components`` and its defining module
    - the two-re-export-hop pattern must not clone the table.

    This pins the aliasing contract: because there is no defensive copy,
    a caller that mutates any of these objects corrupts it for every
    later caller in the process. No test in this module mutates them.

    :param name: Export name from :data:`_CONSTANT_EXPORT_NAMES`.
    :return: None. Raises via ``assert`` on mismatch.
    """
    aliases = _export_aliases(name)
    assert len(aliases) >= 3, f"{name} not reachable from enough namespaces"
    first = aliases[0]
    for other in aliases[1:]:
        assert other is first


@pytest.mark.parametrize("name", sorted(_EXPECTED_LISTS))
def test_output_variable_lists_have_exact_membership_and_order(name):
    """
    Each output-variable list must be a ``list`` of unique ``str`` with
    exactly the documented membership *and order*.

    Order is part of the contract: these lists drive output-column
    construction, so a reordering silently permutes every consumer's
    columns. ``EQUATION_VARS`` also pins the post-PR#1 ``'Shrub-Equ'``
    spelling (the pre-PR#1 code emitted ``'Shurb-Equ'``).

    :param name: One of the six output-variable list export names.
    :return: None. Raises via ``assert`` on mismatch.
    """
    actual = getattr(pyfofem, name)
    expected = _EXPECTED_LISTS[name]
    assert isinstance(actual, list)
    assert all(isinstance(entry, str) for entry in actual)
    assert len(set(actual)) == len(actual)
    assert actual == expected


def test_spp_codes_has_no_duplicate_or_out_of_domain_identifiers():
    """
    ``SPP_CODES``'s identifying key ``species_cd`` must be unique, its
    numeric code ``num_cd`` must be unique and integral, and
    ``tree_type`` must stay inside its closed three-value domain.

    ``fofem_cd`` is deliberately **not** required to be unique - it is
    the many-to-one FOFEM code that several regional species codes map
    onto (e.g. ``FD``/``FDC``/``FDI`` all resolve to ``PSME``). The
    exact duplicate count is pinned so a change in that mapping is
    visible.

    :return: None. Raises via ``assert`` on mismatch.
    """
    table = pyfofem.SPP_CODES
    assert table["species_cd"].is_unique
    assert table["num_cd"].is_unique
    assert pd.api.types.is_integer_dtype(table["num_cd"])
    assert set(table["tree_type"].unique()) == {"conifer", "hardwood", "shrub"}
    assert int(table["fofem_cd"].duplicated().sum()) == 22


@pytest.mark.parametrize(
    ("species_cd", "expected"),
    [(code, values) for code, values in sorted(_SPP_SENTINELS.items())],
)
def test_spp_codes_sentinel_rows_carry_their_real_values(species_cd, expected):
    """
    Spot-check real, verifiable ``SPP_CODES`` rows rather than asserting
    the table is merely non-empty.

    The three Douglas-fir regional codes must all resolve to the single
    FOFEM code ``PSME``, and the three catch-all rows must hold the
    numeric sentinels 997 (other softwood), 998 (other hardwood) and
    999 (unknown) that bound the code domain.

    :param species_cd: Regional species code used as the lookup key.
    :param expected: ``(fofem_cd, tree_type, num_cd)`` triple.
    :return: None. Raises via ``assert`` on mismatch.
    """
    table = pyfofem.SPP_CODES
    rows = table.loc[table["species_cd"] == species_cd]
    assert len(rows) == 1, f"expected exactly one row for {species_cd}"
    row = rows.iloc[0]
    assert (row["fofem_cd"], row["tree_type"], int(row["num_cd"])) == expected


def test_spp_codes_shape_and_columns():
    """
    ``SPP_CODES`` must be a 121-row, 4-column ``DataFrame`` with the
    documented column names in the documented order.

    :return: None. Raises via ``assert`` on mismatch.
    """
    table = pyfofem.SPP_CODES
    assert isinstance(table, pd.DataFrame)
    assert table.shape == (121, 4)
    assert list(table.columns) == ["species_cd", "fofem_cd", "tree_type", "num_cd"]
