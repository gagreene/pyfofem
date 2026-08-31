#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_runtime_data_resources.py - Phase 3 schema / provenance /
semantic-content / installed-resource coverage for **both** runtime
scientific tables pyfofem actually reads, enumerated by
``development/plans/gate0/06-runtime-tables.md`` §1:

* ``src/pyfofem/supporting_data/species_codes_lut.csv`` - read at
  **import time** by ``components/tree_flame_calcs.py`` into
  ``SPP_CODES``, so a missing or malformed file breaks
  ``import pyfofem`` outright, not just one function.
* ``src/pyfofem/supporting_data/emissions_factors.csv`` - read lazily
  and cached by ``components/emission_calcs.py::_load_ef_csv``.

Verified from the loader source (not assumed): those are the only two
data files any pyfofem module opens at runtime.
``supporting_data/FOFEM6.7/`` is a bundled vendor distribution that no
pyfofem code reads and that the packaging config deliberately excludes
from the wheel (``gate0/06-runtime-tables.md`` §3); the wheel-exclusion
half of that is asserted here, and the tracked ``FOF_GUI.exe`` and
Microsoft DLLs are otherwise out of scope for this phase.

**Test-category classification** (see the phase plan
``development/plans/2026-08-26-comprehensive-test-suite-plan.md``):

(a) *Python contract/equation test* - the schema, uniqueness, domain,
    packaging-config, resource-resolution and provenance-digest tests.
(b) *Source-relation cross-check* - the emission-factor table's
    provenance digest is the same SHA-256 as the pinned C++
    ``FOF_UNIX/Emission_Factors.csv`` (see below), so pinning the digest
    pins byte-identity with the C++ table transitively.
(c) *Executable C++ parity* - **none in this module.** Nothing here
    builds or runs C++.

**Emission-factor provenance, verified directly this phase.** At C++ SHA
``78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`` the pinned
``reference/fofem_cpp/FOF_UNIX/Emission_Factors.csv`` (the table
``NES_Read`` loads) hashes to SHA-256
``4DEC3F4D0AFBA3859F7D5AEF8A3E3E27794C2E5BDD8799A9D94071D3C6B1A640``,
byte-identical to the packaged ``emissions_factors.csv`` and to the
bundled ``supporting_data/FOFEM6.7/Emission_Factors.csv``. This
reproduces the byte-identity Gate 0 recorded in
``06-runtime-tables.md`` §1a. The digest is asserted here as a
self-contained constant rather than by re-reading the C++ submodule, so
the check stays valid in an installed-only run and in a checkout whose
submodule has not been initialised.

**Resource-resolution mechanism, stated accurately.** Neither loader
uses ``importlib.resources``; both build a path from the defining
module's own ``__file__``
(``os.path.join(os.path.dirname(__file__), '..', 'supporting_data',
...)``). That is *package*-relative, not repo-relative, so it resolves
correctly from an installed wheel and does not depend on the process's
working directory - which is what the tests below assert, together with
a real child-process check run from an unrelated working directory. See
this module's findings note in ``docs/CODEBASE.md`` for the one
scenario the ``__file__`` approach does not cover (a zipimported
package, where ``importlib.resources`` would be required).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sys

import pandas as pd
import pytest

import pyfofem
from pyfofem.components import emission_calcs, tree_flame_calcs
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._proc import run_bounded

#: SHA-256 of the packaged ``emissions_factors.csv``. Identical to the
#: pinned C++ ``FOF_UNIX/Emission_Factors.csv`` at SHA
#: ``78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`` - see the module
#: docstring.
_EF_CSV_SHA256 = "4DEC3F4D0AFBA3859F7D5AEF8A3E3E27794C2E5BDD8799A9D94071D3C6B1A640"

#: The eight real emission-factor groups: ``Group # -> (cover type,
#: Type)``. Verified against the shipped file and recorded in
#: ``gate0/06-runtime-tables.md`` §1.
_EF_GROUPS = {
    1: ("Southeastern Forest", "STFS"),
    2: ("Boreal Forest", "STFS"),
    3: ("Western Forest - Rx", "STFS"),
    4: ("Western Forest - WF", "STFS"),
    5: ("Shrubland", "STFS"),
    6: ("Grassland", "STFS"),
    7: ("Woody RSC", "CWDRSC"),
    8: ("Duff RSC", "DuffRSC"),
}

#: Number of rows ``_load_ef_csv``'s ``skiprows=1, header=0`` parse
#: actually yields: the 8 real factor rows plus 9 trailing non-factor
#: rows (one cover-type section header and eight cover labels). Pinned
#: because it is the mechanical cause of Gate 0 finding **F-31**.
_EF_PARSED_ROWS = 17

#: Physical line count of the raw emission-factor CSV: 2 header lines,
#: 8 factor rows, 1 blank line, 9 trailing non-factor lines.
_EF_RAW_LINES = 20

#: Timeout (seconds) for the one child-process resource-resolution
#: check. Generous relative to a bare ``import pyfofem`` so a loaded
#: machine cannot make it flaky, but bounded so it can never hang the
#: suite.
_RESOURCE_PROBE_TIMEOUT_S = 180.0

#: SHA-256 of the packaged ``species_codes_lut.csv``, computed from the
#: committed file. No C++ counterpart is claimed: Gate 0
#: ``06-runtime-tables.md`` §4 records the ``SPP_CODES`` vs C++
#: ``sr_MSMT[]`` comparison as an open Phase 4 item.
_SPP_CSV_SHA256 = "6C21C4AB9097CDFBD0013B9C3414C81FF226F1846F9AA3738C296264BCBCA02E"

#: Exact byte length of the committed species table, pinned alongside
#: the digest so a provenance failure reports a useful difference.
_SPP_CSV_BYTES = 2699

#: The two ``SPP_CODES`` rows that ship with a blank ``fofem_cd``.
#: Gate 0 records that the table has no documented missing-value policy;
#: pinning the exact set means adopting one later is a visible change
#: rather than silent drift.
_SPP_MISSING_FOFEM_CD = {"JD", "JH"}


def _declared_package_data_patterns() -> list:
    """
    Read the ``pyfofem`` package-data glob list straight out of
    ``pyproject.toml``.

    Parsed with a targeted regular expression plus
    :func:`ast.literal_eval` rather than ``tomllib``, because
    ``tomllib`` only exists on Python 3.11+ while ``pyproject.toml``
    declares support from 3.10 - importing it unconditionally would make
    this module fail at collection on the lowest supported interpreter.
    The pattern list is a plain array of string literals, so
    ``literal_eval`` reads it exactly.

    :return: The declared glob patterns, in declaration order.
    :raises AssertionError: If the ``[tool.setuptools.package-data]``
        table or its ``pyfofem`` entry cannot be located.
    """
    with open(os.path.join(PROJECT_ROOT, "pyproject.toml"), "r", encoding="utf-8") as handle:
        text = handle.read()

    section = re.search(
        r"^\[tool\.setuptools\.package-data\]\s*$(.*?)(?=^\[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "[tool.setuptools.package-data] table not found"

    entry = re.search(r"^\s*pyfofem\s*=\s*(\[.*?\])", section.group(1), flags=re.DOTALL)
    assert entry is not None, "pyfofem package-data entry not found"

    patterns = ast.literal_eval(entry.group(1))
    assert isinstance(patterns, list)
    return patterns


def _emissions_csv_path() -> str:
    """
    Resolve the emission-factor CSV path exactly as the loader does.

    :return: Absolute, normalised path to the packaged
        ``emissions_factors.csv``, taken from the loader's own module
        constant rather than reconstructed independently.
    """
    return os.path.normpath(emission_calcs._EF_CSV_DEFAULT)


def _package_dir() -> str:
    """
    Locate the directory of the ``pyfofem`` package as imported.

    :return: Absolute path to the imported package's directory - the
        checkout's ``src/pyfofem`` in a development run, or
        ``site-packages/pyfofem`` in an installed-only run.
    """
    return os.path.dirname(os.path.abspath(pyfofem.__file__))


def _sha256_upper(path: str) -> str:
    """
    Hash a file's exact bytes.

    :param path: Path to the file to hash.
    :return: Uppercase hexadecimal SHA-256 digest.
    """
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest().upper()


def _species_csv_path() -> str:
    """
    Resolve the species-table path the same way the import-time loader
    does, from the defining module's own ``__file__``.

    :return: Absolute, normalised path to the packaged
        ``species_codes_lut.csv``.
    """
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(tree_flame_calcs.__file__)),
            "..",
            "supporting_data",
            "species_codes_lut.csv",
        )
    )


def test_emissions_factors_default_group_triple_maps_to_expected_types():
    """
    Category (a). The three FOFEM default groups - flame 3, smoulder 7,
    duff 8 - must resolve to the ``STFS`` / ``CWDRSC`` / ``DuffRSC``
    types respectively, which is what makes the ``expanded`` mode's
    three-group split scientifically meaningful.

    The defaults are read from the loader module's own constants, so a
    change to either the constants or the table breaks this test.

    :return: None. Raises via ``assert`` on mismatch.
    """
    assert emission_calcs._EF_GROUP_DEFAULT == 3
    assert emission_calcs._EF_SMOLDERING_GROUP_DEFAULT == 7
    assert emission_calcs._EF_DUFF_GROUP_DEFAULT == 8

    frame = emission_calcs._load_ef_csv()
    for group, expected_type in ((3, "STFS"), (7, "CWDRSC"), (8, "DuffRSC")):
        row = frame.iloc[group - 1]
        assert int(row["Group #"]) == group
        assert row["Type"] == expected_type


def test_emissions_factors_factor_rows_are_finite_and_non_negative():
    """
    Category (a). Every numeric emission factor in the eight real factor
    rows must be finite and non-negative.

    Restricted to rows 0-7 on purpose: the nine trailing rows the parse
    pulls in are cover-type labels with empty numeric columns, so
    applying a finiteness rule to them would be meaningless.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = emission_calcs._load_ef_csv()
    factors = frame.iloc[:8].drop(columns=["# Cover Type", "Type"])
    numeric = factors.apply(pd.to_numeric, errors="coerce")

    assert numeric.notna().all().all(), "non-numeric value in a factor row"
    assert (numeric.to_numpy() >= 0.0).all(), "negative emission factor"


def test_emissions_factors_group_domain_is_contiguous_one_to_eight():
    """
    Category (a). The eight real factor rows must carry contiguous
    ``Group #`` values 1-8 with the documented cover type and ``Type``
    classification, and ``Type`` must stay inside its closed domain.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = emission_calcs._load_ef_csv()
    factor_rows = frame.iloc[:8]

    assert [int(value) for value in factor_rows["Group #"]] == list(range(1, 9))
    for group, (cover, kind) in _EF_GROUPS.items():
        row = factor_rows.iloc[group - 1]
        assert row["# Cover Type"] == cover
        assert row["Type"] == kind
    assert set(factor_rows["Type"]) == {"STFS", "CWDRSC", "DuffRSC"}


def test_emissions_factors_missing_path_raises_a_specific_diagnostic():
    """
    Category (a). An ``ef_csv_path`` that does not exist must raise
    ``FileNotFoundError`` naming the offending path and the argument to
    fix, not fall back to the bundled table.

    Gate 0 ``06-runtime-tables.md`` §1 required-test 8 also notes the
    loader does **not** validate the schema of a file that *does* exist;
    that gap is recorded rather than asserted here, because asserting it
    would freeze the absence of validation as a contract.

    :return: None. Raises via ``assert`` on mismatch.
    """
    missing = os.path.join(_package_dir(), "supporting_data", "does_not_exist.csv")
    with pytest.raises(FileNotFoundError, match="emissions_factors.csv not found"):
        emission_calcs._load_ef_csv(missing)


def test_emissions_factors_parsed_shape_pins_the_f31_row_count():
    """
    Category (a). Pin the real parsed shape of ``_load_ef_csv``'s output
    and the raw file's line structure.

    The parse yields **17** rows, not 8: ``skiprows=1, header=0`` drops
    only the blank separator line, so the nine trailing non-factor lines
    (one cover-type section header plus eight cover labels) arrive as
    data rows with a NaN ``Group #``. That is the mechanical cause of
    Gate 0 finding **F-31**, where ``_validate_group`` accepts groups
    1-17 instead of the intended 1-8.

    This test documents the parse; it does **not** assert the defective
    validation range is correct. F-31 itself is scheduled as
    ``BR-EMI-GROUPERR`` (``XFAIL-STRICT``) under ``calc_smoke_emissions``
    in Phase 4.

    :return: None. Raises via ``assert`` on mismatch.
    """
    with open(_emissions_csv_path(), "r", encoding="utf-8-sig") as handle:
        raw_lines = handle.read().splitlines()
    assert len(raw_lines) == _EF_RAW_LINES

    frame = emission_calcs._load_ef_csv()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (_EF_PARSED_ROWS, 207)
    assert frame["Group #"].iloc[8:].isna().all()
    assert frame["Group #"].iloc[:8].notna().all()


def test_emissions_factors_provenance_digest_is_exact():
    """
    Category (b). Pin the packaged emission-factor table's exact bytes.

    The digest is the same SHA-256 as the pinned C++
    ``FOF_UNIX/Emission_Factors.csv``, so this assertion pins
    byte-identity with the C++ table that ``NES_Read`` loads (module
    docstring; Gate 0 ``06-runtime-tables.md`` §1a). Any edit to the
    Python copy breaks the identity and must be re-justified against the
    pinned reference.

    :return: None. Raises via ``assert`` on mismatch.
    """
    assert _sha256_upper(_emissions_csv_path()) == _EF_CSV_SHA256


def test_emissions_factors_sentinel_factor_values_are_intact():
    """
    Category (a). Spot-check real, individually checkable factor values
    rather than asserting the table is merely populated.

    Chosen sentinels, all in g/kg: group 3 (Western Forest - Rx, the
    flaming default) CO2 1598 / CO 105; group 7 (Woody RSC, the
    smouldering default) CO2 1408 / CO 229; group 8 (Duff RSC, the duff
    default) CO2 1371 / CO 257. The relationship they encode is
    physically meaningful and is asserted directly: residual smouldering
    produces *less* CO2 and *more* CO per kilogram burned than flaming
    combustion.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = emission_calcs._load_ef_csv()
    flaming = frame.iloc[2]
    woody_smouldering = frame.iloc[6]
    duff_smouldering = frame.iloc[7]

    assert float(flaming["CO2"]) == pytest.approx(1598.0)
    assert float(flaming["CO"]) == pytest.approx(105.0)
    assert float(woody_smouldering["CO2"]) == pytest.approx(1408.0)
    assert float(woody_smouldering["CO"]) == pytest.approx(229.0)
    assert float(duff_smouldering["CO2"]) == pytest.approx(1371.0)
    assert float(duff_smouldering["CO"]) == pytest.approx(257.0)

    for smouldering in (woody_smouldering, duff_smouldering):
        assert float(smouldering["CO2"]) < float(flaming["CO2"])
        assert float(smouldering["CO"]) > float(flaming["CO"])


@pytest.mark.installed_artifact
def test_loaders_resolve_resources_independently_of_the_working_directory():
    """
    Category (a). Prove in a **real child process**, started from an
    unrelated working directory with an empty ``PYTHONPATH``, that
    importing ``pyfofem`` succeeds and that both runtime resources
    resolve to files inside the imported package's own directory.

    This is the check that distinguishes a package-relative
    (``__file__``-based) loader from one that would depend on the
    checkout layout or the process's cwd. Run through
    :func:`tests.cpp_parity_live._proc.run_bounded` so the probe has a
    hard timeout and a real process-tree teardown if it ever hangs.

    :return: None. Raises via ``assert`` on mismatch.
    """
    probe = (
        "import os, pyfofem;"
        "from pyfofem.components import emission_calcs, tree_flame_calcs;"
        "pkg = os.path.dirname(os.path.abspath(pyfofem.__file__));"
        "ef = os.path.normpath(emission_calcs._EF_CSV_DEFAULT);"
        "spp = os.path.normpath(os.path.join("
        "os.path.dirname(os.path.abspath(tree_flame_calcs.__file__)),"
        "'..','supporting_data','species_codes_lut.csv'));"
        "print(pkg);print(ef);print(spp);"
        "print(os.path.isfile(ef), os.path.isfile(spp));"
        "print(len(tree_flame_calcs.SPP_CODES), len(emission_calcs._load_ef_csv()))"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = run_bounded(
        [sys.executable, "-c", probe],
        timeout=_RESOURCE_PROBE_TIMEOUT_S,
        cwd=os.path.dirname(sys.executable),
        env=env,
    )
    assert result.returncode == 0, result.stderr

    package_dir, ef_path, spp_path, exists_line, sizes_line = (
        result.stdout.strip().splitlines()
    )
    assert exists_line == "True True"
    assert sizes_line == f"121 {_EF_PARSED_ROWS}"
    for path in (ef_path, spp_path):
        assert os.path.isabs(path)
        assert os.path.commonpath([package_dir, path]) == package_dir


def test_packaging_config_ships_both_runtime_csvs_and_no_vendor_binaries():
    """
    Category (a). Assert from ``pyproject.toml`` - the packaging config,
    not merely the checkout - that both runtime CSVs are declared as
    installed package data and that the declaration cannot pull in the
    bundled vendor distribution.

    ``[tool.setuptools.package-data] pyfofem = ["supporting_data/*.csv"]``
    matches exactly the two runtime tables and, because the glob is not
    recursive, matches nothing under ``supporting_data/FOFEM6.7/`` - so
    the tracked ``FOF_GUI.exe``, the two Microsoft DLLs and the help PDF
    stay out of the distribution. Gate 0 ``06-runtime-tables.md`` §3
    notes that exclusion is currently *accidental*; this test makes it
    an asserted contract, so widening the glob to
    ``supporting_data/**`` fails here instead of silently shipping a
    Windows executable.

    :return: None. Raises via ``assert`` on mismatch.
    """
    patterns = _declared_package_data_patterns()
    assert patterns == ["supporting_data/*.csv"]

    data_dir = os.path.join(PROJECT_ROOT, "src", "pyfofem", "supporting_data")
    matched = sorted(
        name
        for name in os.listdir(data_dir)
        if name.lower().endswith(".csv")
        and os.path.isfile(os.path.join(data_dir, name))
    )
    assert matched == ["emissions_factors.csv", "species_codes_lut.csv"]

    for pattern in patterns:
        assert "**" not in pattern
        assert "FOFEM6.7" not in pattern


def test_species_table_columns_dtypes_and_row_count():
    """
    Category (a). The species table must parse to a 121-row, 4-column
    frame with the documented column names in order, an integral
    ``num_cd``, and textual identifier/classification columns.

    The text columns are asserted by *value type* (every non-null entry
    is a ``str``) rather than by pandas dtype identity. pandas 3.0
    infers ``StringDtype`` where pandas 1.x/2.x inferred ``object``, so
    an ``== object`` assertion would be a test of the pandas version
    rather than of the shipped data - it failed exactly that way under
    pandas 3.0.5 during the Phase 3 isolated-wheel run.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = pd.read_csv(_species_csv_path())
    assert frame.shape == (121, 4)
    assert list(frame.columns) == ["species_cd", "fofem_cd", "tree_type", "num_cd"]
    assert pd.api.types.is_integer_dtype(frame["num_cd"])

    for column in ("species_cd", "fofem_cd", "tree_type"):
        values = frame[column].dropna()
        assert len(values) > 0
        assert all(isinstance(value, str) for value in values), column


def test_species_table_export_matches_the_file_on_disk():
    """
    Category (a). The exported ``SPP_CODES`` object must be
    value-identical to an independent fresh parse of the packaged CSV -
    proving nothing mutated the shared table during import and that the
    export really is the shipped data.

    :return: None. Raises via ``assert`` on mismatch.
    """
    fresh = pd.read_csv(_species_csv_path())
    pd.testing.assert_frame_equal(pyfofem.SPP_CODES, fresh)


def test_species_table_loader_resolves_inside_the_installed_package():
    """
    Category (a). The import-time species-table path must be absolute
    and resolve to an existing file **inside the imported package's own
    directory**, not relative to the repository or the working
    directory. The same is asserted for the emission-factor loader's
    module-level path constant.

    :return: None. Raises via ``assert`` on mismatch.
    """
    package_dir = _package_dir()
    for path in (_species_csv_path(), _emissions_csv_path()):
        assert os.path.isabs(path)
        assert os.path.isfile(path)
        assert os.path.commonpath([package_dir, path]) == package_dir
        assert os.path.basename(os.path.dirname(path)) == "supporting_data"


def test_species_table_missing_value_policy_is_exactly_two_blank_fofem_codes():
    """
    Category (a). Pin the table's *actual* missing-value content: every
    column except ``fofem_cd`` is fully populated, and exactly the two
    juniper shrub rows ``JD`` and ``JH`` ship without a FOFEM code.

    Gate 0 records that there is no documented missing-value policy for
    these tables. Pinning the exact set means adopting one - or letting
    a third blank row slip in - is a visible change.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = pd.read_csv(_species_csv_path())
    for column in ("species_cd", "tree_type", "num_cd"):
        assert frame[column].notna().all(), f"blank value in {column}"

    blank = frame.loc[frame["fofem_cd"].isna()]
    assert set(blank["species_cd"]) == _SPP_MISSING_FOFEM_CD
    assert set(blank["tree_type"]) == {"shrub"}


def test_species_table_provenance_digest_and_size_are_exact():
    """
    Category (a). Pin the packaged species table's exact bytes and
    length, so a content change is a deliberate, reviewable event.

    No C++ byte-identity is claimed: Gate 0 ``06-runtime-tables.md`` §4
    records the ``SPP_CODES`` vs ``sr_MSMT[]`` row-coverage comparison
    as an open Phase 4 work item.

    :return: None. Raises via ``assert`` on mismatch.
    """
    path = _species_csv_path()
    assert os.path.getsize(path) == _SPP_CSV_BYTES
    assert _sha256_upper(path) == _SPP_CSV_SHA256


def test_species_table_unique_keys_and_closed_domains():
    """
    Category (a). ``species_cd`` and ``num_cd`` are identifying keys and
    must be unique; ``tree_type`` must stay inside its closed
    three-value domain; and ``num_cd`` must stay inside the documented
    100-999 code range whose top three values are the 997/998/999
    catch-all sentinels.

    :return: None. Raises via ``assert`` on mismatch.
    """
    frame = pd.read_csv(_species_csv_path())
    assert frame["species_cd"].is_unique
    assert frame["num_cd"].is_unique
    assert set(frame["tree_type"].unique()) == {"conifer", "hardwood", "shrub"}
    assert int(frame["num_cd"].min()) == 100
    assert int(frame["num_cd"].max()) == 999
    assert set(frame["num_cd"]) >= {997, 998, 999}
