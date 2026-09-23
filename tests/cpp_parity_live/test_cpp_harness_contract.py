#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpp_harness_contract.py - Live harness self-test matrix (Phase 2).

Builds and drives the real, compiled ``fofem_test`` C++ harness
(``reference/fofem_cpp_overlay/source/FOF_UNIX/test_harness.cpp``) and
verifies it against the 19-row + 11a-11g self-test matrix in
``development/plans/gate0/05-harness-contract.md`` §10, plus the CLI/species
-loader contract from the Phase 2 amendment and the SHA-256/consume
qualification checks from the Phase 2 audit.

Every test in this module requires a real MSVC/CMake/Ninja toolchain
(Windows-only by construction — see ``_harness_support.toolchain_status``)
and builds/invokes the real compiled binary; nothing here reimplements a
C++ equation. If the toolchain is unavailable the whole module is skipped
with a specific reason, not silently passed.

AGENTS.md function-order exception: test functions in this module are
deliberately kept in contract-row order (row 1, row 2, row 3, ...), not
alphabetized, since the file's whole purpose is to be read alongside
``gate0/05-harness-contract.md`` §10's numbered table — alphabetizing
would scatter e.g. ``test_row11a_*``/``test_row11g_*``/``test_row12_*``
away from each other and from the table row they verify. Non-test helper
data (``MODES``, ``NUMERIC_FIELD_INDEX``, ``SECOND_ROW_OK``, ``_run``,
``_species_kw``) appears once, near the top, in a natural declaration
order rather than alphabetized, for the same readability reason.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import MODE_SCHEMA_VERSIONS
from tests.cpp_parity_live._harness_support import (
    FOF_UNIX_DIR,
    HARNESS_EXE,
    HARNESS_EXE_OVERRIDE_ENV_VAR,
    HARNESS_SOIDIAG_EXE,
    SPECIES_CSV,
    TIMEOUT_HARNESS_RUN_S,
    ensure_built,
    ensure_soidiag_built,
    run_harness,
    toolchain_status,
)
from tests.cpp_parity_live._proc import run_bounded

pytestmark = pytest.mark.cpp_reference


# ===========================================================================
# Per-mode canonical valid schema + row (mirrors the harness's own header
# constants — see test_harness.cpp's *_HEADER arrays).
# ===========================================================================

CONSUME_HEADER = [
    "case_id", "expect_error",
    "litter_tac", "duff_tac", "duff_depth_in", "duff_moist_pct",
    "herb_tac", "shrub_tac", "crown_fol_tac", "crown_bra_tac", "pct_crown_burn",
    "dw10_moist_pct", "dw1000_moist_pct", "litter_moist_pct",
    "dw1_tac", "dw10_tac", "dw100_tac", "dw1000_tac", "pct_rot",
    "snd_dw3_tac", "snd_dw6_tac", "snd_dw9_tac", "snd_dw20_tac",
    "rot_dw3_tac", "rot_dw6_tac", "rot_dw9_tac", "rot_dw20_tac",
    "region", "season", "fuel_cat", "cover_group", "cover_class",
    "duff_moist_method",
    "intensity_kw_m", "ig_time_s", "windspeed_m_s", "depth_ft",
    "ambient_temp_c",
    "critical_intensity_kw_m", "ef_flame_group", "ef_smolder_group",
    "ef_duff_group",
    "batch_equ", "eq_lit", "eq_duf_loa", "eq_duf_dep", "eq_mse", "eq_herb",
    "eq_shrub",
]
CONSUME_ROW_OK = [
    "c1", "0", "2.0", "10.0", "2.0", "50.0", "0.5", "1.0", "0.5", "0.5", "50",
    "10", "20", "15", "0.5", "0.5", "1.0", "2.0", "10", "0", "0", "0", "0",
    "0", "0", "0", "0", "InteriorWest", "Summer", "Natural", "GrassGroup",
    "Grass", "NFDR", "300", "60", "2", "0.5", "20", "50", "3", "7", "8",
    "No", "-1", "-1", "-1", "-1", "-1", "-1",
]

LITTER_EQ_HEADER = ["case_id", "expect_error", "equ", "load_tac", "dw10_moist_pct"]
LITTER_EQ_ROW_OK = ["c1", "0", "997", "2.0", "15"]

SHRUB_HERB_EQ_HEADER = [
    "case_id", "expect_error", "region", "cover_group", "season", "fuel_cat",
    "shrub_tac", "herb_tac", "litter_tac", "duff_tac", "duff_moist_pct",
    "crown_fol_tac", "crown_bra_tac", "pct_crown_burn", "force_shrub_equ",
]
SHRUB_HERB_EQ_ROW_OK = [
    "c1", "0", "InteriorWest", "GrassGroup", "Summer", "Natural",
    "1.0", "0.5", "2.0", "10.0", "50", "0.5", "0.5", "50", "-1",
]

#: Mortality is the one mode at schema v2 (see MODE_SCHEMA_VERSIONS and
#: test_harness.cpp's MODES[] table). v2 renamed v1's misnamed `ckr_pct`
#: to `ckr_rating` (d_MIS.f_CKR is the 0-4 cambium kill RATING,
#: fof_mrt.cpp:1849-1851/1937) and appended `density_tpa` (d_MIS.f_Den,
#: which ValidInput requires in [1, 20000], fof_mrt.cpp:1854-1856).
MORTALITY_HEADER = [
    "case_id", "expect_error", "species", "equ_type", "dbh_in", "ht_ft",
    "crown_ratio_x10", "fs_value_ft", "fs_kind", "bole_char_ft",
    "fire_severity", "ckr_rating", "cvk_pct", "beetles", "density_tpa",
]
MORTALITY_ROW_OK = [
    "c1", "0", "PSME", "CroSco", "12", "60", "50", "4", "Flame", "0",
    "NA", "0", "0", "0", "100",
]

BARK_THICK_HEADER = ["case_id", "expect_error", "species", "dbh_in"]
BARK_THICK_ROW_OK = ["c1", "0", "PSME", "12"]

CANOPY_COVER_HEADER = ["case_id", "expect_error", "stand_id", "species", "dbh_in", "ht_ft"]
CANOPY_COVER_ROW_OK = ["c1", "0", "s1", "PSME", "12", "60"]

MODES = {
    "consume": dict(
        header=CONSUME_HEADER, row=CONSUME_ROW_OK, needs_species=False,
        suffixes=("_summary", "_components"), primary_suffix="_summary",
    ),
    "litter_eq": dict(
        header=LITTER_EQ_HEADER, row=LITTER_EQ_ROW_OK, needs_species=False,
        suffixes=("",), primary_suffix="",
    ),
    "shrub_herb_eq": dict(
        header=SHRUB_HERB_EQ_HEADER, row=SHRUB_HERB_EQ_ROW_OK, needs_species=False,
        suffixes=("",), primary_suffix="",
    ),
    "mortality": dict(
        header=MORTALITY_HEADER, row=MORTALITY_ROW_OK, needs_species=True,
        suffixes=("",), primary_suffix="",
    ),
    "bark_thick": dict(
        header=BARK_THICK_HEADER, row=BARK_THICK_ROW_OK, needs_species=True,
        suffixes=("",), primary_suffix="",
    ),
    "canopy_cover": dict(
        header=CANOPY_COVER_HEADER, row=CANOPY_COVER_ROW_OK, needs_species=True,
        suffixes=("_trees", "_stands", "_groups"), primary_suffix="_trees",
    ),
}

ALL_MODE_NAMES = list(MODES.keys())

#: Index of one strict-double field per mode, used by the generic
#: blank/non-numeric/nan-inf-overflow/overlong-field parametrizations so
#: rows 5-7 exercise every mode's own field, not only litter_eq's.
NUMERIC_FIELD_INDEX = {
    "consume": 2,          # litter_tac
    "litter_eq": 3,        # load_tac
    "shrub_herb_eq": 6,    # shrub_tac
    "mortality": 4,        # dbh_in
    "bark_thick": 3,       # dbh_in
    "canopy_cover": 4,     # dbh_in
}

#: A SECOND, scientifically distinct valid row per mode — different
#: species/equation/values, not merely a different case_id. "PIPO"
#: (Ponderosa Pine) is a real tracked FOF_SPP.CSV code distinct from
#: "PSME", with its own mortality/bark/canopy equation numbers, used for
#: the species-driven modes. Needed for row 15 (same-process multi-row
#: isolation) and row 17 (order-dependent state): changing only case_id
#: cannot reveal state that depends on the actual computed values.
SECOND_ROW_OK = {
    "consume": [
        "c2", "0", "4.0", "20.0", "3.0", "40.0", "1.0", "2.0", "1.0", "1.0",
        "60", "12", "18", "10", "1.0", "1.0", "2.0", "4.0", "20", "0", "0",
        "0", "0", "0", "0", "0", "0", "PacificWest", "Fall", "Slash",
        "GrassGroup", "Grass", "NFDR", "500", "80", "3", "1.0", "15", "60",
        "3", "7", "8", "No", "-1", "-1", "-1", "-1", "-1", "-1",
    ],
    "litter_eq": ["c2", "0", "998", "3.0", "NA"],
    "shrub_herb_eq": [
        "c2", "0", "PacificWest", "GrassGroup", "Fall", "Slash",
        "2.0", "1.0", "4.0", "20.0", "60", "1.0", "1.0", "60", "-1",
    ],
    "mortality": [
        "c2", "0", "PIPO", "CroSco", "18", "70", "60", "6", "Scorch", "0",
        "NA", "0", "0", "0", "250",
    ],
    "bark_thick": ["c2", "0", "PIPO", "18"],
    "canopy_cover": ["c2", "0", "s1", "PIPO", "18", "70"],
}


def _species_kw(mode: str) -> dict:
    return {"species_csv": SPECIES_CSV} if MODES[mode]["needs_species"] else {}


def _run(mode: str, rows, tmp_path, name="case", **kwargs):
    m = MODES[mode]
    kwargs.setdefault("output_suffixes", m["suffixes"])
    return run_harness(
        mode, m["header"], rows, os.path.join(str(tmp_path), name),
        **_species_kw(mode), **kwargs,
    )


# ===========================================================================
# Session-scoped build gate
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def _built():
    ok, reason = toolchain_status()
    if not ok:
        pytest.skip(f"MSVC/CMake/Ninja toolchain unavailable: {reason}")
    ok, reason = ensure_built()
    if not ok:
        pytest.fail(f"fofem_test build failed:\n{reason}")
    assert os.path.isfile(HARNESS_EXE)


# ===========================================================================
# Row 1 — valid file, all rows ok
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row1_valid_all_rows_ok(mode, tmp_path):
    res = _run(mode, [MODES[mode]["row"]], tmp_path)
    assert res.returncode == 0, res.stderr
    primary = res.rows(MODES[mode]["primary_suffix"])
    assert len(primary) == 1
    assert primary[0]["outcome"] == "ok"


# ===========================================================================
# Row 2 — missing magic/version line
#
# read_input_file() is one shared function for every mode (mode only
# selects which expected_header vector it validates against); rows 2/3/4/9
# are parametrized across all six modes anyway, both to prove that shared
# path integrates correctly with each mode's real header (not assumed from
# one mode) and per the explicit instruction to exercise every applicable
# mode rather than routing most parser/error cases through litter_eq alone.
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row2_missing_magic_line(mode, tmp_path):
    m = MODES[mode]
    prefix = os.path.join(str(tmp_path), "case")
    in_path = prefix + "_in.csv"
    # Write only a header line, no magic line at all.
    with open(in_path, "w", newline="\n") as f:
        f.write(",".join(m["header"]) + "\n")
        f.write(",".join(m["row"]) + "\n")
    proc = run_bounded([HARNESS_EXE, in_path, prefix], cwd=FOF_UNIX_DIR,
                        timeout=TIMEOUT_HARNESS_RUN_S)
    assert proc.returncode != 0
    assert not os.path.isfile(prefix + m["primary_suffix"] + ".csv")


# ===========================================================================
# Row 3 — wrong schema version
#
# The accepted version is declared PER MODE (contract §2/§5 headings,
# test_harness.cpp's MODES[] table, MODE_SCHEMA_VERSIONS). A literal "2"
# is therefore no longer a universally-wrong version — it is mortality's
# CORRECT one — so the generic rejection test uses a version no mode
# declares, and the per-mode cross-rejection is asserted separately
# below.
# ===========================================================================

#: A schema version no mode declares, used by the generic row-3 test.
UNDECLARED_SCHEMA_VERSION = "99"


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row3_wrong_schema_version(mode, tmp_path):
    assert UNDECLARED_SCHEMA_VERSION not in set(MODE_SCHEMA_VERSIONS.values())
    m = MODES[mode]
    res = _run(mode, [m["row"]], tmp_path,
               schema_version=UNDECLARED_SCHEMA_VERSION)
    assert res.returncode != 0
    assert "schema_version" in res.stderr


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row3_declared_schema_version_is_accepted(mode, tmp_path):
    """The version this mode declares is accepted, and is echoed verbatim
    into every output row — so an archived CSV always says which schema
    produced it."""
    m = MODES[mode]
    declared = MODE_SCHEMA_VERSIONS[mode]
    res = _run(mode, [m["row"]], tmp_path, schema_version=declared)
    assert res.returncode == 0, res.stderr
    primary = res.rows(m["primary_suffix"])
    assert primary and primary[0]["schema_version"] == declared


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row3_other_modes_schema_version_is_rejected(mode, tmp_path):
    """A version another mode declares is still rejected here.

    This is what makes mortality's v2 a genuine per-mode revision rather
    than a silent redefinition of "v1": mortality rejects "1", and every
    v1 mode rejects "2".
    """
    m = MODES[mode]
    declared = MODE_SCHEMA_VERSIONS[mode]
    others = sorted(set(MODE_SCHEMA_VERSIONS.values()) - {declared})
    assert others, "expected at least one other declared schema version"
    for other in others:
        res = _run(mode, [m["row"]], tmp_path, name=f"case_v{other}",
                   schema_version=other)
        assert res.returncode != 0, (
            f"mode {mode!r} (declares v{declared}) accepted v{other}"
        )


# ===========================================================================
# Row 4 — column added / removed / reordered / duplicated
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row4_column_removed(mode, tmp_path):
    m = MODES[mode]
    bad_header = m["header"][:-1]
    res = _run(mode, [m["row"][:-1]], tmp_path, header_override=bad_header)
    assert res.returncode != 0


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row4_column_added(mode, tmp_path):
    m = MODES[mode]
    bad_header = m["header"] + ["extra_col"]
    res = _run(mode, [m["row"] + ["x"]], tmp_path, header_override=bad_header)
    assert res.returncode != 0


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row4_column_reordered(mode, tmp_path):
    m = MODES[mode]
    bad_header = list(m["header"])
    bad_header[2], bad_header[3] = bad_header[3], bad_header[2]
    res = _run(mode, [m["row"]], tmp_path, header_override=bad_header)
    assert res.returncode != 0


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row4_column_duplicated(mode, tmp_path):
    m = MODES[mode]
    bad_header = m["header"] + [m["header"][-1]]
    res = _run(mode, [m["row"] + [m["row"][-1]]], tmp_path,
               header_override=bad_header)
    assert res.returncode != 0


# ===========================================================================
# Row 5 — blank numeric field (never 0.0)
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row5_blank_numeric_field(mode, tmp_path):
    m = MODES[mode]
    row = list(m["row"])
    row[NUMERIC_FIELD_INDEX[mode]] = ""
    res = _run(mode, [row], tmp_path)
    assert res.returncode != 0
    assert "blank" in res.stderr.lower()


# ===========================================================================
# Row 6 — non-numeric numeric field
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
@pytest.mark.parametrize("bad", ["abc", "1.2.3", "1e", "0x10"])
def test_row6_non_numeric_field(mode, bad, tmp_path):
    m = MODES[mode]
    row = list(m["row"])
    row[NUMERIC_FIELD_INDEX[mode]] = bad
    res = _run(mode, [row], tmp_path, name=f"case_{bad!r}")
    assert res.returncode != 0


# ===========================================================================
# Row 7 — nan / inf / -inf where forbidden, plus the audit's overflow set
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
@pytest.mark.parametrize(
    "bad", ["nan", "inf", "-inf", "1e999", "-1e999", "1e-9999"]
)
def test_row7_nan_inf_and_overflow(mode, bad, tmp_path):
    m = MODES[mode]
    row = list(m["row"])
    row[NUMERIC_FIELD_INDEX[mode]] = bad
    res = _run(mode, [row], tmp_path, name=f"case_{bad!r}")
    assert res.returncode != 0


# ===========================================================================
# Row 8 — value out of the field's documented range
# ===========================================================================

def test_row8_ef_group_out_of_domain(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[CONSUME_HEADER.index("ef_flame_group")] = "9"  # domain is 1-8
    res = _run("consume", [row], tmp_path)
    assert res.returncode != 0


@pytest.mark.parametrize("mode", ["litter_eq", "mortality", "canopy_cover"])
def test_row8_expect_error_out_of_domain(mode, tmp_path):
    # expect_error's 0/1 domain is enforced by one shared parser
    # (parse_expect_error) called identically by every mode; exercised
    # here across three distinct modes (not just one) as direct evidence
    # that the shared path integrates correctly everywhere it's used.
    m = MODES[mode]
    row = list(m["row"])
    row[1] = "2"
    res = _run(mode, [row], tmp_path)
    assert res.returncode != 0


# ===========================================================================
# Row 9 — duplicate case_id
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row9_duplicate_case_id(mode, tmp_path):
    m = MODES[mode]
    res = _run(mode, [m["row"], m["row"]], tmp_path)
    assert res.returncode != 0


# ===========================================================================
# Row 10 — empty file / header-only.
#
# BOTH must exit nonzero with a distinct message — a header-only (zero
# data row) file is NOT a valid zero-row run; read_input_file() rejects it
# explicitly ("no data rows (header-only input)"), a message distinct from
# the "empty file (no magic/version line)" case.
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row10_empty_file(mode, tmp_path):
    prefix = os.path.join(str(tmp_path), "case")
    in_path = prefix + "_in.csv"
    open(in_path, "w").close()
    proc = run_bounded([HARNESS_EXE, in_path, prefix], cwd=FOF_UNIX_DIR,
                        timeout=TIMEOUT_HARNESS_RUN_S)
    assert proc.returncode != 0
    assert "empty" in proc.stderr.lower() or "magic" in proc.stderr.lower()


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row10_header_only(mode, tmp_path):
    m = MODES[mode]
    res = _run(mode, [], tmp_path)
    assert res.returncode != 0
    assert "header-only" in res.stderr.lower() or "no data rows" in res.stderr.lower()
    # Distinct from the fully-empty-file message (row 10's other case).
    assert "no magic/version line" not in res.stderr.lower()


# ===========================================================================
# Row 11 / 11a-11g — expect_error shapes
# ===========================================================================

def test_row11a_single_primary_litter_eq_expected_error(tmp_path):
    row = list(LITTER_EQ_ROW_OK)
    row[1] = "1"
    row[2] = "5"  # neither 997 nor 998 -> harness dispatch error
    res = _run("litter_eq", [row], tmp_path)
    assert res.returncode == 0
    rows = res.rows()
    assert len(rows) == 1
    assert rows[0]["outcome"] == "expected_model_error"
    assert rows[0]["err_text"]


def test_row11a_single_primary_mortality_expected_error(tmp_path):
    row = list(MORTALITY_ROW_OK)
    row[1] = "1"
    row[2] = "ZZZZZZ"  # unknown species
    res = _run("mortality", [row], tmp_path)
    assert res.returncode == 0
    rows = res.rows()
    assert rows[0]["outcome"] == "expected_model_error"


def test_row11a_single_primary_bark_thick_expected_error(tmp_path):
    row = list(BARK_THICK_ROW_OK)
    row[1] = "1"
    row[2] = "ZZZZZZ"
    res = _run("bark_thick", [row], tmp_path)
    assert res.returncode == 0
    assert res.rows()[0]["outcome"] == "expected_model_error"


def test_row11a_single_primary_shrub_herb_eq_expected_error(tmp_path):
    row = list(SHRUB_HERB_EQ_ROW_OK)
    row[1] = "1"
    row[SHRUB_HERB_EQ_HEADER.index("force_shrub_equ")] = "999"  # not implemented
    res = _run("shrub_herb_eq", [row], tmp_path)
    assert res.returncode == 0
    rows = res.rows()
    assert len(rows) == 1
    assert rows[0]["outcome"] == "expected_model_error"
    assert rows[0]["err_text"]


def test_row11b_consume_constant_fanout_expected_error(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[1] = "1"
    row[CONSUME_HEADER.index("ig_time_s")] = "5"  # out of C++ fire bounds
    res = _run("consume", [row], tmp_path)
    assert res.returncode == 0
    summary = res.rows("_summary")
    assert summary[0]["outcome"] == "expected_model_error"
    components = res.rows("_components")
    assert len(components) == 0  # zero component rows for the errored row

    # rows(components) == 11 * count(ok) also holds when mixed with an ok row
    ok_row = list(CONSUME_ROW_OK)
    ok_row[0] = "c2"
    res2 = _run("consume", [row, ok_row], tmp_path, name="mixed")
    assert res2.returncode == 0
    assert len(res2.rows("_components")) == 11 * 1


def test_row11d_canopy_cover_aggregate_suppressed_on_expected_error(tmp_path):
    rows = [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "1", "s1", "ZZZZZZ", "12", "60"],
    ]
    res = _run("canopy_cover", rows, tmp_path)
    assert res.returncode == 0
    groups = res.rows("_groups")
    assert len(groups) == 1
    assert groups[0]["aggregate_emitted"] == "0"
    assert groups[0]["suppression_reason"] == "expected_model_error_member"
    assert res.rows("_stands") == []


def test_row11e_two_sided_unexpectedly_succeeds(tmp_path):
    row = list(LITTER_EQ_ROW_OK)
    row[1] = "1"  # expect_error=1 but this row is valid and will succeed
    res = _run("litter_eq", [row], tmp_path)
    assert res.returncode != 0
    assert res.rows()[0]["outcome"] == "unexpected_failure"


def test_row11f_canopy_cover_unexpected_failure_member(tmp_path):
    rows = [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "1", "s1", "PSME", "12", "60"],  # expect_error=1 but succeeds
    ]
    res = _run("canopy_cover", rows, tmp_path)
    assert res.returncode != 0
    groups = res.rows("_groups")
    assert groups[0]["suppression_reason"] == "unexpected_failure_member"
    assert res.rows("_stands") == []


def test_row11g_aggregate_state_matches_membership_invariant(tmp_path):
    # First, the two states the normal code path CAN construct on its own:
    # all-ok (aggregate present) and mixed (aggregate absent). Observing
    # these two alone does NOT prove the harness would reject an
    # inconsistent state — it only proves the two states it already
    # produces are individually self-consistent. See the fault-injection
    # test below for the actual rejection proof.
    res_ok = _run("canopy_cover", [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "0", "s1", "PSME", "14", "65"],
    ], tmp_path, name="allok")
    groups_ok = res_ok.rows("_groups")
    assert groups_ok[0]["aggregate_emitted"] == "1"
    assert len(res_ok.rows("_stands")) == 1

    res_mixed = _run("canopy_cover", [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "1", "s1", "ZZZZZZ", "12", "60"],
    ], tmp_path, name="mixed")
    groups_mixed = res_mixed.rows("_groups")
    assert groups_mixed[0]["aggregate_emitted"] == "0"
    assert res_mixed.rows("_stands") == []


def test_row11g_injected_inconsistent_aggregate_is_rejected(tmp_path):
    """Actually inject an invalid reconciliation state (FOFEM_TEST_FAULT=
    canopy_aggregate_mismatch forces emit_aggregate=true for a mixed group
    that would otherwise correctly suppress it) and prove the harness's
    own final-reconciliation pass rejects it. This is the real proof rows
    11d/11f/11g's "aggregate present iff all members ok" invariant is
    actually enforced, not merely never violated by the paths the normal
    code happens to take."""
    m = MODES["canopy_cover"]
    rows = [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "1", "s1", "ZZZZZZ", "12", "60"],  # mixed group: n_ok != n_members
    ]
    prefix = os.path.join(str(tmp_path), "fault")
    in_path = prefix + "_in.csv"
    with open(in_path, "w", newline="\n") as f:
        f.write("#fofem-harness,canopy_cover,1\n")
        f.write(",".join(m["header"]) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    env = dict(os.environ)
    env["FOFEM_TEST_FAULT"] = "canopy_aggregate_mismatch"
    proc = run_bounded(
        [HARNESS_EXE, in_path, prefix, "--species-csv", SPECIES_CSV],
        cwd=FOF_UNIX_DIR, env=env, timeout=TIMEOUT_HARNESS_RUN_S,
    )
    assert proc.returncode != 0
    assert "aggregate-reconciliation" in proc.stderr

    # Sanity: the SAME input, WITHOUT the fault env var, succeeds normally
    # (proves the rejection above is really caused by the injected fault,
    # not by the input itself being otherwise invalid).
    proc_clean = run_bounded(
        [HARNESS_EXE, in_path, prefix + "_clean", "--species-csv", SPECIES_CSV],
        cwd=FOF_UNIX_DIR, timeout=TIMEOUT_HARNESS_RUN_S,
    )
    assert proc_clean.returncode == 0


# ===========================================================================
# Row 12 — row unexpectedly errors (expect_error=0, model errors)
# ===========================================================================

def test_row12_row_unexpectedly_errors_bark_thick(tmp_path):
    row = list(BARK_THICK_ROW_OK)
    row[2] = "ZZZZZZ"  # expect_error stays 0
    res = _run("bark_thick", [row], tmp_path)
    assert res.returncode != 0
    assert res.rows()[0]["outcome"] == "unexpected_failure"


def test_row12_row_unexpectedly_errors_canopy_cover(tmp_path):
    row = list(CANOPY_COVER_ROW_OK)
    row[CANOPY_COVER_HEADER.index("species")] = "ZZZZZZ"
    res = _run("canopy_cover", [row], tmp_path)
    assert res.returncode != 0
    assert res.rows("_trees")[0]["outcome"] == "unexpected_failure"


def test_row12_row_unexpectedly_errors_mortality(tmp_path):
    row = list(MORTALITY_ROW_OK)
    row[2] = "ZZZZZZ"  # expect_error stays 0
    res = _run("mortality", [row], tmp_path)
    assert res.returncode != 0
    assert res.rows()[0]["outcome"] == "unexpected_failure"


# ===========================================================================
# Row 13 — output path unwritable
# ===========================================================================

@pytest.mark.parametrize("mode", ["litter_eq", "mortality", "canopy_cover"])
def test_row13_output_path_unwritable(mode, tmp_path):
    # The input file must exist and be readable (it lives in tmp_path, a
    # real directory); only the OUTPUT prefix's directory is missing, so
    # this genuinely exercises CsvWriter's fail-closed open() check rather
    # than failing earlier trying to write a nonexistent input path.
    m = MODES[mode]
    in_path = os.path.join(str(tmp_path), "case_in.csv")
    with open(in_path, "w", newline="\n") as f:
        f.write(f"#fofem-harness,{mode},1\n")
        f.write(",".join(m["header"]) + "\n")
        f.write(",".join(m["row"]) + "\n")
    bad_prefix = os.path.join(str(tmp_path), "no_such_dir", "deeper", "case")
    args = [HARNESS_EXE, in_path, bad_prefix]
    if m["needs_species"]:
        args += ["--species-csv", SPECIES_CSV]
    proc = run_bounded(args, cwd=FOF_UNIX_DIR, timeout=TIMEOUT_HARNESS_RUN_S)
    assert proc.returncode != 0


# ===========================================================================
# Row 14 — over-long field (audit finding #3: never silently truncated)
# ===========================================================================

def test_row14_overlong_field_consume_region_size20(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[CONSUME_HEADER.index("region")] = "X" * 25  # cr_Region[20]
    res = _run("consume", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


def test_row14_overlong_field_consume_cover_group_size50(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[CONSUME_HEADER.index("cover_group")] = "X" * 55  # cr_CoverGroup[50]
    res = _run("consume", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


def test_row14_overlong_field_consume_cover_class_size1000(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[CONSUME_HEADER.index("cover_class")] = "X" * 1005  # cr_CoverClass[1000]
    res = _run("consume", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


def test_row14_overlong_field_consume_batch_equ_size25(tmp_path):
    row = list(CONSUME_ROW_OK)
    row[CONSUME_HEADER.index("batch_equ")] = "X" * 30  # cr_BatchEqu[25]
    res = _run("consume", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


def test_row14_overlong_field_mortality_fire_severity_size10(tmp_path):
    row = list(MORTALITY_ROW_OK)
    row[MORTALITY_HEADER.index("fire_severity")] = "X" * 15  # cr_FirSev[10]
    res = _run("mortality", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


def test_row14_overlong_field_mortality_species_size20(tmp_path):
    row = list(MORTALITY_ROW_OK)
    row[MORTALITY_HEADER.index("species")] = "X" * 25  # cr_Spe[20]
    res = _run("mortality", [row], tmp_path)
    assert res.returncode != 0
    assert "buffer capacity" in res.stderr


# ===========================================================================
# Row 15 — repeat within ONE process; row 16 — repeat across fresh processes
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row15_same_process_multi_row_matches_isolated_fresh_process(mode, tmp_path):
    """Genuinely tests same-PROCESS state isolation: TWO scientifically
    distinct rows (different species/equation/values, not just case_id) in
    ONE invocation, cross-checked against each row run alone in its OWN
    isolated fresh process. If row 1's execution left any state behind
    that leaked into row 2 (or vice versa), the combined-run result for at
    least one row would differ from that row's isolated result."""
    m = MODES[mode]
    row_a = list(m["row"])
    row_b = list(SECOND_ROW_OK[mode])

    combined = _run(mode, [row_a, row_b], tmp_path, name="combined")
    isolated_a = _run(mode, [row_a], tmp_path, name="isolated_a")
    isolated_b = _run(mode, [row_b], tmp_path, name="isolated_b")
    assert combined.returncode == isolated_a.returncode == isolated_b.returncode == 0

    combined_rows = {r["case_id"]: r for r in combined.rows(m["primary_suffix"])}
    isolated_rows = {}
    isolated_rows.update({r["case_id"]: r for r in isolated_a.rows(m["primary_suffix"])})
    isolated_rows.update({r["case_id"]: r for r in isolated_b.rows(m["primary_suffix"])})
    # input_sha256 legitimately differs between the two invocation shapes
    # only in the sense that it's per-row content-derived, not
    # position-derived, so it should still match; compare every field.
    assert combined_rows == isolated_rows


@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row16_repeat_fresh_process_byte_identical(mode, tmp_path):
    m = MODES[mode]
    r1 = _run(mode, [m["row"]], tmp_path, name="p1")
    r2 = _run(mode, [m["row"]], tmp_path, name="p2")
    assert r1.returncode == r2.returncode == 0
    for suffix in m["suffixes"]:
        p1 = os.path.join(str(tmp_path), "p1" + suffix + ".csv")
        p2 = os.path.join(str(tmp_path), "p2" + suffix + ".csv")
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            assert f1.read() == f2.read()


# ===========================================================================
# Row 17 — row order permuted; per-case_id results unchanged.
#
# Uses SECOND_ROW_OK (a scientifically distinct row — different species/
# equation/values), not merely a second case_id: swapping two rows that
# differ only in case_id can never reveal order-dependent state, since
# every other field (and therefore every computed value) would be
# identical regardless of order.
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row17_independent_row_permutation(mode, tmp_path):
    m = MODES[mode]
    row_a = list(m["row"])
    row_b = list(SECOND_ROW_OK[mode])
    forward = _run(mode, [row_a, row_b], tmp_path, name="forward")
    reversed_ = _run(mode, [row_b, row_a], tmp_path, name="reversed")
    assert forward.returncode == reversed_.returncode == 0
    fwd_by_id = {r["case_id"]: r for r in forward.rows(m["primary_suffix"])}
    rev_by_id = {r["case_id"]: r for r in reversed_.rows(m["primary_suffix"])}
    assert fwd_by_id == rev_by_id


def test_row17_canopy_cover_group_block_permutation(tmp_path):
    rows_forward = [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "0", "s1", "PIPO", "14", "65"],
        ["c3", "0", "s2", "PIPO", "10", "50"],
    ]
    rows_swapped = [
        ["c3", "0", "s2", "PIPO", "10", "50"],
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "0", "s1", "PIPO", "14", "65"],
    ]
    fwd = _run("canopy_cover", rows_forward, tmp_path, name="fwd")
    swp = _run("canopy_cover", rows_swapped, tmp_path, name="swp")
    assert fwd.returncode == swp.returncode == 0
    fwd_trees = {r["case_id"]: r for r in fwd.rows("_trees")}
    swp_trees = {r["case_id"]: r for r in swp.rows("_trees")}
    assert fwd_trees == swp_trees
    fwd_stands = {r["stand_id"]: r["total_area_ft2"] for r in fwd.rows("_stands")}
    swp_stands = {r["stand_id"]: r["total_area_ft2"] for r in swp.rows("_stands")}
    assert fwd_stands == swp_stands


# ===========================================================================
# Row 18 — species-driven mode run without a successfully loaded FOF_SPP.CSV
# ===========================================================================

@pytest.mark.parametrize("mode", ["mortality", "bark_thick", "canopy_cover"])
def test_row18_species_mode_without_species_csv(mode, tmp_path):
    m = MODES[mode]
    res = run_harness(mode, m["header"], [m["row"]],
                       os.path.join(str(tmp_path), "case"),
                       output_suffixes=m["suffixes"])  # no species_csv kwarg
    assert res.returncode != 0
    assert res.output_files == {} or all(v == [] for v in res.output_files.values())


@pytest.mark.parametrize("mode", ["mortality", "bark_thick", "canopy_cover"])
def test_row18_species_mode_with_invalid_species_csv(mode, tmp_path):
    m = MODES[mode]
    bogus = os.path.join(str(tmp_path), "does_not_exist.csv")
    res = run_harness(mode, m["header"], [m["row"]],
                       os.path.join(str(tmp_path), "case"),
                       species_csv=bogus, output_suffixes=m["suffixes"])
    assert res.returncode != 0


# ===========================================================================
# Row 19 — canopy_cover grouped-mode contiguity violation
# ===========================================================================

def test_row19_canopy_cover_noncontiguous_stand(tmp_path):
    rows = [
        ["c1", "0", "s1", "PSME", "12", "60"],
        ["c2", "0", "s2", "PSME", "12", "60"],
        ["c3", "0", "s1", "PSME", "12", "60"],  # s1 reappears -> violation
    ]
    res = _run("canopy_cover", rows, tmp_path)
    assert res.returncode != 0
    assert "s1" in res.stderr


# ===========================================================================
# consume §2b — expanded-emissions qualification: a successful return with
# all-zero expanded emissions is a hard failure, not a pass.
# ===========================================================================

def test_consume_expanded_emissions_are_nonzero(tmp_path):
    res = _run("consume", [CONSUME_ROW_OK], tmp_path)
    assert res.returncode == 0
    row = res.rows("_summary")[0]
    assert row["outcome"] == "ok"
    factor_fields = [
        "PM25F", "PM25S", "PM10F", "PM10S", "CH4F", "CH4S", "COF", "COS",
        "CO2F", "CO2S", "NOXF", "NOXS", "SO2F", "SO2S",
    ]
    values = [float(row[k]) for k in factor_fields]
    assert any(v != 0.0 for v in values), (
        "consume returned outcome=ok with an all-zero expanded-emissions "
        "block — a hard failure per gate0/05-harness-contract.md §2b, not "
        "a legitimate zero-emission scientific result"
    )


# ===========================================================================
# CLI contract (Phase 2 amendment): --species-csv required/rejected per mode
# ===========================================================================

@pytest.mark.parametrize("mode", ["consume", "litter_eq", "shrub_herb_eq"])
def test_cli_species_csv_rejected_for_non_species_modes(mode, tmp_path):
    m = MODES[mode]
    res = run_harness(mode, m["header"], [m["row"]],
                       os.path.join(str(tmp_path), "case"),
                       species_csv=SPECIES_CSV, output_suffixes=m["suffixes"])
    assert res.returncode != 0
    assert "does not accept" in res.stderr


def test_cli_duplicate_species_csv_rejected(tmp_path):
    res = run_harness(
        "mortality", MORTALITY_HEADER, [MORTALITY_ROW_OK],
        os.path.join(str(tmp_path), "case"),
        extra_args=["--species-csv", SPECIES_CSV, "--species-csv", SPECIES_CSV],
        output_suffixes=("",),
    )
    assert res.returncode != 0
    assert "more than once" in res.stderr


def test_cli_unknown_option_rejected(tmp_path):
    res = run_harness(
        "litter_eq", LITTER_EQ_HEADER, [LITTER_EQ_ROW_OK],
        os.path.join(str(tmp_path), "case"),
        extra_args=["--not-a-real-flag"], output_suffixes=("",),
    )
    assert res.returncode != 0
    assert "unknown option" in res.stderr


# ===========================================================================
# Audit finding #5 — SHA-256 known-vector self-test
# ===========================================================================

def test_sha256_known_vectors_and_independent_file_cross_check(tmp_path):
    import hashlib

    proc = run_bounded(
        [HARNESS_EXE, "--selftest-sha256", SPECIES_CSV],
        timeout=TIMEOUT_HARNESS_RUN_S,
    )
    assert proc.returncode == 0, proc.stderr
    lines = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    assert lines["SHA256_EMPTY"] == hashlib.sha256(b"").hexdigest()
    assert lines["SHA256_ABC"] == hashlib.sha256(b"abc").hexdigest()
    with open(SPECIES_CSV, "rb") as f:
        expected_file_hash = hashlib.sha256(f.read()).hexdigest()
    assert lines["SHA256_FILE"] == expected_file_hash


# ===========================================================================
# Phase 2 correction item 9 — FOFEM_TEST_HARNESS_EXE diagnostic-binary
# override is validated, never silently ignored.
# ===========================================================================

def test_harness_exe_override_rejects_a_nonexistent_path(monkeypatch, tmp_path):
    import tests.cpp_parity_live._harness_support as hs

    bogus = os.path.join(str(tmp_path), "does_not_exist.exe")
    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, bogus)
    with pytest.raises(hs.HarnessConfigError) as exc_info:
        hs.resolve_harness_exe()
    assert repr(bogus) in str(exc_info.value)


def test_harness_exe_override_is_used_when_valid(monkeypatch, tmp_path):
    """A valid override must actually be the path run_harness() invokes —
    proven by pointing it at a harmless stand-in exe (not the real
    fofem_test) and observing that stand-in run, not the default binary."""
    import tests.cpp_parity_live._harness_support as hs

    stand_in = os.path.join(str(tmp_path), "stand_in.exe")
    with open(stand_in, "w") as f:
        f.write("not a real PE binary, only os.path.isfile matters here")

    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, stand_in)
    assert hs.resolve_harness_exe() == stand_in


def test_harness_exe_override_unset_resolves_to_default(monkeypatch):
    """Round 4 correction item 6: this test previously asserted the
    override was unset unconditionally, which is false (and was reported
    as "1 failed by design") whenever a caller — including a deliberate
    ASan-diagnostic-binary qualification run — exports
    FOFEM_TEST_HARNESS_EXE for the whole process. A diagnostic
    qualification gate may not contain an intentional failure, so this
    now explicitly removes the override for the scope of this one test
    regardless of what the surrounding process environment has set."""
    import tests.cpp_parity_live._harness_support as hs

    monkeypatch.delenv(HARNESS_EXE_OVERRIDE_ENV_VAR, raising=False)
    assert hs.resolve_harness_exe() == HARNESS_EXE


# ===========================================================================
# Phase 2 correction item 5 — stale/partial output can never be mistaken
# for a completed run.
# ===========================================================================

def test_run_harness_removes_stale_pre_existing_output(tmp_path):
    """A stale output file left over at the same out_prefix from an
    earlier run must be gone after run_harness() — even for a call whose
    OWN input is malformed enough that the harness process itself never
    touches that path, proving the removal is run_harness()'s own
    pre-invocation step, not incidental to what the process does."""
    prefix = os.path.join(str(tmp_path), "case")
    out_path = prefix + ".csv"
    with open(out_path, "w") as f:
        f.write("stale,content\n")

    m = MODES["litter_eq"]
    res = run_harness(
        "litter_eq", m["header"], [m["row"]], prefix,
        output_suffixes=("",), magic_override="not-a-real-magic-line",
    )
    assert res.returncode != 0
    assert not os.path.isfile(out_path), "stale pre-existing output survived run_harness()"


def test_run_harness_removes_partial_output_after_timeout(monkeypatch, tmp_path):
    """If the underlying process is killed for exceeding its timeout after
    already writing a partial output file, run_harness() must remove that
    file — a partial file surviving a timeout could otherwise be mistaken
    for a completed run's real output."""
    import tests.cpp_parity_live._harness_support as hs

    prefix = os.path.join(str(tmp_path), "case")
    out_path = prefix + ".csv"

    def _fake_run_bounded(args, **kwargs):
        with open(out_path, "w") as f:
            f.write("partial,garbage\n")
        raise hs.ProcTimeout("simulated timeout after a partial write")

    monkeypatch.setattr(hs, "run_bounded", _fake_run_bounded)
    m = MODES["litter_eq"]
    with pytest.raises(hs.HarnessTimeout):
        hs.run_harness(
            "litter_eq", m["header"], [m["row"]], prefix, output_suffixes=("",),
        )
    assert not os.path.isfile(out_path), "partial output survived a timeout"


# ===========================================================================
# Normalized string handling / hash identity (item 4 fix): every field is
# trimmed exactly once at read time and that same trimmed value is used
# for BOTH execution and input_sha256 — so two rows differing only in
# incidental whitespace must both hash identically AND execute
# identically (not hash-identically while executing differently, which was
# the bug: a string field with untrimmed whitespace was previously copied
# as-is into the executed C struct while the hash used a locally trimmed
# copy).
# ===========================================================================

def test_normalized_whitespace_hashes_identically_to_trimmed(tmp_path):
    # Same case_id in both (input_sha256 is computed over the WHOLE
    # normalized row, including case_id, so it must be held constant here
    # — only isolated single-row runs, never together, since case_id must
    # be unique within one input file).
    clean_row = list(MORTALITY_ROW_OK)
    whitespace_row = list(MORTALITY_ROW_OK)
    whitespace_row[2] = "  PSME  "  # species, with incidental whitespace

    res_clean = _run("mortality", [clean_row], tmp_path, name="clean")
    res_ws = _run("mortality", [whitespace_row], tmp_path, name="whitespace")
    assert res_clean.returncode == res_ws.returncode == 0
    clean_row_out = res_clean.rows()[0]
    ws_row_out = res_ws.rows()[0]

    # Executes identically: same species resolves, same probability/equ.
    assert clean_row_out["outcome"] == ws_row_out["outcome"] == "ok"
    assert clean_row_out["prob"] == ws_row_out["prob"]
    assert clean_row_out["mort_equ"] == ws_row_out["mort_equ"]

    # Hashes identically too: whitespace is normalized before hashing, the
    # same as it is before execution.
    assert clean_row_out["input_sha256"] == ws_row_out["input_sha256"]


def test_normalized_whitespace_species_still_rejects_when_actually_unknown(tmp_path):
    # Whitespace normalization must not become an accidental laxness that
    # makes an otherwise-invalid species resolve.
    row = list(MORTALITY_ROW_OK)
    row[1] = "1"
    row[2] = "  ZZZZZZ  "
    res = _run("mortality", [row], tmp_path)
    assert res.returncode == 0
    assert res.rows()[0]["outcome"] == "expected_model_error"


# ===========================================================================
# Mortality schema v2 — `density_tpa` boundaries and the corrected error rule
#
# Contract rows 5/6/7/8 for the NEW column, plus the error-signal change the
# column exists to expose. `density_tpa` fills d_MIS.f_Den, which ValidInput
# checks with `f_Den < 1.0 || f_Den > 20000` (fof_mrt.cpp:1854-1856), so 1
# and 20000 are the inclusive valid boundaries and 0 / 20001 are the first
# invalid values on each side.
#
# Only the CroDam (PFI_Calc) route validates density at all: MRT_CalcMngr
# (fof_mrt.cpp:157-187) sends CroSco to MRT_Calc and BolCha to BC_Calc,
# neither of which calls ValidInput. These tests therefore drive CroDam
# rows, and the CroSco insensitivity is asserted explicitly rather than
# assumed.
# ===========================================================================

#: Index of the mortality mode's `density_tpa` column.
DENSITY_INDEX = MORTALITY_HEADER.index("density_tpa")

#: A CroDam row that routes through PFI_Calc -> ValidInput. ABCO's `Mort`
#: entry in the tracked FOF_SPP.CSV selects equation WF (Eq_WhiteFir_WF,
#: fof_mrt.cpp:1911-1942), whose required fields are "dbh len ckr btl".
MORTALITY_CRODAM_ROW = [
    "d1", "0", "ABCO", "CroDam", "12", "60", "50", "0", "Scorch", "0",
    "NA", "3", "100", "0", "100",
]


def _crodam_row(case_id, density, expect_error="0"):
    row = list(MORTALITY_CRODAM_ROW)
    row[0] = case_id
    row[1] = expect_error
    row[DENSITY_INDEX] = density
    return row


@pytest.mark.parametrize("density", ["1", "20000", "100"])
def test_mortality_density_within_bounds_is_accepted(density, tmp_path):
    """1 and 20000 are ValidInput's inclusive boundaries; a mid-range value
    is included as the control."""
    res = _run("mortality", [_crodam_row("d_ok", density)], tmp_path,
               name=f"case_d{density}")
    assert res.returncode == 0, res.stderr
    row = res.rows("")[0]
    assert row["outcome"] == "ok"
    assert row["err_text"].strip() == ""
    assert 0.0 <= float(row["prob"]) <= 1.0


@pytest.mark.parametrize("density", ["0", "0.999", "20001"])
def test_mortality_density_out_of_bounds_is_a_model_error(density, tmp_path):
    """0 and 20001 are the first invalid values on each side; 0.999 pins that
    the lower bound is a real `< 1.0` float comparison, not an integer one.

    The row declares expect_error=1, so the two-sided rule requires the
    harness to actually observe the error — a silently-successful row would
    be an `unexpected_failure` and a nonzero exit.
    """
    res = _run("mortality",
               [_crodam_row("d_bad", density, expect_error="1")], tmp_path,
               name=f"case_d{density}")
    assert res.returncode == 0, res.stderr
    row = res.rows("")[0]
    assert row["outcome"] == "expected_model_error"
    assert "Invalid input: Density" in row["err_text"]
    assert row["ret"] == "-1"
    assert row["prob"] == "NA"


@pytest.mark.parametrize("density", ["0", "20001"])
def test_mortality_out_of_bounds_density_without_expect_error_fails(
        density, tmp_path):
    """The same rejection with expect_error=0 must exit nonzero and be
    recorded `unexpected_failure`.

    This is the regression guard for the schema-v1 defect: under the old
    `prob < 0` error rule the row was written `ok` with `prob=0.000000`
    despite carrying ValidInput's error text, and the process exited 0.
    """
    res = _run("mortality", [_crodam_row("d_bad", density)], tmp_path,
               name=f"case_dnx{density}")
    assert res.returncode != 0
    row = res.rows("")[0]
    assert row["outcome"] == "unexpected_failure"
    assert "Invalid input: Density" in row["err_text"]


@pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "1e", "0x10"])
def test_mortality_density_parsing_is_strict(bad, tmp_path):
    """Rows 5/6 for `density_tpa`: blank and non-numeric are hard errors,
    never a silent 0.0 (which is exactly the value that made the schema-v1
    defect invisible)."""
    res = _run("mortality", [_crodam_row("d_parse", bad)], tmp_path,
               name=f"case_dp{bad!r}")
    assert res.returncode != 0
    assert "density_tpa" in res.stderr


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", "-1e999"])
def test_mortality_density_nonfinite_is_rejected(bad, tmp_path):
    """Row 7 for `density_tpa`."""
    res = _run("mortality", [_crodam_row("d_nf", bad)], tmp_path,
               name=f"case_dnf{bad!r}")
    assert res.returncode != 0
    assert "density_tpa" in res.stderr


@pytest.mark.parametrize("density", ["1", "20000"])
def test_mortality_crosco_probability_is_independent_of_density(
        density, tmp_path):
    """CroSco routes to MRT_Calc, which never calls ValidInput; density only
    reaches MRT_Total's stand accumulators (fof_mrt.cpp:818-875), which this
    mode does not emit. Asserted, not assumed — it is what makes the Phase 2
    CroSco golden's `prob` unchanged by the v2 column."""
    row = list(MODES["mortality"]["row"])
    row[DENSITY_INDEX] = density
    res = _run("mortality", [row], tmp_path, name=f"case_cs{density}")
    assert res.returncode == 0, res.stderr
    assert res.rows("")[0]["prob"] == "0.976129"


def test_mortality_error_text_alone_marks_a_model_error(tmp_path):
    """The v2 error rule's other half: an error signalled ONLY through
    cr_ErrMes (no negative return) is a model error.

    PFI_Calc returns 0 — a perfectly ordinary probability — on a ValidInput
    rejection (fof_mrt.cpp:1800-1801), so `prob < 0` alone cannot see it.
    """
    res = _run("mortality",
               [_crodam_row("d_errtext", "0", expect_error="1")], tmp_path)
    assert res.returncode == 0, res.stderr
    row = res.rows("")[0]
    assert row["outcome"] == "expected_model_error"
    assert row["err_text"].strip() != ""


# ===========================================================================
# Mode: soil_campbell (Phase 5, gate0/05-harness-contract.md section 7)
#
# soil_campbell is deliberately NOT added to the shared MODES/ALL_MODE_NAMES/
# NUMERIC_FIELD_INDEX/SECOND_ROW_OK structures above: MODES is imported by
# generate_phase2_goldens.py and iterated at MODULE IMPORT TIME to build
# GOLDEN_TOLERANCE_KEYS via PHASE2_CANONICAL_ROUTE_KEYS[mode] (a fixed dict
# in _output_contract.py scoped to the six Phase 2 modes) -- adding
# "soil_campbell" there was verified (by direct import) to raise a hard
# KeyError at collection time, breaking every test that transitively imports
# generate_phase2_goldens.py (including test_tolerance_policy_completeness.py
# and this module's own generator-driver test). Golden generation and
# tolerance-policy work for soil_campbell are explicitly deferred (Phase 5
# items 4-9), so this mode gets its own small, self-contained set of
# constants/helpers below instead, applying the same row-1..19 self-test
# discipline directly rather than through the shared parametrization.
# ===========================================================================

SOIL_CAMPBELL_HEADER = [
    "case_id", "expect_error", "brn_ignited", "soil_type", "moist_cond",
    "duff_dep_pre_in", "duff_dep_pos_in", "soil_moist_pct",
    "wl_efficiency", "hs_efficiency", "n_steps",
    "fi_series_path", "fi_hs_series_path",
    "duff_load_tac", "duff_consumed_pct", "duff_moist_pct",
]
SOIL_CAMPBELL_SUFFIXES = ("_summary", "_field")
#: F-70 diagnostic pass: includes the opt-in "_soidiag" file. Passed
#: explicitly (not the default) so ordinary soil_campbell tests above are
#: unaffected -- the diagnostic file is absent unless requested via
#: FOFEM_TEST_SOIL_DIAG, and its absence must not itself be an error.
SOIL_CAMPBELL_DIAG_SUFFIXES = ("_summary", "_field", "_soidiag")
SOIL_CAMPBELL_N_STEPS = 20
SOIL_CAMPBELL_FI_WL_NAME = "fi_wl.csv"
SOIL_CAMPBELL_FI_HS_NAME = "fi_hs.csv"


def _soil_zduff_row(case_id="c1", expect_error="0", soil_type="Fine-Silt",
                     moist_cond="Dry", soil_moist_pct="10",
                     wl_name=SOIL_CAMPBELL_FI_WL_NAME,
                     hs_name=SOIL_CAMPBELL_FI_HS_NAME,
                     n_steps=SOIL_CAMPBELL_N_STEPS):
    """duff_dep_pre_in=0 selects the ZDuff (SE_Mngr_Array) route."""
    return [case_id, expect_error, "YES", soil_type, moist_cond,
            "0", "0", soil_moist_pct, "-1", "-1", str(n_steps),
            wl_name, hs_name, "0", "-1", "0"]


def _soil_duff_row(case_id="c3", expect_error="0",
                    wl_name=SOIL_CAMPBELL_FI_WL_NAME,
                    hs_name=SOIL_CAMPBELL_FI_HS_NAME,
                    n_steps=SOIL_CAMPBELL_N_STEPS):
    """duff_dep_pre_in=2 (>0) selects the Duff (SD_Mngr_New) route. The
    trailing three columns (duff_load_tac, duff_consumed_pct,
    duff_moist_pct) are the Phase 5 item-1 audit's schema-gap addition --
    see test_harness.cpp's soil_campbell header comment and
    _golden_manifest.MODE_SCHEMA_VERSIONS's soil_campbell docstring."""
    return [case_id, expect_error, "YES", "Fine-Silt", "Dry",
            "2", "1", "10", "-1", "-1", str(n_steps),
            wl_name, hs_name, "5", "50", "60"]


SOIL_CAMPBELL_ROW_OK = _soil_zduff_row()

#: A SECOND, scientifically distinct valid row (different soil type /
#: moisture condition / soil moisture) -- mirrors SECOND_ROW_OK's role for
#: the other six modes: needed for the same-process multi-row isolation and
#: order-dependent-state tests below, since changing only case_id cannot
#: reveal state that depends on the actual computed values.
SOIL_CAMPBELL_SECOND_ROW_OK = _soil_zduff_row(
    case_id="c2", soil_type="Loamy-Skeletal", moist_cond="Wet",
    soil_moist_pct="20")


def _write_soil_side_files(tmp_path, n_steps=SOIL_CAMPBELL_N_STEPS,
                            wl_name=SOIL_CAMPBELL_FI_WL_NAME,
                            hs_name=SOIL_CAMPBELL_FI_HS_NAME,
                            wl_values=None, hs_values=None):
    """Write the two default fire-intensity side files into *tmp_path* if
    not already present. Deliberately short and clearly decaying to zero:
    neither SD_Mngr_New nor SE_Mngr_Array's stepping loop has a hard
    iteration cap independent of SHA_Get's eC_Tim(10000) table bound (see
    test_harness.cpp's soil_campbell header comment), so a qualification
    row must be constructed to terminate on its own rather than relying on
    an artificial cap."""
    wl_path = os.path.join(str(tmp_path), wl_name)
    hs_path = os.path.join(str(tmp_path), hs_name)
    if wl_values is None:
        wl_values = [max(0.0, 50.0 - i * 3.0) for i in range(n_steps)]
    if hs_values is None:
        hs_values = [max(0.0, 10.0 - i * 0.5) for i in range(n_steps)]
    if not os.path.isfile(wl_path):
        with open(wl_path, "w", newline="\n") as f:
            f.write("\n".join(str(v) for v in wl_values) + "\n")
    if not os.path.isfile(hs_path):
        with open(hs_path, "w", newline="\n") as f:
            f.write("\n".join(str(v) for v in hs_values) + "\n")


def _run_soil(rows, tmp_path, name="case", write_side_files=True, **kwargs):
    if write_side_files:
        _write_soil_side_files(tmp_path)
    kwargs.setdefault("output_suffixes", SOIL_CAMPBELL_SUFFIXES)
    return run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, rows,
        os.path.join(str(tmp_path), name), **kwargs,
    )


def test_soil_row1_valid_all_rows_ok(tmp_path):
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_summary")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["model"] == "Zero-Duff"


def test_soil_row2_missing_magic_line(tmp_path):
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, magic_override="")
    assert res.returncode != 0
    assert not res.rows("_summary")


def test_soil_row3_wrong_schema_version(tmp_path):
    """soil_campbell is schema v2 (bumped for the duff_burn_* columns added
    by the Campbell duff-forcing correction pass -- see
    _golden_manifest.MODE_SCHEMA_VERSIONS's soil_campbell docstring); its
    own stale v1 must not be silently accepted."""
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, schema_version="1")
    assert res.returncode != 0


def test_soil_row3_declared_schema_version_is_accepted(tmp_path):
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, schema_version="2")
    assert res.returncode == 0, res.stderr
    assert res.rows("_summary")[0]["schema_version"] == "2"


def test_soil_row3_unknown_schema_version_is_rejected(tmp_path):
    """soil_campbell (v2) and mortality (v2) now coincidentally share the
    same schema_version STRING, so that string is no longer usable as a
    cross-mode-confusion probe here; a genuinely unknown version (never
    declared by any mode in MODE_SCHEMA_VERSIONS) is rejected instead."""
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, schema_version="99")
    assert res.returncode != 0


def test_soil_row4_column_removed(tmp_path):
    header = SOIL_CAMPBELL_HEADER[:-1]
    res = _run_soil([SOIL_CAMPBELL_ROW_OK[:-1]], tmp_path, header_override=header)
    assert res.returncode != 0


def test_soil_row4_column_added(tmp_path):
    header = SOIL_CAMPBELL_HEADER + ["extra_col"]
    row = SOIL_CAMPBELL_ROW_OK + ["0"]
    res = _run_soil([row], tmp_path, header_override=header)
    assert res.returncode != 0


def test_soil_row4_column_reordered(tmp_path):
    header = list(SOIL_CAMPBELL_HEADER)
    header[0], header[1] = header[1], header[0]
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, header_override=header)
    assert res.returncode != 0


def test_soil_row4_column_duplicated(tmp_path):
    header = list(SOIL_CAMPBELL_HEADER) + [SOIL_CAMPBELL_HEADER[-1]]
    row = list(SOIL_CAMPBELL_ROW_OK) + [SOIL_CAMPBELL_ROW_OK[-1]]
    res = _run_soil([row], tmp_path, header_override=header)
    assert res.returncode != 0


def test_soil_row5_blank_numeric_field(tmp_path):
    row = list(SOIL_CAMPBELL_ROW_OK)
    row[7] = ""  # soil_moist_pct
    res = _run_soil([row], tmp_path)
    assert res.returncode != 0
    assert "0.0" not in res.stdout  # never silently defaulted to 0.0


@pytest.mark.parametrize("bad", ["abc", "1.2.3", "1e", "0x10"])
def test_soil_row6_non_numeric_field(bad, tmp_path):
    row = list(SOIL_CAMPBELL_ROW_OK)
    row[7] = bad
    res = _run_soil([row], tmp_path, name=f"case_{bad!r}")
    assert res.returncode != 0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999", "-1e999"])
def test_soil_row7_nan_inf_and_overflow(bad, tmp_path):
    row = list(SOIL_CAMPBELL_ROW_OK)
    row[7] = bad
    res = _run_soil([row], tmp_path, name=f"case_{bad!r}")
    assert res.returncode != 0


@pytest.mark.parametrize("bad", ["0", "-1", "6001", "999999"])
def test_soil_row8_n_steps_out_of_domain(bad, tmp_path):
    """n_steps is a harness-only bookkeeping value bounded to [1, eC_sfi]
    (eC_sfi=6000, fof_co.h:222) -- not a real d_SI field, but still subject
    to self-test row 8's "value out of the field's documented range"."""
    row = list(SOIL_CAMPBELL_ROW_OK)
    row[10] = bad
    res = _run_soil([row], tmp_path, name=f"case_n{bad}", write_side_files=False)
    assert res.returncode != 0


def test_soil_row9_duplicate_case_id(tmp_path):
    res = _run_soil([SOIL_CAMPBELL_ROW_OK, SOIL_CAMPBELL_ROW_OK], tmp_path)
    assert res.returncode != 0


def test_soil_row10_empty_file(tmp_path):
    res = _run_soil([], tmp_path, magic_override="")
    assert res.returncode != 0


def test_soil_row10_header_only(tmp_path):
    res = _run_soil([], tmp_path)
    assert res.returncode != 0


def test_soil_row11c_variable_fanout_expected_error_zero_field_rows(tmp_path):
    """A row expected to error must contribute zero field rows -- the
    variable-fan-out counterpart of consume's row11b constant-fan-out
    check (harness-contract section 1 fan-out reconciliation table,
    section 7's explicit "expected field-row count is therefore zero")."""
    bad = _soil_zduff_row(case_id="bad1", expect_error="1", soil_type="NotARealSoil")
    ok = _soil_zduff_row(case_id="ok1")
    res = _run_soil([bad, ok], tmp_path)
    assert res.returncode == 0, res.stderr
    summary = {r["case_id"]: r for r in res.rows("_summary")}
    assert summary["bad1"]["outcome"] == "expected_model_error"
    assert summary["bad1"]["n_layers"] == "NA"
    assert summary["bad1"]["n_time_indices"] == "NA"
    field = res.rows("_field")
    assert all(r["case_id"] != "bad1" for r in field)
    assert any(r["case_id"] == "ok1" for r in field)


def test_soil_row11e_two_sided_unexpectedly_succeeds(tmp_path):
    row = _soil_zduff_row(expect_error="1")  # a normal, succeeding row
    res = _run_soil([row], tmp_path)
    assert res.returncode != 0
    assert res.rows("_summary")[0]["outcome"] == "unexpected_failure"


def test_soil_row12_row_unexpectedly_errors(tmp_path):
    row = _soil_zduff_row(soil_type="NotARealSoil")  # expect_error=0
    res = _run_soil([row], tmp_path)
    assert res.returncode != 0
    assert res.rows("_summary")[0]["outcome"] == "unexpected_failure"


def test_soil_row13_output_path_unwritable(tmp_path):
    # The input file (and its side files) must exist and be readable; only
    # the OUTPUT prefix's directory is missing -- mirrors the generic
    # test_row13_output_path_unwritable's approach of writing the input CSV
    # directly and invoking the binary, since run_harness() itself would
    # otherwise fail earlier trying to write a nonexistent input path.
    _write_soil_side_files(tmp_path)
    in_path = os.path.join(str(tmp_path), "case_in.csv")
    with open(in_path, "w", newline="\n") as f:
        f.write("#fofem-harness,soil_campbell,1\n")
        f.write(",".join(SOIL_CAMPBELL_HEADER) + "\n")
        f.write(",".join(SOIL_CAMPBELL_ROW_OK) + "\n")
    bad_prefix = os.path.join(str(tmp_path), "no_such_dir", "deeper", "case")
    proc = run_bounded([HARNESS_EXE, in_path, bad_prefix], cwd=FOF_UNIX_DIR,
                        timeout=TIMEOUT_HARNESS_RUN_S)
    assert proc.returncode != 0


def test_soil_row14_overlong_soil_type_field(tmp_path):
    """d_SI.cr_SoilType is char[30] (eC_SoilType, fof_sh.h:84); a 40-char
    value must be rejected fail-closed (safe_copy), never truncated."""
    row = _soil_zduff_row(soil_type="X" * 40)
    res = _run_soil([row], tmp_path)
    assert res.returncode != 0
    assert "not truncated" in res.stderr or "capacity" in res.stderr


def test_soil_row15_same_process_multi_row_matches_isolated_fresh_process(tmp_path):
    row_a = list(SOIL_CAMPBELL_ROW_OK)
    row_b = list(SOIL_CAMPBELL_SECOND_ROW_OK)
    multi = _run_soil([row_a, row_b], tmp_path, name="multi")
    assert multi.returncode == 0, multi.stderr
    iso_a = _run_soil([row_a], tmp_path, name="iso_a")
    iso_b = _run_soil([row_b], tmp_path, name="iso_b")
    assert iso_a.returncode == 0 and iso_b.returncode == 0

    def scientific(row):
        return {k: v for k, v in row.items() if k not in ("case_id", "input_sha256")}

    multi_by_id = {r["case_id"]: r for r in multi.rows("_summary")}
    assert scientific(multi_by_id["c1"]) == scientific(iso_a.rows("_summary")[0])
    assert scientific(multi_by_id["c2"]) == scientific(iso_b.rows("_summary")[0])
    # This is the mandatory fof_soi.cpp state-hazard proof (crosswalk F-E,
    # harness-contract section 7 "State hazard"): fof_soi.cpp keeps
    # file-scope statics (rr_w/rr_t/rr_p/r_bd/r_m/... -- see the Phase 5
    # report's item-1 audit) that soiltemp_initconsts()/initprofile() only
    # get reset by being called again at the TOP of SD_Mngr_New/
    # SE_Mngr_Array on every SH_Mngr invocation. If that reset were ever
    # skipped or partial for a second row in the same process, row c2's
    # temperature trajectory would silently inherit row c1's leftover
    # profile/constants and this equality would fail while the isolated
    # fresh-process runs (which cannot share any state) would still be
    # correct -- exactly the class of bug this comparison is built to catch.


def test_soil_row16_repeat_fresh_process_byte_identical(tmp_path):
    first = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, name="rep1")
    second = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, name="rep2")
    assert first.returncode == second.returncode == 0
    assert first.rows("_summary") == [
        {**r, "case_id": r["case_id"]} for r in second.rows("_summary")
    ]
    assert first.rows("_field") == second.rows("_field")


def test_soil_row17_independent_row_permutation(tmp_path):
    row_a = list(SOIL_CAMPBELL_ROW_OK)
    row_b = list(SOIL_CAMPBELL_SECOND_ROW_OK)
    forward = _run_soil([row_a, row_b], tmp_path, name="fwd")
    backward = _run_soil([row_b, row_a], tmp_path, name="bwd")
    assert forward.returncode == backward.returncode == 0

    def by_id(res):
        return {r["case_id"]: {k: v for k, v in r.items() if k != "case_id"}
                for r in res.rows("_summary")}

    assert by_id(forward) == by_id(backward)
    # Mandatory cross-order isolation proof: the two rows differ (Fine-Silt
    # vs Loamy-Skeletal, Dry vs Wet, 10% vs 20% soil moisture, hence
    # different soiltemp_initconsts()/initprofile() inputs); if row order
    # inside one process leaked state (e.g. a stale fof_soi.cpp static from
    # whichever row ran first), c1's or c2's own result would differ between
    # the forward and backward runs even though both contain the same two
    # rows -- this assertion is exactly what would catch that.


def test_soil_valid_duff_route(tmp_path):
    res = _run_soil([_soil_duff_row()], tmp_path)
    assert res.returncode == 0, res.stderr
    row = res.rows("_summary")[0]
    assert row["outcome"] == "ok"
    assert row["model"] == "Duff"
    assert float(row["heat_frac"]) > 0.0
    # Duff-route scientific values additionally depend on duff_load_tac /
    # duff_consumed_pct / duff_moist_pct -- the Phase 5 item-1 audit's
    # schema-gap addition (see the module docstring above and the Phase 5
    # report). Route selection and reproducibility are asserted here;
    # independent numerical validation against a Python oracle is Phase 5
    # items 4-9 (deferred), not this pass.


def test_soil_valid_zduff_route(tmp_path):
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path)
    assert res.returncode == 0, res.stderr
    row = res.rows("_summary")[0]
    assert row["outcome"] == "ok"
    assert row["model"] == "Zero-Duff"
    assert row["heat_frac"] == "0.000000"  # only ever set on the Duff route


def test_soil_no_ignition(tmp_path):
    """Empirically observed, corrected from an initial source-reading
    prediction (the brief's explicit requirement -- "must be tested from
    ACTUAL SH_Mngr behavior observed by running it, not assumed from
    reading the code alone"): SH_Mngr's cr_BrnIg=="NO" branch calls
    SHA_Init_0() then returns immediately -- it NEVER reaches the `Load:`
    label / calls SO_Load() (fof_sh.cpp:51-53). So d_SO is left exactly as
    SO_Init() set it at function entry: every summary maximum genuinely
    reads 0, not 21 -- matching gate0/07-branch-traceability.csv's
    BR-SOI-NOIG "all zeros" description for the SUMMARY file specifically.
    But SHA_Init_0() DOES fill the raw rr_SHA[][] table (which this
    harness reads independently via SHA_Get, bypassing d_SO entirely) with
    a constant 21.0C (e_StaSoiTem) over 60 fake 60-second-interval steps.
    So field.csv genuinely disagrees with summary.csv here: field shows
    real 21.0 values, summary shows 0 -- both faithful reflections of two
    genuinely different pieces of real, unmodified C++ state on this one
    path, not a harness inconsistency."""
    row = _soil_zduff_row(case_id="noig")
    row[2] = "NO"  # brn_ignited
    res = _run_soil([row], tmp_path)
    assert res.returncode == 0, res.stderr
    summ = res.rows("_summary")[0]
    assert summ["outcome"] == "ok"
    assert summ["model"] == ""  # SH_Mngr never sets cr_Model on this path
    assert summ["n_time_indices"] == "60"
    for i in range(14):
        assert summ[f"lay{i:02d}_max_temp_c"] == "0"
        assert summ[f"lay{i:02d}_max_time_s"] == "0"
    assert summ["duf_pre_cm"] == "0.000000"
    assert summ["duf_post_cm"] == "0.000000"
    assert summ["heat_frac"] == "0.000000"
    assert summ["lay_max_deg1_index"] == "-1"
    assert summ["lay_max_deg2_index"] == "-1"
    field = res.rows("_field")
    assert len(field) == 14 * 60
    assert all(r["temp_c"] == "21.000000" for r in field)
    # time_s (Phase 5 correction pass item-2): SHA_Init_0() sets the
    # recording interval to a hardcoded 60 s (fof_sha.cpp:161) on this
    # no-ignition path, independent of any real solver step.
    assert all(int(r["time_s"]) == int(r["time_index"]) * 60 for r in field)


def test_soil_missing_side_file_is_rejected(tmp_path):
    row = _soil_zduff_row(wl_name="does_not_exist.csv")
    res = _run_soil([row], tmp_path)
    assert res.returncode != 0
    assert "missing or unreadable" in res.stderr


def test_soil_empty_side_file_is_rejected(tmp_path):
    _write_soil_side_files(tmp_path)
    empty_path = os.path.join(str(tmp_path), "empty.csv")
    with open(empty_path, "w"):
        pass
    row = _soil_zduff_row(wl_name="empty.csv")
    res = _run_soil([row], tmp_path, write_side_files=False)
    assert res.returncode != 0
    assert "empty" in res.stderr


@pytest.mark.parametrize("bad", ["abc", "nan", "inf", "1,2"])
def test_soil_malformed_side_file_line_is_rejected(bad, tmp_path):
    _write_soil_side_files(tmp_path)
    bad_path = os.path.join(str(tmp_path), f"bad_{abs(hash(bad))}.csv")
    with open(bad_path, "w", newline="\n") as f:
        f.write("\n".join([bad] * SOIL_CAMPBELL_N_STEPS) + "\n")
    row = _soil_zduff_row(wl_name=os.path.basename(bad_path))
    res = _run_soil([row], tmp_path, name=f"case_{abs(hash(bad))}",
                     write_side_files=False)
    assert res.returncode != 0


def test_soil_too_short_side_file_is_rejected(tmp_path):
    _write_soil_side_files(tmp_path)
    short_path = os.path.join(str(tmp_path), "short.csv")
    with open(short_path, "w", newline="\n") as f:
        f.write("\n".join(["1.0"] * (SOIL_CAMPBELL_N_STEPS - 5)) + "\n")
    row = _soil_zduff_row(wl_name="short.csv")
    res = _run_soil([row], tmp_path, write_side_files=False)
    assert res.returncode != 0
    assert "n_steps is" in res.stderr


def test_soil_too_long_side_file_is_rejected(tmp_path):
    _write_soil_side_files(tmp_path)
    long_path = os.path.join(str(tmp_path), "long.csv")
    with open(long_path, "w", newline="\n") as f:
        f.write("\n".join(["1.0"] * (SOIL_CAMPBELL_N_STEPS + 5)) + "\n")
    row = _soil_zduff_row(wl_name="long.csv")
    res = _run_soil([row], tmp_path, write_side_files=False)
    assert res.returncode != 0
    assert "n_steps is" in res.stderr


@pytest.mark.parametrize("bad_path", ["/abs/path.csv", "C:/abs/path.csv",
                                       "..\\escape.csv", "sub/../../escape.csv"])
def test_soil_unsafe_side_file_path_is_rejected(bad_path, tmp_path):
    row = _soil_zduff_row(wl_name=bad_path)
    res = _run_soil([row], tmp_path, name=f"case_{abs(hash(bad_path))}")
    assert res.returncode != 0
    assert "relative" in res.stderr


def test_soil_field_reconciliation_matches_summary_counts(tmp_path):
    """harness-contract section 7: rows(field) == sum of n_layers *
    n_time_indices(case_id) over ok rows; every (case_id, layer_index,
    time_index) triple unique, both indices dense from 0."""
    res = _run_soil([SOIL_CAMPBELL_ROW_OK, SOIL_CAMPBELL_SECOND_ROW_OK], tmp_path)
    assert res.returncode == 0, res.stderr
    summary = {r["case_id"]: r for r in res.rows("_summary")}
    field = res.rows("_field")
    seen = set()
    counts = {}
    for r in field:
        key = (r["case_id"], int(r["layer_index"]), int(r["time_index"]))
        assert key not in seen, f"duplicate field row {key}"
        seen.add(key)
        counts[r["case_id"]] = counts.get(r["case_id"], 0) + 1
    for cid, row in summary.items():
        expected = int(row["n_layers"]) * int(row["n_time_indices"])
        assert counts[cid] == expected
    for cid in summary:
        lay_idx = sorted({int(r["layer_index"]) for r in field if r["case_id"] == cid})
        assert lay_idx == list(range(int(summary[cid]["n_layers"])))
        tim_idx = sorted({int(r["time_index"]) for r in field if r["case_id"] == cid})
        assert tim_idx == list(range(int(summary[cid]["n_time_indices"])))


def test_soil_field_time_s_is_time_index_times_the_real_recording_interval(tmp_path):
    """harness-contract section 7 / Phase 5 correction pass item-2:
    field.csv's ``time_s`` column must equal ``time_index`` times the SAME
    per-row recording interval for every layer/time_index of that case_id
    (i.e. linear in time_index with no per-layer variation) -- proven
    directly from the harness's own output, not assumed. duf/zduf routes
    use different i_dt values (20 s / 10 s), so this also proves the
    interval is read per-row rather than hardcoded to one route's value."""
    res = _run_soil(
        [SOIL_CAMPBELL_ROW_OK, SOIL_CAMPBELL_SECOND_ROW_OK, _soil_duff_row()],
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    field = res.rows("_field")
    by_case: dict = {}
    for r in field:
        by_case.setdefault(r["case_id"], []).append(r)
    for cid, case_rows in by_case.items():
        times = sorted({int(r["time_index"]) for r in case_rows})
        assert times[0] == 0
        if len(times) < 2:
            continue
        interval = next(
            int(r["time_s"]) for r in case_rows if int(r["time_index"]) == 1
        )
        assert interval > 0, cid
        for r in case_rows:
            expected = int(r["time_index"]) * interval
            assert int(r["time_s"]) == expected, (cid, r)


def test_soil_normalized_whitespace_hashes_identically(tmp_path):
    # Two SEPARATE runs (not two rows in one file) sharing the identical
    # case_id: since case_id is itself part of the hashed field list, two
    # rows in the same file could never hash identically even with perfect
    # whitespace normalisation -- separate runs isolate the one thing under
    # test (incidental padding around otherwise-identical fields).
    # case_id itself is validated (parse_case_id) against the RAW field
    # before the normalisation/trim pass that produces input_sha256's
    # inputs -- padding it would be rejected outright as "case_id contains
    # a disallowed character", not silently trimmed, so it is left
    # unpadded here; every other field is padded.
    _write_soil_side_files(tmp_path)
    row_padded = [SOIL_CAMPBELL_ROW_OK[0]] + [f"  {v}  " for v in SOIL_CAMPBELL_ROW_OK[1:]]
    padded_res = _run_soil([row_padded], tmp_path, name="padded",
                            write_side_files=False)
    trimmed_res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path, name="trimmed",
                             write_side_files=False)
    assert padded_res.returncode == 0, padded_res.stderr
    assert trimmed_res.returncode == 0, trimmed_res.stderr
    padded_hash = padded_res.rows("_summary")[0]["input_sha256"]
    trimmed_hash = trimmed_res.rows("_summary")[0]["input_sha256"]
    assert padded_hash == trimmed_hash


def test_soil_side_file_hash_identity_rename_vs_content_change(tmp_path):
    """input_sha256 depends on the referenced BYTES, not the path: a
    renamed-but-identical file hashes the same; a same-named-but-different-
    content file hashes differently (harness-contract section 7).

    Uses SEPARATE runs sharing one case_id (not multiple rows in one file)
    for the same reason as the whitespace test above: case_id is itself
    part of the hashed field list, so two rows could never hash identically
    regardless of how the side-file substitution behaves."""
    _write_soil_side_files(tmp_path, wl_name="fi_a.csv")
    import shutil
    shutil.copyfile(os.path.join(str(tmp_path), "fi_a.csv"),
                     os.path.join(str(tmp_path), "fi_b.csv"))
    row_a = _soil_zduff_row(wl_name="fi_a.csv")
    row_b = _soil_zduff_row(wl_name="fi_b.csv")  # same content, renamed
    res_a = _run_soil([row_a], tmp_path, name="run_a", write_side_files=False)
    res_b = _run_soil([row_b], tmp_path, name="run_b", write_side_files=False)
    assert res_a.returncode == 0, res_a.stderr
    assert res_b.returncode == 0, res_b.stderr
    hash_a = res_a.rows("_summary")[0]["input_sha256"]
    hash_b = res_b.rows("_summary")[0]["input_sha256"]
    assert hash_a == hash_b

    diff_values = [v + 1.0 for v in
                   [max(0.0, 50.0 - i * 3.0) for i in range(SOIL_CAMPBELL_N_STEPS)]]
    with open(os.path.join(str(tmp_path), "fi_c.csv"), "w", newline="\n") as f:
        f.write("\n".join(str(v) for v in diff_values) + "\n")
    row_c = _soil_zduff_row(wl_name="fi_c.csv")  # same NAME pattern, different content
    res_c = _run_soil([row_c], tmp_path, name="run_c", write_side_files=False)
    assert res_c.returncode == 0, res_c.stderr
    hash_c = res_c.rows("_summary")[0]["input_sha256"]
    assert hash_a != hash_c


def _soil_duff_row_with_burn_inputs(case_id, load, consumed, moist, dep_pre="2"):
    """
    Build a Duff-route ``soil_campbell`` input row with explicit
    duff_load_tac/duff_consumed_pct/duff_moist_pct, for the schema-v2
    ``duff_burn_*`` self-tests below (Campbell duff-forcing correction
    pass). A local variant of :func:`_soil_duff_row` -- not
    ``_soil_campbell_contract.soil_campbell_duff_row`` -- because ``_soil_campbell_contract.py``
    itself imports ``SOIL_CAMPBELL_HEADER`` from THIS module, so importing
    it back here would be circular.

    :param case_id: Scenario identifier.
    :param load: ``duff_load_tac`` value (string).
    :param consumed: ``duff_consumed_pct`` value (string).
    :param moist: ``duff_moist_pct`` value (string).
    :param dep_pre: ``duff_dep_pre_in`` value (string), default "2".
    :return: A 16-field row list.
    """
    return [case_id, "0", "YES", "Fine-Silt", "Dry", dep_pre, "1", "10",
            "-1", "-1", str(SOIL_CAMPBELL_N_STEPS),
            SOIL_CAMPBELL_FI_WL_NAME, SOIL_CAMPBELL_FI_HS_NAME,
            load, consumed, moist]


def test_soil_duff_burn_columns_match_direct_duffburn_formula(tmp_path):
    """Schema v2 (Campbell duff-forcing correction pass): the three new
    ``duff_burn_*`` summary columns match a direct, independent hand
    transcription of the pinned C++ ``DuffBurn()`` (``bur_brn.cpp:
    1950-1986``), NOT the harness's own computation -- this is (b)
    source-relation evidence for the harness addition itself, exercised
    live. Uses the same normal moisture/load inputs as
    ``_soil_duff_row()``."""
    row = _soil_duff_row_with_burn_inputs("burn-normal", "5", "50", "60")
    res = _run_soil([row], tmp_path)
    assert res.returncode == 0, res.stderr
    r = res.rows("_summary")[0]

    wdf = 5.0 / 4.46  # TPA_To_KiSq, fof_util.cpp:543-549
    dfm = 60.0 / 100.0
    expected_dfi = 11.25 - 4.05 * dfm
    ff = 50.0 / 100.0
    expected_tdf = 1.0e4 * ff * wdf / (7.5 - 2.7 * dfm)
    expected_amt = (ff * wdf) / expected_tdf

    assert float(r["duff_burn_intensity_kw"]) == pytest.approx(expected_dfi, abs=1e-4)
    assert float(r["duff_burn_duration_s"]) == pytest.approx(expected_tdf, abs=1e-2)
    assert float(r["duff_burn_consumed_per_sec"]) == pytest.approx(expected_amt, abs=1e-8)


def test_soil_duff_burn_columns_zero_at_zero_load(tmp_path):
    """Schema v2: DuffBurn()'s ``wdf <= 0`` guard (``bur_brn.cpp:1960-
    1961``) -- zero duff_load_tac gives all-zero duff_burn_* columns."""
    row = _soil_duff_row_with_burn_inputs("burn-zero-load", "0", "50", "60")
    res = _run_soil([row], tmp_path)
    assert res.returncode == 0, res.stderr
    r = res.rows("_summary")[0]
    assert float(r["duff_burn_intensity_kw"]) == 0.0
    assert float(r["duff_burn_duration_s"]) == 0.0
    assert float(r["duff_burn_consumed_per_sec"]) == 0.0


def test_soil_duff_burn_columns_zero_at_moisture_threshold(tmp_path):
    """Schema v2: DuffBurn()'s ``dfm >= 1.96`` guard (``bur_brn.cpp:1960-
    1961``) -- duff_moist_pct=196 (ratio 1.96) gives all-zero duff_burn_*
    columns, the non-burning boundary."""
    row = _soil_duff_row_with_burn_inputs("burn-wet", "5", "50", "196")
    res = _run_soil([row], tmp_path)
    assert res.returncode == 0, res.stderr
    r = res.rows("_summary")[0]
    assert float(r["duff_burn_intensity_kw"]) == 0.0
    assert float(r["duff_burn_duration_s"]) == 0.0
    assert float(r["duff_burn_consumed_per_sec"]) == 0.0


def test_soil_duff_burn_columns_partial_consumption_case(tmp_path):
    """Schema v2: a genuinely distinct discriminating case (near-total
    consumption, matching Phase 5's SOI-DUF-06 inputs) -- verifies the
    duff_burn_* columns match the direct formula for a SECOND, materially
    different input combination, not just the one 'normal' case above."""
    row = _soil_duff_row_with_burn_inputs("burn-partial", "5", "97.5", "45")
    res = _run_soil([row], tmp_path)
    assert res.returncode == 0, res.stderr
    r = res.rows("_summary")[0]

    wdf = 5.0 / 4.46
    dfm = 45.0 / 100.0
    expected_dfi = 11.25 - 4.05 * dfm
    ff = 97.5 / 100.0
    expected_tdf = 1.0e4 * ff * wdf / (7.5 - 2.7 * dfm)
    expected_amt = (ff * wdf) / expected_tdf

    assert float(r["duff_burn_intensity_kw"]) == pytest.approx(expected_dfi, abs=1e-4)
    assert float(r["duff_burn_duration_s"]) == pytest.approx(expected_tdf, abs=1e-2)
    assert float(r["duff_burn_consumed_per_sec"]) == pytest.approx(expected_amt, abs=1e-8)
    # Genuinely distinct from the "normal" case above -- proves this test
    # discriminates rather than trivially re-confirming the same numbers.
    normal_row = _soil_duff_row_with_burn_inputs("burn-normal-2", "5", "50", "60")
    normal_res = _run_soil([normal_row], tmp_path, name="burn_normal_2")
    assert normal_res.returncode == 0, normal_res.stderr
    normal_r = normal_res.rows("_summary")[0]
    assert float(r["duff_burn_duration_s"]) != pytest.approx(
        float(normal_r["duff_burn_duration_s"]), abs=1.0,
    )


def _soil_diag_env(spec):
    """
    Build a subprocess environment requesting the F-70 soil-solver
    diagnostic output.

    :param spec: Value for ``FOFEM_TEST_SOIL_DIAG`` (a case_id, a
        comma-separated list of case_ids, or ``"*"`` for every case_id in
        the run).
    :return: A copy of ``os.environ`` with ``FOFEM_TEST_SOIL_DIAG`` set.
    """
    env = dict(os.environ)
    env["FOFEM_TEST_SOIL_DIAG"] = spec
    return env


#: Expected diagnostic column set (harness-contract, F-70). Checked by
#: name (not just count) so a column reordering is caught, not just a
#: width change.
SOIL_DIAG_COLUMNS = (
    "case_id", "record_kind", "time_index", "time_s", "node_index",
    "temp_tn_c", "temp_t_c", "water_content_wn", "water_content_w",
    "matric_potential_p", "humidity_h", "vapor_pressure_psat_pa",
    "cond_kh", "cond_kv", "cond_enh", "surface_flux_w", "heat_frac_pc",
    "ambient_rabs_w", "fire_forcing_w", "input_sha256",
)

#: The 4 representative nodes the harness emits "final_node"/"timestep"
#: diagnostics for (fof_soi.cpp 1-based node index): surface, 1cm, 4cm,
#: and the fixed deep boundary. Matches kSoilDiagNodes in test_harness.cpp
#: exactly.
SOIL_DIAG_NODES = (1, 2, 5, 14)


def test_soil_diag_absent_by_default(tmp_path):
    """Class: opt-in only. Without FOFEM_TEST_SOIL_DIAG set, no
    ``_soidiag.csv`` file is produced at all -- the normal summary/field
    schema and every existing golden run are completely unaffected."""
    res = _run_soil([_soil_duff_row()], tmp_path,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    assert res.rows("_soidiag") == []
    assert not os.path.isfile(os.path.join(str(tmp_path), "case_soidiag.csv"))


def test_soil_diag_absent_when_env_var_present_but_empty(tmp_path):
    """An empty (but present) FOFEM_TEST_SOIL_DIAG is treated identically
    to absent -- no diagnostic file, matching the harness's own
    ``raw[0] == '\\0'`` check."""
    env = _soil_diag_env("")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    assert res.rows("_soidiag") == []


def test_soil_diag_emitted_only_for_requested_case_id(tmp_path):
    """Emission is per-case_id, not per-run: a 2-row input with only ONE
    case_id named in FOFEM_TEST_SOIL_DIAG produces diagnostic rows for
    that case only."""
    row_a = _soil_duff_row(case_id="wanted")
    row_b = _soil_duff_row(case_id="unwanted")
    env = _soil_diag_env("wanted")
    res = _run_soil([row_a, row_b], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows, "expected diagnostic rows for the requested case_id"
    case_ids = {r["case_id"] for r in rows}
    assert case_ids == {"wanted"}


def test_soil_diag_wildcard_covers_every_case_id(tmp_path):
    """``FOFEM_TEST_SOIL_DIAG=*`` requests diagnostics for every case_id
    in the run, not just one."""
    row_a = _soil_duff_row(case_id="c_a")
    row_b = _soil_duff_row(case_id="c_b")
    env = _soil_diag_env("*")
    res = _run_soil([row_a, row_b], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    case_ids = {r["case_id"] for r in res.rows("_soidiag")}
    assert case_ids == {"c_a", "c_b"}


def test_soil_diag_header_matches_declared_columns(tmp_path):
    """The written header matches :data:`SOIL_DIAG_COLUMNS` exactly, by
    name and order -- a silent column reorder would otherwise pass a
    row-count-only check."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows
    assert tuple(rows[0].keys()) == SOIL_DIAG_COLUMNS


def test_soil_diag_row_counts_deterministic(tmp_path):
    """Two independent runs of the SAME scenario produce the SAME
    diagnostic row count -- the harness's own reconciliation check
    (soidiag_rows_written == soidiag_rows_expected) already enforces this
    internally (a mismatch is a FATAL, nonzero-exit error), but this
    proves the observable row count itself is stable across runs, not
    merely that the internal check never fires."""
    env = _soil_diag_env("*")
    res1 = _run_soil([_soil_duff_row()], tmp_path, name="run1", env=env,
                      output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    res2 = _run_soil([_soil_duff_row()], tmp_path, name="run2", env=env,
                      output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res1.returncode == 0, res1.stderr
    assert res2.returncode == 0, res2.stderr
    rows1 = res1.rows("_soidiag")
    rows2 = res2.rows("_soidiag")
    assert len(rows1) == len(rows2) > 0


def test_soil_diag_keys_are_unique(tmp_path):
    """Every (case_id, record_kind, time_index, node_index) key is
    unique -- "timestep" rows keyed by (case_id, 'timestep', time_index,
    node_index), "final_node" rows keyed by (case_id, 'final_node', 'NA',
    node_index); no duplicate/overwritten rows."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    keys = [(r["case_id"], r["record_kind"], r["time_index"], r["node_index"])
            for r in rows]
    assert len(keys) == len(set(keys)), "duplicate diagnostic row key(s) found"


def test_soil_diag_row_multiplicity_matches_nodes_times_timesteps(tmp_path):
    """Exact multiplicity check: "timestep" rows number
    ``n_time_indices * len(SOIL_DIAG_NODES)``, and "final_node" rows
    number exactly ``len(SOIL_DIAG_NODES)`` -- proving the declared node
    set is really what got emitted, not just a nonzero row count."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    summary_row = res.rows("_summary")[0]
    n_time_indices = int(summary_row["n_time_indices"])
    rows = res.rows("_soidiag")
    timestep_rows = [r for r in rows if r["record_kind"] == "timestep"]
    final_rows = [r for r in rows if r["record_kind"] == "final_node"]
    assert len(timestep_rows) == n_time_indices * len(SOIL_DIAG_NODES)
    assert len(final_rows) == len(SOIL_DIAG_NODES)
    assert {int(r["node_index"]) for r in final_rows} == set(SOIL_DIAG_NODES)


def test_soil_diag_finite_on_success(tmp_path):
    """Every non-NA numeric field on a successful row is finite -- no
    "nan"/"inf" text, matching the harness's own general finiteness
    contract."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    numeric_cols = [c for c in SOIL_DIAG_COLUMNS
                    if c not in ("case_id", "record_kind", "input_sha256")]
    for row in res.rows("_soidiag"):
        for col in numeric_cols:
            val = row[col]
            if val == "NA":
                continue
            assert math.isfinite(float(val)), (col, val, row["record_kind"])


def test_soil_diag_timestep_time_s_uses_real_recorded_interval(tmp_path):
    """time_s on "timestep" rows equals time_index * the REAL recorded
    interval (SHA_GetInc(), read from the harness's own summary/field
    output for this exact run), not an assumed constant -- proven by
    cross-checking against the SAME field.csv time_s values for this
    run."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    field_rows = res.rows("_field")
    field_time_s = {int(r["time_index"]): int(r["time_s"]) for r in field_rows}
    diag_rows = [r for r in res.rows("_soidiag") if r["record_kind"] == "timestep"]
    assert diag_rows
    for row in diag_rows:
        ti = int(row["time_index"])
        assert int(row["time_s"]) == field_time_s[ti]


def test_soil_diag_final_node_fields_are_na_only_for_out_of_scope_columns(tmp_path):
    """"final_node" rows carry real values for the per-node physics
    columns and NA for the per-timestep-only columns (surface_flux_w/
    heat_frac_pc/ambient_rabs_w/fire_forcing_w) -- proving the schema's
    own NA/real split is applied correctly, not merely that SOME columns
    are non-NA somewhere."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    final_rows = [r for r in res.rows("_soidiag") if r["record_kind"] == "final_node"]
    assert final_rows
    real_cols = ("temp_tn_c", "temp_t_c", "water_content_wn", "water_content_w",
                 "matric_potential_p", "humidity_h", "vapor_pressure_psat_pa",
                 "cond_kh", "cond_kv", "cond_enh")
    na_cols = ("surface_flux_w", "heat_frac_pc", "ambient_rabs_w", "fire_forcing_w")
    for row in final_rows:
        for col in real_cols:
            assert row[col] != "NA", (col, row)
        for col in na_cols:
            assert row[col] == "NA", (col, row)


def test_soil_diag_timestep_fields_are_na_only_for_out_of_scope_columns(tmp_path):
    """"timestep" rows carry a real temp_tn_c and real surface-flux/
    heat-fraction/ambient/fire-forcing values, and NA for every
    "final_node"-only physics column (temp_t_c, water content, matric
    potential, humidity, vapor pressure, conductivities)."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    ts_rows = [r for r in res.rows("_soidiag") if r["record_kind"] == "timestep"]
    assert ts_rows
    final_only_cols = ("temp_t_c", "water_content_wn", "water_content_w",
                        "matric_potential_p", "humidity_h",
                        "vapor_pressure_psat_pa", "cond_kh", "cond_kv", "cond_enh")
    for row in ts_rows:
        assert row["temp_tn_c"] != "NA"
        assert row["surface_flux_w"] != "NA"
        assert row["heat_frac_pc"] != "NA"
        assert row["ambient_rabs_w"] != "NA"
        assert row["fire_forcing_w"] != "NA"
        for col in final_only_cols:
            assert row[col] == "NA", (col, row)


def test_soil_diag_surface_flux_equals_ambient_plus_fire_forcing(tmp_path):
    """A trivial but real consistency check on the harness's own
    arithmetic: surface_flux_w == ambient_rabs_w + fire_forcing_w exactly
    (both computed inside the harness, never invented)."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    ts_rows = [r for r in res.rows("_soidiag") if r["record_kind"] == "timestep"]
    assert ts_rows
    for row in ts_rows:
        surface = float(row["surface_flux_w"])
        ambient = float(row["ambient_rabs_w"])
        fire = float(row["fire_forcing_w"])
        assert surface == pytest.approx(ambient + fire, abs=1e-3)


def test_soil_diag_works_for_a_stable_duff_case(tmp_path):
    """Diagnostics run successfully for a previously near-matching duff
    case (SOI-DUF-02-like: Loamy-Skeletal, wetter soil, moderate
    consumption) -- one of the two required scenario categories."""
    row = _soil_duff_row_with_burn_inputs(
        "stable-duff", load="8", consumed="40", moist="70",
    )
    row[3] = "Loamy-Skeletal"
    row[7] = "20"
    env = _soil_diag_env("*")
    res = _run_soil([row], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows
    assert any(r["record_kind"] == "final_node" for r in rows)


def test_soil_diag_works_for_a_divergent_dry_duff_case(tmp_path):
    """Diagnostics run successfully for SOI-DUF-04 (Coarse-Silt, 5%
    soil moisture) -- the materially-divergent case named explicitly by
    the task."""
    row = _soil_duff_row_with_burn_inputs(
        "divergent-duff", load="5", consumed="50", moist="45",
    )
    row[3] = "Coarse-Silt"
    row[7] = "5"
    env = _soil_diag_env("*")
    res = _run_soil([row], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows
    assert any(r["record_kind"] == "final_node" for r in rows)


def test_soil_diag_works_for_a_dry_nonduff_case(tmp_path):
    """Diagnostics run successfully for a dry non-duff case
    (SOI-NOD-04-like: Coarse-Silt, 5% soil moisture, zero-duff route) --
    the third required scenario category."""
    row = _soil_zduff_row(case_id="dry-nonduff", soil_type="Coarse-Silt",
                          soil_moist_pct="5")
    env = _soil_diag_env("*")
    res = _run_soil([row], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows
    assert any(r["record_kind"] == "final_node" for r in rows)
    assert all(r["record_kind"] != "timestep" or r["heat_frac_pc"] != "NA"
               for r in rows)


def test_soil_diag_unrequested_case_id_among_requested_ones_emits_nothing(tmp_path):
    """Fail-closed-adjacent: a case_id that never appears in
    FOFEM_TEST_SOIL_DIAG's list, alongside ones that do, gets zero
    diagnostic rows -- proves the per-row gate genuinely filters rather
    than defaulting to "on" once any case_id matches."""
    row_a = _soil_duff_row(case_id="present")
    row_b = _soil_duff_row(case_id="also_present")
    env = _soil_diag_env("present,also_present")
    res = _run_soil([row_a, row_b], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    row_c = _soil_duff_row(case_id="never_requested")
    env2 = _soil_diag_env("present,also_present")
    res2 = _run_soil([row_a, row_c], tmp_path, name="case2", env=env2,
                      output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res2.returncode == 0, res2.stderr
    case_ids2 = {r["case_id"] for r in res2.rows("_soidiag")}
    assert case_ids2 == {"present"}
    assert "never_requested" not in case_ids2


def test_soil_diag_row_width_matches_header_on_every_row(tmp_path):
    """Regression guard for the exact bug this pass found and fixed: the
    "timestep" row builder originally omitted one NA placeholder
    (cond_enh), writing 19 fields against the declared 20-column header
    -- silently dropped by the harness's own CsvWriter (a width-mismatched
    ``row()`` call sets ``failed`` and returns without writing, so ALL
    "timestep" rows vanished from the output while the run still
    exited 0) until the final close/flush check caught the latched
    ``failed`` flag and turned it into a real, nonzero-exit FATAL error
    (reproduced directly during this pass, before the fix). Pinned here
    as a permanent regression: every row, of both record kinds, must have
    exactly ``len(SOIL_DIAG_COLUMNS)`` fields, and the run must succeed."""
    env = _soil_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows
    for row in rows:
        assert len(row) == len(SOIL_DIAG_COLUMNS)
    assert any(r["record_kind"] == "timestep" for r in rows)
    assert any(r["record_kind"] == "final_node" for r in rows)


def test_soil_diag_case_id_matching_is_exact_not_fuzzy(tmp_path):
    """Fail-closed-adjacent: a malformed request (extra whitespace inside
    a comma-separated case_id token) does NOT fuzzy-match the real
    case_id -- the parser does exact substring equality per token, with
    no trimming. Proves the "opt-in" gate cannot be accidentally
    satisfied by an almost-right request."""
    row = _soil_duff_row(case_id="exact_case")
    env = _soil_diag_env(" exact_case ")  # whitespace-padded token
    res = _run_soil([row], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    assert res.rows("_soidiag") == []

    # Sanity: the SAME case_id, unpadded, does match.
    env2 = _soil_diag_env("exact_case")
    res2 = _run_soil([row], tmp_path, name="case2", env=env2,
                      output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res2.returncode == 0, res2.stderr
    assert res2.rows("_soidiag") != []


def test_soil_duff_burn_columns_present_on_zduff_route_too(tmp_path):
    """Schema v2: the duff_burn_* columns are computed unconditionally from
    the three v1 input columns, independent of whether the row selects the
    Duff or Zero-Duff route (SD_Mngr_New vs SE_Mngr_Array) -- confirmed
    directly rather than assumed, since SOIL_CAMPBELL_ROW_OK's own trailing
    three columns are the SI_Init defaults (0 / -1 / 0)."""
    res = _run_soil([SOIL_CAMPBELL_ROW_OK], tmp_path)
    assert res.returncode == 0, res.stderr
    r = res.rows("_summary")[0]
    assert r["model"] == "Zero-Duff"
    # duff_load_tac=0 on this row -> zero forcing, but the columns exist
    # and are well-formed floats, not NA/missing.
    assert float(r["duff_burn_intensity_kw"]) == 0.0
    assert float(r["duff_burn_duration_s"]) == 0.0
    assert float(r["duff_burn_consumed_per_sec"]) == 0.0


# ===========================================================================
# F-70 second diagnostic pass: FOFEM_TEST_SOIL_STATE_DIAG
# (_soistate.csv full per-timestep coupled state, _soisubiter.csv
# per-Newton-sub-iteration surface-node trace). Requires the SEPARATE
# fofem_test_soidiag binary (built from the overlay's fof_soi_instr.cpp)
# -- session-scoped so it builds at most once per test-session, and
# skips (not fails) when the toolchain is unavailable, exactly like the
# module's own `_built` fixture for the normal binary.
# ===========================================================================

SOIL_STATE_DIAG_COLUMNS = (
    "case_id", "step_index", "node_index", "temp_tn_c", "temp_t_true_c",
    "water_content_wn", "water_content_w_true", "matric_potential_p",
    "humidity_h", "vapor_pressure_psat_pa", "cond_kh", "cond_kv",
    "cond_enh", "resid_water_r_sev", "resid_heat_r_seh", "n_subiter",
    "surface_flux_w_rabs_in", "input_sha256",
)

SOIL_SUBITER_COLUMNS = (
    "case_id", "step_index", "n_subiter", "temp_tn1_c",
    "matric_potential_p1", "resid_water_r_sev", "resid_heat_r_seh",
    "input_sha256",
)

#: The full surface-node-update crosswalk record (F-70 third round).
#: MUST match test_harness.cpp's own SOIL_SURFUP_DIAG_COLUMNS exactly
#: (48 fields, same order as SoiSurfaceUpdateDiag's field declarations).
SOIL_SURFUP_DIAG_COLUMNS = (
    "case_id", "step_index", "n_subiter",
    "old_tn1", "new_tn1", "old_p1", "new_p1", "old_wn1", "new_wn1",
    "old_h1", "new_h1", "psat0", "h0", "psat1", "psat2", "h2",
    "s1", "hvap1", "kh1", "enh1", "kv1", "kh2", "kv2",
    "ke0", "ke1", "kev0", "kev1", "conv1", "vcon1", "cp1",
    "d_jv", "d_jvdt", "d_jvdp",
    "dC_before_boundary", "dC_after_boundary", "dv",
    "dCdp", "dvdp", "dCdt_before_boundary", "dCdt_after_boundary", "dvdt",
    "r_rabs_in", "stefan_term", "tk_old",
    "dtn_temperature_raw", "dtn_temperature_clamped", "dtn_matric_raw",
    "p1_before_range_clamp", "p1_clamp_branch",
    "r_sev_running", "r_seh_running", "input_sha256",
)

#: e_mplus1 + 1 (fof_soi.cpp), the exact node_index cardinality every
#: SoiDiagRecordTimestep() call writes -- matches
#: test_harness.cpp's kSoilStateDiagNodeCount.
SOIL_STATE_DIAG_NODE_COUNT = 15


@pytest.fixture(scope="session")
def soidiag_exe():
    """Build (once per session) and return the path to the F-70
    diagnostic-observer binary. Skips cleanly if the toolchain is
    unavailable -- mirrors the module's own `_built` fixture."""
    ok, reason = toolchain_status()
    if not ok:
        pytest.skip(f"MSVC/CMake/Ninja toolchain unavailable: {reason}")
    ok, reason = ensure_soidiag_built()
    if not ok:
        pytest.fail(f"fofem_test_soidiag build failed:\n{reason}")
    assert os.path.isfile(HARNESS_SOIDIAG_EXE)
    return HARNESS_SOIDIAG_EXE


def _soil_state_diag_env(spec):
    """Build a subprocess environment requesting FOFEM_TEST_SOIL_STATE_DIAG.

    :param spec: Value for the env var (a case_id, comma-list, or ``"*"``).
    :return: A copy of ``os.environ`` with the var set.
    """
    env = dict(os.environ)
    env["FOFEM_TEST_SOIL_STATE_DIAG"] = spec
    return env


def _run_soil_soidiag_binary(rows, tmp_path, soidiag_exe, monkeypatch,
                              name="case", state_diag_spec="*",
                              write_side_files=True, **kwargs):
    """Run *rows* through the diagnostic-observer binary with
    FOFEM_TEST_SOIL_STATE_DIAG set, requesting both new diagnostic
    files.

    :func:`_harness_support.resolve_harness_exe` reads
    ``os.environ`` of the CALLING (pytest) process to pick the binary --
    NOT the ``env=`` dict handed to the subprocess -- so the override
    must be set via *monkeypatch* on the real process environment, in
    addition to being present in the subprocess ``env=`` dict for
    ``FOFEM_TEST_SOIL_STATE_DIAG`` itself to reach the child.
    """
    if write_side_files:
        _write_soil_side_files(tmp_path)
    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, soidiag_exe)
    env = _soil_state_diag_env(state_diag_spec)
    env[HARNESS_EXE_OVERRIDE_ENV_VAR] = soidiag_exe
    kwargs.setdefault(
        "output_suffixes",
        SOIL_CAMPBELL_SUFFIXES + ("_soistate", "_soisubiter", "_soisurfup"))
    return run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, rows,
        os.path.join(str(tmp_path), name), env=env, **kwargs,
    )


def test_soil_state_diag_absent_by_default(tmp_path, soidiag_exe, monkeypatch):
    """Opt-in only: even on the diagnostic BINARY, without
    FOFEM_TEST_SOIL_STATE_DIAG set, no ``_soistate.csv``/``_soisubiter.csv``
    rows are produced."""
    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, soidiag_exe)
    env = dict(os.environ)
    _write_soil_side_files(tmp_path)
    res = run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, [_soil_zduff_row()],
        os.path.join(str(tmp_path), "case"), env=env,
        output_suffixes=SOIL_CAMPBELL_SUFFIXES + ("_soistate", "_soisubiter", "_soisurfup"),
    )
    assert res.returncode == 0, res.stderr
    assert res.rows("_soistate") == []
    assert res.rows("_soisubiter") == []
    assert res.rows("_soisurfup") == []


def test_soil_state_diag_requires_the_diagnostic_binary(tmp_path):
    """Documented fail-quiet (not fail-closed) behavior: running the
    NORMAL fofem_test binary (no override) with FOFEM_TEST_SOIL_STATE_DIAG
    set opens the file, writes only the header, and closes cleanly --
    SoiDiagRecordTimestep()/SoiDiagRecordSubIteration() are simply never
    called by the real, unmodified fof_soi.cpp. Proves the normal target
    is unaffected by the new env var, and that a caller who forgets to
    set the override gets an empty (not missing, not erroring) file
    rather than a silent success that looks identical to real data."""
    env = _soil_state_diag_env("*")
    res = _run_soil([_soil_duff_row()], tmp_path, env=env,
                     output_suffixes=SOIL_CAMPBELL_SUFFIXES + ("_soistate", "_soisubiter", "_soisurfup"))
    assert res.returncode == 0, res.stderr
    assert res.rows("_soistate") == []
    assert res.rows("_soisubiter") == []
    assert res.rows("_soisurfup") == []


def test_soil_state_diag_normal_binary_output_unchanged(tmp_path, soidiag_exe, monkeypatch):
    """The required "prove normal output is byte-identical" check: the
    SAME scenario, run through the normal binary and the diagnostic
    binary, with FOFEM_TEST_SOIL_STATE_DIAG UNSET in both cases, produces
    byte-identical _summary/_field rows -- the instrumented
    fof_soi_instr.cpp is a provably no-op copy of the pinned fof_soi.cpp
    when its two hooks are not exercised."""
    row = _soil_duff_row_with_burn_inputs(
        "cmp", load="5", consumed="50", moist="45", dep_pre="2")
    row[3] = "Coarse-Silt"
    _write_soil_side_files(tmp_path)

    # resolve_harness_exe() reads os.environ of the CALLING process, not
    # the env= dict handed to the subprocess -- run the NORMAL case
    # BEFORE monkeypatch.setenv below sets the override, or this
    # comparison would silently run the same binary twice and pass
    # trivially regardless of whether the diagnostic binary works.
    env_normal = dict(os.environ)
    res_normal = run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, [row],
        os.path.join(str(tmp_path), "normal"), env=env_normal,
        output_suffixes=SOIL_CAMPBELL_SUFFIXES,
    )
    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, soidiag_exe)
    env_diag = dict(os.environ)
    env_diag[HARNESS_EXE_OVERRIDE_ENV_VAR] = soidiag_exe
    res_diag = run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, [row],
        os.path.join(str(tmp_path), "diag"), env=env_diag,
        output_suffixes=SOIL_CAMPBELL_SUFFIXES,
    )
    assert res_normal.returncode == 0, res_normal.stderr
    assert res_diag.returncode == 0, res_diag.stderr
    assert res_normal.rows("_summary") == res_diag.rows("_summary")
    assert res_normal.rows("_field") == res_diag.rows("_field")


def test_soil_state_diag_header_matches_declared_columns(tmp_path, soidiag_exe, monkeypatch):
    """All three new files' written headers match the declared column
    tuples exactly, by name and order."""
    row = _soil_duff_row_with_burn_inputs(
        "hdr", load="8", consumed="40", moist="70", dep_pre="3")
    row[3] = "Loamy-Skeletal"
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    state_rows = res.rows("_soistate")
    subiter_rows = res.rows("_soisubiter")
    surfup_rows = res.rows("_soisurfup")
    assert state_rows and subiter_rows and surfup_rows
    assert tuple(state_rows[0].keys()) == SOIL_STATE_DIAG_COLUMNS
    assert tuple(subiter_rows[0].keys()) == SOIL_SUBITER_COLUMNS
    assert tuple(surfup_rows[0].keys()) == SOIL_SURFUP_DIAG_COLUMNS


def test_soil_state_diag_emitted_only_for_requested_case_id(tmp_path, soidiag_exe, monkeypatch):
    """Per-case_id gating, matching FOFEM_TEST_SOIL_DIAG's own contract:
    a 2-row input with only one case_id requested produces rows for that
    case only, in ALL THREE new files."""
    row_a = _soil_duff_row_with_burn_inputs(
        "wanted", load="5", consumed="50", moist="45", dep_pre="2")
    row_a[3] = "Coarse-Silt"
    row_b = _soil_duff_row_with_burn_inputs(
        "unwanted", load="8", consumed="40", moist="70", dep_pre="3")
    row_b[3] = "Loamy-Skeletal"
    res = _run_soil_soidiag_binary([row_a, row_b], tmp_path, soidiag_exe,
                                    monkeypatch, state_diag_spec="wanted")
    assert res.returncode == 0, res.stderr
    assert {r["case_id"] for r in res.rows("_soistate")} == {"wanted"}
    assert {r["case_id"] for r in res.rows("_soisubiter")} == {"wanted"}
    assert {r["case_id"] for r in res.rows("_soisurfup")} == {"wanted"}


def test_soil_state_diag_case_id_matching_is_exact_not_fuzzy(tmp_path, soidiag_exe, monkeypatch):
    """Fail-closed-adjacent, matching FOFEM_TEST_SOIL_DIAG's own
    contract: a whitespace-padded token does not fuzzy-match the real
    case_id."""
    row = _soil_duff_row_with_burn_inputs(
        "exact_case", load="5", consumed="50", moist="45", dep_pre="2")
    row[3] = "Coarse-Silt"
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe,
                                    monkeypatch, state_diag_spec=" exact_case ")
    assert res.returncode == 0, res.stderr
    assert res.rows("_soistate") == []
    assert res.rows("_soisubiter") == []
    assert res.rows("_soisurfup") == []


def test_soil_state_diag_row_counts_deterministic(tmp_path, soidiag_exe, monkeypatch):
    """Two independent runs of the same scenario produce the same row
    count in both new files -- observable stability, not just an
    internal reconciliation check that never fires."""
    row = _soil_duff_row_with_burn_inputs(
        "det", load="5", consumed="50", moist="45", dep_pre="2")
    row[3] = "Coarse-Silt"
    res1 = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch, name="run1")
    res2 = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch, name="run2")
    assert res1.returncode == 0, res1.stderr
    assert res2.returncode == 0, res2.stderr
    assert len(res1.rows("_soistate")) == len(res2.rows("_soistate")) > 0
    assert len(res1.rows("_soisubiter")) == len(res2.rows("_soisubiter")) > 0


def test_soil_state_diag_keys_are_unique(tmp_path, soidiag_exe, monkeypatch):
    """Every (case_id, step_index, node_index) key in _soistate.csv, and
    every (case_id, step_index, n_subiter) key in _soisubiter.csv, is
    unique -- no duplicate/overwritten rows."""
    row = _soil_duff_row_with_burn_inputs(
        "uniq", load="5", consumed="50", moist="45", dep_pre="2")
    row[3] = "Coarse-Silt"
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    state_keys = [(r["case_id"], r["step_index"], r["node_index"])
                  for r in res.rows("_soistate")]
    assert len(state_keys) == len(set(state_keys))
    subiter_keys = [(r["case_id"], r["step_index"], r["n_subiter"])
                     for r in res.rows("_soisubiter")]
    assert len(subiter_keys) == len(set(subiter_keys))


def test_soil_state_diag_row_multiplicity_is_multiple_of_node_count(tmp_path, soidiag_exe, monkeypatch):
    """Every step_index in _soistate.csv contributes exactly
    SOIL_STATE_DIAG_NODE_COUNT rows -- the harness's own per-row fail-closed
    reconciliation (rows_written % node_count == 0) already enforces this
    internally; this proves the OBSERVABLE row count is really a multiple
    of the node count, not merely that the internal check never fires."""
    row = _soil_duff_row_with_burn_inputs(
        "mult", load="5", consumed="50", moist="45", dep_pre="2")
    row[3] = "Coarse-Silt"
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soistate")
    assert rows
    assert len(rows) % SOIL_STATE_DIAG_NODE_COUNT == 0
    step_indices = sorted({int(r["step_index"]) for r in rows})
    for step in step_indices:
        this_step = [r for r in rows if int(r["step_index"]) == step]
        assert len(this_step) == SOIL_STATE_DIAG_NODE_COUNT
        assert {int(r["node_index"]) for r in this_step} == set(range(SOIL_STATE_DIAG_NODE_COUNT))


def test_soil_state_diag_finite_on_successful_rows(tmp_path, soidiag_exe, monkeypatch):
    """Every numeric field on every row of all three new files is finite
    for a successful (oc == OK) scenario -- a NaN/inf here would
    otherwise silently pass through csv.DictReader as an unparsed
    string."""
    row = _soil_duff_row_with_burn_inputs(
        "finite", load="8", consumed="40", moist="70", dep_pre="3")
    row[3] = "Loamy-Skeletal"
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    numeric_state_cols = [c for c in SOIL_STATE_DIAG_COLUMNS
                           if c not in ("case_id", "input_sha256")]
    for r in res.rows("_soistate"):
        for c in numeric_state_cols:
            assert math.isfinite(float(r[c])), (c, r[c])
    numeric_subiter_cols = [c for c in SOIL_SUBITER_COLUMNS
                             if c not in ("case_id", "input_sha256")]
    for r in res.rows("_soisubiter"):
        for c in numeric_subiter_cols:
            assert math.isfinite(float(r[c])), (c, r[c])
    numeric_surfup_cols = [c for c in SOIL_SURFUP_DIAG_COLUMNS
                            if c not in ("case_id", "input_sha256")]
    for r in res.rows("_soisurfup"):
        for c in numeric_surfup_cols:
            assert math.isfinite(float(r[c])), (c, r[c])


def test_soil_surfup_row_count_matches_subiter_row_count(tmp_path, soidiag_exe, monkeypatch):
    """SoiDiagRecordSubIteration() and SoiDiagRecordSurfaceUpdate() are
    called exactly once each, from the SAME i==1 pass of the SAME
    sub-iteration loop -- their row counts must be identical (the
    harness's own internal reconciliation already fail-closes on this;
    this proves the OBSERVABLE counts agree, not merely that the
    internal check never fires)."""
    row = _soil_zduff_row(case_id="surfup-count", soil_type="Coarse-Silt",
                           soil_moist_pct="5")
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    subiter_rows = res.rows("_soisubiter")
    surfup_rows = res.rows("_soisurfup")
    assert subiter_rows and surfup_rows
    assert len(subiter_rows) == len(surfup_rows)
    subiter_keys = {(r["step_index"], r["n_subiter"]) for r in subiter_rows}
    surfup_keys = {(r["step_index"], r["n_subiter"]) for r in surfup_rows}
    assert subiter_keys == surfup_keys


def test_soil_surfup_matches_subiter_temperature_and_matric_potential(
        tmp_path, soidiag_exe, monkeypatch):
    """Both hooks report the SAME surface-node temperature/matric
    potential for the same (step_index, n_subiter) -- an independent
    cross-check that neither hook's own field wiring is wrong (e.g. an
    accidental off-by-one in which array element is read)."""
    row = _soil_zduff_row(case_id="surfup-cross", soil_type="Coarse-Silt",
                           soil_moist_pct="5")
    res = _run_soil_soidiag_binary([row], tmp_path, soidiag_exe, monkeypatch)
    assert res.returncode == 0, res.stderr
    subiter_by_key = {(r["step_index"], r["n_subiter"]): r
                       for r in res.rows("_soisubiter")}
    surfup_by_key = {(r["step_index"], r["n_subiter"]): r
                      for r in res.rows("_soisurfup")}
    assert subiter_by_key.keys() == surfup_by_key.keys()
    checked = 0
    for key, sub_row in subiter_by_key.items():
        surf_row = surfup_by_key[key]
        assert float(sub_row["temp_tn1_c"]) == pytest.approx(float(surf_row["new_tn1"]), abs=1e-4), key
        assert float(sub_row["matric_potential_p1"]) == pytest.approx(float(surf_row["new_p1"]), abs=1e-2), key
        checked += 1
    assert checked > 0


def test_soil_state_diag_covers_the_three_required_scenarios(tmp_path, soidiag_exe, monkeypatch):
    """F-70's three required scenario categories (a previously
    near-matching duff case, a materially-divergent dry duff case, and a
    materially-divergent dry non-duff case) all produce real, nonempty,
    finite diagnostic rows in all three new files -- the actual facility
    :mod:`test_soil_solver_diagnostic_comparison` depends on to locate
    the first Python/C++ divergence for each."""
    stable_duff = _soil_duff_row_with_burn_inputs(
        "stable-duff", load="8", consumed="40", moist="70", dep_pre="3")
    stable_duff[3] = "Loamy-Skeletal"
    stable_duff[6] = "1.5"
    stable_duff[7] = "20"

    divergent_duff = _soil_duff_row_with_burn_inputs(
        "divergent-duff", load="5", consumed="50", moist="45", dep_pre="2")
    divergent_duff[3] = "Coarse-Silt"
    divergent_duff[6] = "1"
    divergent_duff[7] = "5"

    dry_nonduff = _soil_zduff_row(
        case_id="dry-nonduff", soil_type="Coarse-Silt", soil_moist_pct="5")

    res = _run_soil_soidiag_binary(
        [stable_duff, divergent_duff, dry_nonduff], tmp_path, soidiag_exe,
        monkeypatch)
    assert res.returncode == 0, res.stderr

    state_rows = res.rows("_soistate")
    subiter_rows = res.rows("_soisubiter")
    surfup_rows = res.rows("_soisurfup")
    for case_id in ("stable-duff", "divergent-duff", "dry-nonduff"):
        this_state = [r for r in state_rows if r["case_id"] == case_id]
        this_subiter = [r for r in subiter_rows if r["case_id"] == case_id]
        this_surfup = [r for r in surfup_rows if r["case_id"] == case_id]
        assert this_state, f"no _soistate rows for {case_id}"
        assert this_subiter, f"no _soisubiter rows for {case_id}"
        assert this_surfup, f"no _soisurfup rows for {case_id}"
