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

import os
import tempfile

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._harness_support import (
    FOF_UNIX_DIR,
    HARNESS_EXE,
    HARNESS_EXE_OVERRIDE_ENV_VAR,
    SPECIES_CSV,
    TIMEOUT_HARNESS_RUN_S,
    ensure_built,
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

MORTALITY_HEADER = [
    "case_id", "expect_error", "species", "equ_type", "dbh_in", "ht_ft",
    "crown_ratio_x10", "fs_value_ft", "fs_kind", "bole_char_ft",
    "fire_severity", "ckr_pct", "cvk_pct", "beetles",
]
MORTALITY_ROW_OK = [
    "c1", "0", "PSME", "CroSco", "12", "60", "50", "4", "Flame", "0",
    "NA", "0", "0", "0",
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
        "NA", "0", "0", "0",
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
# ===========================================================================

@pytest.mark.parametrize("mode", ALL_MODE_NAMES)
def test_row3_wrong_schema_version(mode, tmp_path):
    m = MODES[mode]
    res = _run(mode, [m["row"]], tmp_path, schema_version="2")
    assert res.returncode != 0


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
