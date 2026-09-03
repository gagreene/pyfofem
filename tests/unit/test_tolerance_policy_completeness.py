#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_tolerance_policy_completeness.py - Schema/completeness checks for
``tests/cpp_parity_live/tolerance_policy.json`` and the output-column
contract (``_output_contract.py``) it classifies against (Phase 2
correction items 1/2/6): every entry carries the required evidence
fields, every "unverified" entry is honestly untolerance'd (null
atol/rtol), every key any Phase 2 golden actually cites
(``generate_phase2_goldens.GOLDEN_TOLERANCE_KEYS``) resolves in the
policy, and — the round 4 addition — every REAL scientific column of
every generated Phase 2 golden output file has exactly one applicable
policy classification, derived from the real CSV headers rather than a
second hardcoded list.

Reads the already-generated goldens under
``tests/test_data/test_golden_output/phase2/`` (no live C++ build
performed by this module itself; if that directory is absent — e.g. a
fresh checkout before the harness has ever been built — the
column-coverage tests skip with an explicit reason rather than silently
passing or erroring).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import MODE_OUTPUT_SUFFIXES, load_tolerance_policy
from tests.cpp_parity_live._output_contract import (
    METADATA_COLUMNS,
    classify_columns,
    component_field_policy_key,
    read_real_header,
    real_distinct_components,
)
from tests.cpp_parity_live._phase4_contract import (
    GOLDEN_ROOT as PHASE4_GOLDEN_ROOT,
    PHASE4_MODES,
    PHASE4_ROUTE_KEYS,
    phase4_policy_keys,
)
from tests.cpp_parity_live.generate_phase2_goldens import GOLDEN_ROOT, GOLDEN_TOLERANCE_KEYS

_REQUIRED_FIELDS = frozenset({"status", "atol", "rtol", "justification", "traceability"})

#: Statuses documented as legitimately having no Python counterpart to
#: compare against at all — see item 6's "contract_only where no Python
#: counterpart exists". No entry in the current policy uses this status
#: yet, but the completeness rule below is written to allow it.
_CONTRACT_ONLY_STATUSES = frozenset({"contract_only"})

#: Phase 4 policy-section name to the harness mode whose real output columns
#: it classifies. Phase 4 reuses the six qualified Phase 2 modes unchanged, so
#: a ``<mode>_p4`` entry's ``covers_columns`` must name real columns of
#: ``<mode>`` exactly as a bare ``<mode>`` entry must.
_PHASE4_SECTION_TO_MODE = {f"{mode}_p4": mode for mode in PHASE4_MODES}

#: (mode, suffix) pairs whose real header is read from a per-mode-only
#: (not per-file) generated golden, i.e. every declared output file of
#: every mode MODE_OUTPUT_SUFFIXES knows about. Excludes "_components",
#: which is covered by a dedicated component-resolution test instead of
#: the generic column->covers_columns coverage test (it is long-format:
#: the same four generic value columns repeat per component row, so
#: "coverage" means resolving real (component, value_column) pairs, not
#: matching a fixed column name).
_WIDE_FORMAT_OUTPUT_FILES = [
    (mode, suffix)
    for mode, suffixes in MODE_OUTPUT_SUFFIXES.items()
    for suffix in suffixes
    if not (mode == "consume" and suffix == "_components")
]


def _all_entries():
    """Yield ``(mode, scenario, entry)`` for every real (non-"NOTE") policy
    entry."""
    policy = load_tolerance_policy()
    for mode, scenarios in policy.items():
        if mode == "NOTE":
            continue
        for scenario, entry in scenarios.items():
            yield mode, scenario, entry


def _golden_csv_path(mode: str, suffix: str) -> str:
    return os.path.join(GOLDEN_ROOT, mode, f"{mode}{suffix}.csv")


def _phase4_golden_csv_path(mode: str, suffix: str) -> str:
    return os.path.join(PHASE4_GOLDEN_ROOT, mode, f"{mode}{suffix}.csv")


def _require_phase4_golden_present(mode: str, suffix: str) -> str:
    path = _phase4_golden_csv_path(mode, suffix)
    if not os.path.isfile(path):
        pytest.skip(
            f"Phase 4 golden {path!r} does not exist - run "
            "tests/cpp_parity_live/generate_phase4_goldens.py at least once "
            "before this column-coverage check can run (see docs/CODEBASE.md)."
        )
    return path


def _phase4_real_scientific_columns_by_mode():
    """Return ``{mode: set(real scientific column names)}`` derived from the
    actual generated PHASE 4 golden CSVs - the ground truth every ``*_p4``
    policy entry's ``covers_columns`` is checked against."""
    out = {}
    for mode, suffix in _WIDE_FORMAT_OUTPUT_FILES:
        path = _require_phase4_golden_present(mode, suffix)
        header = read_real_header(path)
        _metadata, scientific = classify_columns(mode, suffix, header)
        out.setdefault(mode, set()).update(scientific)
    return out


def _require_golden_present(mode: str, suffix: str) -> str:
    path = _golden_csv_path(mode, suffix)
    if not os.path.isfile(path):
        pytest.skip(
            f"Phase 2 golden {path!r} does not exist — run "
            "generate_phase2_goldens.py at least once before this "
            "column-coverage check can run (see docs/CODEBASE.md)."
        )
    return path


def _real_scientific_columns_by_mode():
    """Return ``{mode: set(real scientific column names across every
    wide-format output file for that mode)}`` — the ground truth every
    consume/non-consume policy entry's ``covers_columns`` is checked
    against."""
    out = {}
    for mode, suffix in _WIDE_FORMAT_OUTPUT_FILES:
        path = _require_golden_present(mode, suffix)
        header = read_real_header(path)
        _metadata, scientific = classify_columns(mode, suffix, header)
        out.setdefault(mode, set()).update(scientific)
    return out


def test_consume_components_real_component_fields_all_resolve():
    """Every real ``(component, value_column)`` pair found in the
    generated ``consume_components.csv`` must resolve, via
    ``component_field_policy_key``, to a key that actually exists in
    ``tolerance_policy.json``'s ``consume`` section."""
    path = _require_golden_present("consume", "_components")
    policy = load_tolerance_policy()
    consume_keys = {
        key.partition(".")[2] for key in GOLDEN_TOLERANCE_KEYS["consume"]
    }
    components = real_distinct_components(path)
    assert components, "no components found in the generated consume_components.csv"

    value_columns = ["pre_tac", "con_tac", "pos_tac", "pct_con"]
    unresolved = []
    for component in components:
        for value_column in value_columns:
            try:
                key = component_field_policy_key(component, value_column)
            except KeyError as exc:
                unresolved.append(f"({component!r}, {value_column!r}): {exc}")
                continue
            if key not in consume_keys:
                unresolved.append(
                    f"({component!r}, {value_column!r}) resolves to "
                    f"{key!r}, which is not referenced by the canonical "
                    "consume manifest contract"
                )
    assert not unresolved, unresolved


def test_every_entry_has_all_required_fields():
    missing = []
    for mode, scenario, entry in _all_entries():
        absent = _REQUIRED_FIELDS - set(entry.keys())
        if absent:
            missing.append(f"{mode}.{scenario}: missing {sorted(absent)}")
    assert not missing, missing


def test_every_entry_has_a_non_empty_status():
    bad = []
    for mode, scenario, entry in _all_entries():
        status = entry.get("status")
        if not isinstance(status, str) or not status:
            bad.append(f"{mode}.{scenario}: status={status!r}")
    assert not bad, bad


def test_every_golden_tolerance_key_resolves_in_policy():
    policy = load_tolerance_policy()
    unresolved = []
    for mode, keys in GOLDEN_TOLERANCE_KEYS.items():
        for key in keys:
            key_mode, _, scenario = key.partition(".")
            if key_mode not in policy or scenario not in policy.get(key_mode, {}):
                unresolved.append(key)
    assert not unresolved, unresolved


def test_every_golden_reference_covers_its_real_scientific_columns():
    """The exact per-manifest key set must cover every scientific value
    emitted by that canonical row, not merely resolve somewhere in the
    global policy."""
    policy = load_tolerance_policy()
    real_by_mode = _real_scientific_columns_by_mode()
    missing = []
    for mode, real_columns in real_by_mode.items():
        covered = set()
        for dotted_key in GOLDEN_TOLERANCE_KEYS[mode]:
            key_mode, _, route = dotted_key.partition(".")
            assert key_mode == mode
            covered.update(policy[mode][route].get("covers_columns", []))
        uncovered = real_columns - covered
        if uncovered:
            missing.append(
                f"{mode}: canonical manifest keys do not cover "
                f"{sorted(uncovered)}"
            )
    assert not missing, missing


def test_every_metadata_registered_output_file_has_a_known_suffix():
    """`_output_contract.METADATA_COLUMNS`'s keys must exactly match
    `_golden_manifest.MODE_OUTPUT_SUFFIXES`'s full (mode, suffix) set — a
    new output file added to one without the other is a silent coverage
    gap."""
    expected = {
        (mode, suffix)
        for mode, suffixes in MODE_OUTPUT_SUFFIXES.items()
        for suffix in suffixes
    }
    assert set(METADATA_COLUMNS.keys()) == expected


def test_every_numeric_tolerance_has_unit_recorded():
    """A "verified"/"known_divergent*" entry that DOES carry real
    atol/rtol numbers must also record their unit — a bare number with no
    unit is not usable evidence."""
    missing_unit = []
    for mode, scenario, entry in _all_entries():
        if entry["atol"] is not None or entry["rtol"] is not None:
            if not entry.get("unit"):
                missing_unit.append(f"{mode}.{scenario}")
    assert not missing_unit, missing_unit


def test_every_scientific_column_is_classified():
    """Every real scientific column (derived from the actual generated
    golden CSVs, not a second hardcoded list) of every wide-format Phase 2
    output file must be covered by at least one policy entry's
    ``covers_columns`` for that mode."""
    real_by_mode = _real_scientific_columns_by_mode()
    policy = load_tolerance_policy()
    unclassified = []
    for mode, real_columns in real_by_mode.items():
        covered = set()
        for entry in policy[mode].values():
            covered.update(entry.get("covers_columns", []))
        missing = real_columns - covered
        if missing:
            unclassified.append(f"{mode}: unclassified columns {sorted(missing)}")
    assert not unclassified, unclassified


def test_no_duplicate_column_classification_without_shared_column_group():
    """A real column may be covered by MORE than one policy entry only
    when those entries share the same declared ``column_group`` (an
    intentional multi-equation-route classification of the same output,
    e.g. mortality's three routes all writing through ``prob``) — an
    unexplained duplicate (no shared group) is treated as an accidental
    copy-paste error, not a real second scientific claim."""
    policy = load_tolerance_policy()
    bad = []
    for mode, scenarios in policy.items():
        if mode == "NOTE":
            continue
        column_to_groups = {}
        for scenario, entry in scenarios.items():
            group = entry.get("column_group")
            for col in entry.get("covers_columns", []):
                column_to_groups.setdefault(col, []).append((scenario, group))
        for col, occurrences in column_to_groups.items():
            if len(occurrences) <= 1:
                continue
            groups = {g for _scenario, g in occurrences}
            if None in groups or len(groups) != 1:
                bad.append(f"{mode}.{col}: covered by {occurrences} without one shared column_group")
    assert not bad, bad


def test_no_policy_entry_covers_a_nonexistent_column():
    """A ``covers_columns`` entry must name a column that actually exists
    in that mode's wide-format output file(s) — an entry describing a
    column that was renamed or removed is exactly as dangerous as a
    missing one."""
    real_by_mode = _real_scientific_columns_by_mode()
    policy = load_tolerance_policy()
    bad = []
    for mode, real_columns in real_by_mode.items():
        for scenario, entry in policy[mode].items():
            extra = set(entry.get("covers_columns", [])) - real_columns
            if extra:
                bad.append(f"{mode}.{scenario}: covers_columns has nonexistent column(s) {sorted(extra)}")
    assert not bad, bad


def test_no_scientific_column_is_also_registered_as_metadata():
    """A column claimed as scientific by some policy entry's
    ``covers_columns`` must never ALSO appear in that (mode, suffix)'s
    metadata-exclusion set — that would be a direct contradiction in the
    contract."""
    policy = load_tolerance_policy()
    covered_by_mode = {}
    for mode, scenarios in policy.items():
        if mode == "NOTE":
            continue
        covered_by_mode[mode] = {
            col for entry in scenarios.values() for col in entry.get("covers_columns", [])
        }
    bad = []
    for (mode, suffix), metadata_set in METADATA_COLUMNS.items():
        overlap = metadata_set & covered_by_mode.get(mode, set())
        if overlap:
            bad.append(f"({mode!r}, {suffix!r}): columns both metadata and policy-covered: {sorted(overlap)}")
    assert not bad, bad


def test_unverified_entries_have_no_invented_tolerance():
    """Do not invent a tolerance for an output that has not actually been
    compared — every "unverified" entry must carry null atol AND null
    rtol."""
    bad = []
    for mode, scenario, entry in _all_entries():
        if entry.get("status") == "unverified" and (
            entry["atol"] is not None or entry["rtol"] is not None
        ):
            bad.append(f"{mode}.{scenario}: atol={entry['atol']} rtol={entry['rtol']}")
    assert not bad, bad


def test_phase4_every_golden_reference_covers_its_real_scientific_columns():
    """Every real scientific column of every Phase 4 output file must be
    covered by the exact key set that mode's Phase 4 manifest cites - not
    merely resolve somewhere in the global policy."""
    policy = load_tolerance_policy()
    real_by_mode = _phase4_real_scientific_columns_by_mode()
    missing = []
    for mode, real_columns in real_by_mode.items():
        covered = set()
        for dotted_key in phase4_policy_keys(mode):
            section, _, route = dotted_key.partition(".")
            covered.update(policy[section][route].get("covers_columns", []))
        uncovered = real_columns - covered
        if uncovered:
            missing.append(
                f"{mode}: Phase 4 manifest keys do not cover {sorted(uncovered)}"
            )
    assert not missing, missing


def test_phase4_every_route_key_resolves_in_policy():
    """Every dotted key a Phase 4 manifest cites must exist in the policy."""
    policy = load_tolerance_policy()
    unresolved = []
    for mode in PHASE4_MODES:
        for key in phase4_policy_keys(mode):
            section, _, route = key.partition(".")
            if section not in policy or route not in policy.get(section, {}):
                unresolved.append(key)
    assert not unresolved, unresolved


def test_phase4_route_keys_match_the_policy_sections_exactly():
    """A ``*_p4`` section must contain exactly the routes the Phase 4
    contract declares - an orphaned entry is as dangerous as a missing one."""
    policy = load_tolerance_policy()
    mismatched = []
    for section, mode in _PHASE4_SECTION_TO_MODE.items():
        assert section in policy, f"missing policy section {section!r}"
        declared = set(PHASE4_ROUTE_KEYS[mode])
        present = set(policy[section])
        if declared != present:
            mismatched.append(
                f"{section}: declared {sorted(declared)}, present "
                f"{sorted(present)}"
            )
    assert not mismatched, mismatched


def test_phase4_sections_cover_no_nonexistent_column():
    """A ``*_p4`` entry's ``covers_columns`` must name real columns of its
    mode's Phase 4 output files."""
    policy = load_tolerance_policy()
    real_by_mode = _phase4_real_scientific_columns_by_mode()
    bad = []
    for section, mode in _PHASE4_SECTION_TO_MODE.items():
        real_columns = real_by_mode.get(mode, set())
        for route, entry in policy[section].items():
            extra = set(entry.get("covers_columns", [])) - real_columns
            if extra:
                bad.append(
                    f"{section}.{route}: covers_columns has nonexistent "
                    f"column(s) {sorted(extra)}"
                )
    assert not bad, bad


def test_phase4_sections_exist_for_every_phase4_mode():
    """Every Phase 4 mode must have its own ``<mode>_p4`` policy section, so
    a new mode cannot silently inherit the frozen Phase 2 evidence."""
    policy = load_tolerance_policy()
    missing = [s for s in _PHASE4_SECTION_TO_MODE if s not in policy]
    assert not missing, missing


def test_phase4_scenario_scoped_entries_record_a_measured_maximum():
    """A ``known_divergent_scenario_scoped`` entry carries a real tolerance,
    so its justification MUST state the measured maximum among the agreeing
    scenarios - otherwise the tolerance is unevidenced."""
    policy = load_tolerance_policy()
    bad = []
    for section in _PHASE4_SECTION_TO_MODE:
        for route, entry in policy[section].items():
            if entry["status"] != "known_divergent_scenario_scoped":
                continue
            justification = entry["justification"]
            if "measured maximum" not in justification:
                bad.append(f"{section}.{route}: no measured maximum recorded")
            if entry["atol"] is None and entry["rtol"] is None:
                bad.append(
                    f"{section}.{route}: scenario-scoped status but no "
                    "tolerance recorded for the agreeing scenarios"
                )
    assert not bad, bad
