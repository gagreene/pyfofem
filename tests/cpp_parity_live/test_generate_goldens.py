#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_generate_goldens.py - Driver tests for the four scenario-matrix golden
generators (``generate_expanded_matrix_goldens.py``,
``generate_soil_campbell_goldens.py``,
``generate_emissions_equivalence_goldens.py``,
``generate_burnup_extended_goldens.py``), consolidated from four
near-identical modules (test-suite renaming plan, Slice 1) into one
dataset-parametrized module.

Every one of these four generators deliberately owns no promotion, locking
or comparison logic of its own: each imports and reuses the canonical
(``generate_canonical_goldens.py``) generator's ``_promote``/
``_qualify_all``/``verify_regeneration`` verbatim. Those mechanisms are
already covered end to end by ``test_generate_canonical_goldens.py``, so
this module does NOT duplicate them. What it covers, once per dataset, is:
the fail-closed pinned-SHA gate, deterministic generation of that dataset's
own scenario matrix, detection of a corrupted/missing/extra/mismatched
committed file through the SAME production comparison function, the
``dataset``/tolerance-policy-prefix manifest fields, reuse (not
re-implementation) of the promotion helpers, and that generating one
dataset never touches any sibling dataset's committed tree.

The ``burnup_extended`` (formerly Phase 7) dataset keeps its two genuinely
different behaviors from the other three, preserved via its own
:class:`_DatasetCase`'s ``fail_closed_no_skip`` field:

- it FAILS CLOSED (never skips) when the live MSVC/CMake/Ninja toolchain is
  unavailable - a missing toolchain in the acceptance/full lane must be a
  loud, diagnosable failure, not a silent skip;
- every ``tmp_path`` it uses is overridden to a repository-local,
  collision-safe directory (``_scratch.py``'s scratch root), never the
  system/user temporary directory.

Every test here drives the real compiled harness, so the whole module
(apart from ``burnup_extended``'s own fail-closed toolchain gate) skips
cleanly when the MSVC/CMake/Ninja toolchain is unavailable.

Function order: private helpers/fixtures first, then public test
functions, each group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pytest

import tests.cpp_parity_live.generate_burnup_extended_goldens as gbe
import tests.cpp_parity_live.generate_canonical_goldens as gcn
import tests.cpp_parity_live.generate_emissions_equivalence_goldens as gee
import tests.cpp_parity_live.generate_expanded_matrix_goldens as gem
import tests.cpp_parity_live.generate_soil_campbell_goldens as gsc
from tests.cpp_parity_live._burnup_extended_contract import (
    BURNUP_EXTENDED_MODES,
    burnup_extended_rows,
    GOLDEN_ROOT as BURNUP_EXTENDED_GOLDEN_ROOT,
)
from tests.cpp_parity_live._emissions_equivalence_contract import (
    EMISSIONS_EQUIVALENCE_MODES,
    emissions_equivalence_rows,
    GOLDEN_ROOT as EMISSIONS_EQUIVALENCE_GOLDEN_ROOT,
)
from tests.cpp_parity_live._expanded_matrix_contract import (
    EXPANDED_MATRIX_MODES,
    expanded_matrix_rows,
    GOLDEN_ROOT as EXPANDED_MATRIX_GOLDEN_ROOT,
)
from tests.cpp_parity_live._golden_manifest import ProvenanceError, validate_manifest
from tests.cpp_parity_live._harness_support import toolchain_status
from tests.cpp_parity_live._scratch import scratch_tempdir
from tests.cpp_parity_live._soil_campbell_contract import (
    GOLDEN_ROOT as SOIL_CAMPBELL_GOLDEN_ROOT,
    soil_campbell_rows,
    SOIL_CAMPBELL_MODES,
)

pytestmark = pytest.mark.cpp_reference


@dataclass(frozen=True)
class _DatasetCase:
    """One dataset's complete generator-driver test configuration."""

    #: Short label used in test IDs and error messages (e.g. "expanded_matrix").
    label: str
    #: The dataset's generator module.
    gen_module: object
    #: The dataset's harness modes, in declared order.
    modes: Tuple[str, ...]
    #: The dataset's committed golden-tree root.
    golden_root: str
    #: Builds the dataset's complete input-row matrix for one mode.
    rows_func: Callable[[str], List[List[str]]]
    #: The ``dataset`` value every generated manifest for this dataset must record.
    dataset_value: str
    #: Mode used for the corrupted/extra/missing-file probe tests.
    probe_mode: str
    #: Output-CSV suffix (e.g. ``"_summary"``, ``"_field"``) used for the
    #: corrupted-file probe.
    probe_output_suffix: str
    #: Mode used for the manifest-mismatch probe test (may differ from
    #: ``probe_mode``).
    manifest_mismatch_mode: str
    #: Output-CSV suffix removed by the missing-file probe test.
    missing_output_suffix: str
    #: Callable asserting the tolerance_policy_reference key prefix for one
    #: (mode, key) pair.
    assert_key_prefix: Callable[[str, str], None]
    #: ``(label, golden_root)`` for every dataset this dataset's own
    #: generation must never disturb.
    sibling_roots: Tuple[Tuple[str, str], ...]
    #: Asserts the dataset's own scenario-matrix invariants (row count,
    #: unique case IDs, per-branch partition counts).
    assert_scenario_matrix: Callable[[List[List[str]]], None]
    #: ``True`` only for burnup_extended: fails closed (never skips) on a
    #: missing toolchain, and uses a repo-local ``tmp_path`` override.
    fail_closed_no_skip: bool = False


def _hash_tree(root: str) -> Dict[str, str]:
    """
    Return ``{relative_path: sha256}`` for every file under *root*.

    :param root: Directory to hash.
    :returns: Mapping of forward-slashed relative path to hex digest.
    """
    digests = {}
    for directory, _subdirs, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(directory, name)
            relative = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as handle:
                digests[relative] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def _assert_default_matrix(rows: List[List[str]], expected_prefixes: Dict[str, int]) -> None:
    """
    Generic scenario-matrix assertion: unique case IDs, and per-prefix
    case-ID counts matching *expected_prefixes*.

    :param rows: Input rows (each row's first field is its ``case_id``).
    :param expected_prefixes: ``{case_id_prefix: expected_count}``.
    :returns: None.
    """
    case_ids = [row[0] for row in rows]
    assert len(set(case_ids)) == len(case_ids)
    for prefix, expected in expected_prefixes.items():
        assert sum(1 for c in case_ids if c.startswith(prefix)) == expected, prefix


_CASES: Tuple[_DatasetCase, ...] = (
    _DatasetCase(
        label="expanded_matrix",
        gen_module=gem,
        modes=EXPANDED_MATRIX_MODES,
        golden_root=EXPANDED_MATRIX_GOLDEN_ROOT,
        rows_func=expanded_matrix_rows,
        dataset_value="expanded_matrix",
        probe_mode="litter_eq",
        probe_output_suffix="",
        manifest_mismatch_mode="bark_thick",
        missing_output_suffix="",
        assert_key_prefix=lambda mode, key: (_ for _ in ()).throw(
            AssertionError(f"{mode}: {key}")
        ) if not key.startswith(f"{mode}_expanded_matrix.") else None,
        sibling_roots=(("canonical", gcn.GOLDEN_ROOT),),
        assert_scenario_matrix=lambda rows: (
            (lambda: (_ for _ in ()).throw(AssertionError("expected >1 row")))()
            if len(rows) <= 1 else None
        ),
    ),
    _DatasetCase(
        label="soil_campbell",
        gen_module=gsc,
        modes=SOIL_CAMPBELL_MODES,
        golden_root=SOIL_CAMPBELL_GOLDEN_ROOT,
        rows_func=soil_campbell_rows,
        dataset_value="soil_campbell",
        probe_mode="soil_campbell",
        probe_output_suffix="_summary",
        manifest_mismatch_mode="soil_campbell",
        missing_output_suffix="_field",
        assert_key_prefix=lambda mode, key: (_ for _ in ()).throw(
            AssertionError(f"{mode}: {key}")
        ) if not key.startswith(f"{mode}.") else None,
        sibling_roots=(
            ("canonical", gcn.GOLDEN_ROOT), ("expanded_matrix", gem.GOLDEN_ROOT),
        ),
        assert_scenario_matrix=lambda rows: _assert_default_matrix(rows, {
            "SOI-NOD-": 6, "SOI-DUF-": 6, "SOI-NOIG-": 1,
        }) if len(rows) == 13 else (_ for _ in ()).throw(
            AssertionError(f"expected 13 rows, got {len(rows)}")
        ),
    ),
    _DatasetCase(
        label="emissions_equivalence",
        gen_module=gee,
        modes=EMISSIONS_EQUIVALENCE_MODES,
        golden_root=EMISSIONS_EQUIVALENCE_GOLDEN_ROOT,
        rows_func=emissions_equivalence_rows,
        dataset_value="emissions_equivalence",
        probe_mode="consume",
        probe_output_suffix="_summary",
        manifest_mismatch_mode="consume",
        missing_output_suffix="_components",
        assert_key_prefix=lambda mode, key: (_ for _ in ()).throw(
            AssertionError(f"{mode}: {key}")
        ) if not key.startswith("consume_emissions_equivalence.") else None,
        sibling_roots=(
            ("canonical", gcn.GOLDEN_ROOT), ("expanded_matrix", gem.GOLDEN_ROOT),
            ("soil_campbell", gsc.GOLDEN_ROOT),
        ),
        assert_scenario_matrix=lambda rows: _assert_default_matrix(rows, {
            "default-equiv-": 3, "default-mismatch-control": 1,
        }) if len(rows) == 4 else (_ for _ in ()).throw(
            AssertionError(f"expected 4 rows, got {len(rows)}")
        ),
    ),
    _DatasetCase(
        label="burnup_extended",
        gen_module=gbe,
        modes=BURNUP_EXTENDED_MODES,
        golden_root=BURNUP_EXTENDED_GOLDEN_ROOT,
        rows_func=burnup_extended_rows,
        dataset_value="burnup_extended",
        probe_mode="consume",
        probe_output_suffix="_summary",
        manifest_mismatch_mode="consume",
        missing_output_suffix="_components",
        assert_key_prefix=lambda mode, key: (_ for _ in ()).throw(
            AssertionError(f"{mode}: {key}")
        ) if not key.startswith("consume_burnup_extended.") else None,
        sibling_roots=(
            ("canonical", gcn.GOLDEN_ROOT), ("expanded_matrix", gem.GOLDEN_ROOT),
            ("soil_campbell", gsc.GOLDEN_ROOT), ("emissions_equivalence", gee.GOLDEN_ROOT),
        ),
        assert_scenario_matrix=lambda rows: (
            None if len(rows) == 4 and {r[0] for r in rows} == {
                "rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime",
            } else (_ for _ in ()).throw(AssertionError(f"unexpected rows: {rows}"))
        ),
        fail_closed_no_skip=True,
    ),
)

_CASE_IDS = [case.label for case in _CASES]


@pytest.fixture(autouse=True)
def _toolchain_gate(case: _DatasetCase):
    """
    Gate every test on the live C++ toolchain being available, per-case:
    ``burnup_extended`` FAILS CLOSED (never skips, per the Phase 7
    correction pass's explicit "no new skips" rule); the other three
    datasets skip cleanly, exactly as they always did as separate modules.

    Deliberately function-scoped (not module-scoped): a module-scoped
    ``autouse`` fixture would run once for the whole module regardless of
    parametrization and could skip the ``burnup_extended`` cases too,
    silently defeating the fail-closed requirement this fixture exists to
    enforce - this must be evaluated once per parametrized case.

    :param case: The dataset case under test.
    :returns: None.
    """
    ok, reason = toolchain_status()
    if ok:
        return
    if case.fail_closed_no_skip:
        pytest.fail(
            f"MSVC/CMake/Ninja toolchain unavailable: {reason} - the "
            f"{case.label} dataset drives the real compiled harness and "
            "does not skip (Phase 7 forbids new skips).",
            pytrace=False,
        )
    pytest.skip(f"MSVC/CMake/Ninja toolchain unavailable: {reason}")


@pytest.fixture
def _tmp(case: _DatasetCase, tmp_path, request):
    """
    Return a working directory for one test: pytest's ordinary ``tmp_path``
    for every dataset except ``burnup_extended``, which uses a
    repository-local scratch directory instead (Phase 7 correction pass's
    explicit filesystem boundary).

    :param case: The dataset case under test.
    :param tmp_path: Pytest's built-in fixture.
    :param request: The requesting test node.
    :returns: A ``pathlib.Path``.
    """
    if not case.fail_closed_no_skip:
        yield tmp_path
        return
    with scratch_tempdir("generate_goldens", prefix=request.node.name) as path:
        yield Path(path)


@pytest.fixture(params=_CASES, ids=_CASE_IDS)
def case(request) -> _DatasetCase:
    """Parametrize every test in this module over all four datasets."""
    return request.param


def test_committed_tree_is_present_and_validates(case: _DatasetCase):
    """The committed tree must exist and every manifest in it must
    validate against the live checkout."""
    for mode in case.modes:
        mode_dir = os.path.join(case.golden_root, mode)
        assert os.path.isdir(mode_dir), mode_dir
        path = os.path.join(mode_dir, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        errors = validate_manifest(
            manifest, check_against_live_checkout=True, golden_dir=mode_dir,
        )
        assert not errors, (case.label, mode, errors)


def test_corrupted_committed_golden_is_detected(case: _DatasetCase, _tmp):
    """A byte-corrupted committed CSV must be detected by the SAME
    production comparison function ``--verify-only`` uses."""
    committed_root = str(_tmp / "committed")
    fresh_root = str(_tmp / "fresh")
    case.gen_module.generate_all(committed_root, qualify=False)
    case.gen_module.generate_all(fresh_root, qualify=False)
    target = os.path.join(
        committed_root, case.probe_mode, f"{case.probe_mode}{case.probe_output_suffix}.csv",
    )
    with open(target, "ab") as handle:
        handle.write(b"corrupt\n")

    mismatches = gcn.verify_regeneration(committed_root, fresh_root, list(case.modes))
    assert any(
        message.startswith(f"{case.probe_mode}:") and "content differs" in message
        for message in mismatches
    ), mismatches


def test_deterministic_generation(case: _DatasetCase, _tmp):
    """Two independent generation runs must agree byte-for-byte on every
    CSV and field-for-field on every manifest."""
    run_a = str(_tmp / "a")
    run_b = str(_tmp / "b")
    case.gen_module.generate_all(run_a, qualify=False)
    case.gen_module.generate_all(run_b, qualify=False)
    assert gcn.verify_regeneration(run_a, run_b, list(case.modes)) == []


def test_extra_committed_file_is_detected(case: _DatasetCase, _tmp):
    """An extra file in a freshly generated tree must be reported."""
    committed_root = str(_tmp / "committed")
    fresh_root = str(_tmp / "fresh")
    case.gen_module.generate_all(committed_root, qualify=False)
    case.gen_module.generate_all(fresh_root, qualify=False)
    with open(
            os.path.join(fresh_root, case.probe_mode, "stray.csv"), "w", encoding="utf-8",
    ) as handle:
        handle.write("stray\n")
    mismatches = gcn.verify_regeneration(committed_root, fresh_root, list(case.modes))
    assert any("file set differs" in message for message in mismatches), mismatches


def test_generated_manifests_declare_the_dataset(case: _DatasetCase, _tmp):
    """Every generated manifest must record the dataset's own ``dataset``
    value and cite only that dataset's own tolerance-policy keys."""
    out_root = str(_tmp / "out")
    case.gen_module.generate_all(out_root, qualify=False)
    for mode in case.modes:
        path = os.path.join(out_root, mode, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["dataset"] == case.dataset_value, mode
        assert manifest["tolerance_policy_reference"], mode
        for key in manifest["tolerance_policy_reference"]:
            case.assert_key_prefix(mode, key)
        errors = validate_manifest(
            manifest, check_against_live_checkout=True,
            golden_dir=os.path.join(out_root, mode),
        )
        assert not errors, (mode, errors)


def test_harness_failure_raises(case: _DatasetCase, _tmp):
    """A harness invocation that exits nonzero must raise, never yield a
    golden built from a failed run."""

    class _FakeResult:
        returncode = 1
        stdout = "simulated stdout"
        stderr = "simulated stderr"

    def _fake_run_harness(*_args, **_kwargs):
        return _FakeResult()

    original = case.gen_module.run_harness
    case.gen_module.run_harness = _fake_run_harness
    try:
        with pytest.raises(RuntimeError, match=f"{case.label} golden generation for mode="):
            case.gen_module._generate_one(case.probe_mode, str(_tmp / "out"))
    finally:
        case.gen_module.run_harness = original


def test_manifest_mismatch_is_detected(case: _DatasetCase, _tmp):
    """A manifest field altered after generation must be reported by the
    production comparison function."""
    committed_root = str(_tmp / "committed")
    fresh_root = str(_tmp / "fresh")
    case.gen_module.generate_all(committed_root, qualify=False)
    case.gen_module.generate_all(fresh_root, qualify=False)
    mode = case.manifest_mismatch_mode
    path = os.path.join(committed_root, mode, f"{mode}.manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["build_type"] = "TamperedRelease"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    mismatches = gcn.verify_regeneration(committed_root, fresh_root, list(case.modes))
    assert any("manifest content differs" in message for message in mismatches), mismatches


def test_missing_committed_file_is_detected(case: _DatasetCase, _tmp):
    """A file missing from a freshly generated tree must be reported."""
    committed_root = str(_tmp / "committed")
    fresh_root = str(_tmp / "fresh")
    case.gen_module.generate_all(committed_root, qualify=False)
    case.gen_module.generate_all(fresh_root, qualify=False)
    os.remove(os.path.join(
        fresh_root, case.probe_mode, f"{case.probe_mode}{case.missing_output_suffix}.csv",
    ))
    mismatches = gcn.verify_regeneration(committed_root, fresh_root, list(case.modes))
    assert any("file set differs" in message for message in mismatches), mismatches


def test_generation_never_touches_sibling_trees(case: _DatasetCase, _tmp):
    """Generating one dataset into a temporary root must leave every
    committed sibling-dataset file byte-identical - datasets are siblings,
    never merged."""
    before = {}
    for label, root in case.sibling_roots:
        before[label] = _hash_tree(root)
        assert before[label], f"no committed {label} tree to protect"
    case.gen_module.generate_all(str(_tmp / "out"), qualify=False)
    for label, root in case.sibling_roots:
        assert _hash_tree(root) == before[label], label


def test_reuses_the_canonical_promotion_machinery(case: _DatasetCase):
    """Every dataset's generator must REUSE the canonical (Phase 2)
    transactional promotion helpers, not re-implement a weaker copy of
    them."""
    assert case.gen_module._promote is gcn._promote
    assert case.gen_module._qualify_all is gcn._qualify_all
    assert case.gen_module.verify_regeneration is gcn.verify_regeneration


def test_scenario_matrix_is_non_trivial_and_unique(case: _DatasetCase):
    """Every mode must contribute a real, multi-row scenario matrix with
    unique case IDs, matching the dataset's own known partition."""
    for mode in case.modes:
        case.assert_scenario_matrix(case.rows_func(mode))


def test_wrong_pinned_sha_fails_closed(case: _DatasetCase, _tmp, monkeypatch):
    """``generate_all`` must fail closed before building, qualifying or
    generating against a checkout at any other SHA."""

    def _raise():
        raise ProvenanceError("simulated: checkout is not at the pinned SHA")

    monkeypatch.setattr(case.gen_module, "check_pinned_sha", _raise)
    with pytest.raises(ProvenanceError, match="pinned SHA"):
        case.gen_module.generate_all(str(_tmp / "out"), qualify=False)
