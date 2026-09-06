#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_generate_phase6_goldens.py - Driver tests for the Phase 6
default-emissions-equivalence golden generator
(``generate_phase6_goldens.py``).

The Phase 6 generator deliberately owns no promotion, locking or comparison
logic of its own: it imports and reuses the Phase 2 generator's
``_promote``/``_qualify_all``/``verify_regeneration`` verbatim, exactly as
Phase 4/5's generators do. Those mechanisms are already covered end to end
by ``test_generate_phase2_goldens.py``, so this module does NOT duplicate
them. What it covers is everything specific to the Phase 6 dataset: the
fail-closed pinned-SHA gate, deterministic generation of the same-group/
mismatched-group ``consume`` scenario matrix, detection of a corrupted/
missing/extra/mismatched committed golden through the SAME production
comparison function, the ``dataset``/``consume_p6.*`` manifest fields, reuse
(not re-implementation) of the promotion helpers, and that generating
Phase 6 never reads or writes the frozen Phase 2/4/5 trees.

Every test here drives the real compiled harness, so the whole module
skips cleanly when the MSVC/CMake/Ninja toolchain is unavailable.

Function order: private helpers/fixtures first, then public test
functions, each group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import pytest

import tests.cpp_parity_live.generate_phase2_goldens as gp2
import tests.cpp_parity_live.generate_phase4_goldens as gp4
import tests.cpp_parity_live.generate_phase5_goldens as gp5
import tests.cpp_parity_live.generate_phase6_goldens as gp6
from tests.cpp_parity_live._golden_manifest import ProvenanceError, validate_manifest
from tests.cpp_parity_live._harness_support import toolchain_status
from tests.cpp_parity_live._phase6_contract import (
    GOLDEN_ROOT as PHASE6_GOLDEN_ROOT,
    PHASE6_MODES,
    phase6_rows,
)

pytestmark = pytest.mark.cpp_reference


def _hash_tree(root):
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


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_toolchain():
    """Skip the whole module when no live C++ toolchain is available."""
    ok, reason = toolchain_status()
    if not ok:
        pytest.skip(f"MSVC/CMake/Ninja toolchain unavailable: {reason}")


def test_committed_phase6_tree_is_present_and_validates():
    """The committed Phase 6 tree must exist and every manifest in it must
    validate against the live checkout."""
    for mode in PHASE6_MODES:
        mode_dir = os.path.join(PHASE6_GOLDEN_ROOT, mode)
        assert os.path.isdir(mode_dir), mode_dir
        path = os.path.join(mode_dir, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        errors = validate_manifest(
            manifest, check_against_live_checkout=True, golden_dir=mode_dir,
        )
        assert not errors, (mode, errors)


def test_corrupted_committed_phase6_golden_is_detected(tmp_path):
    """A byte-corrupted committed Phase 6 CSV must be detected by the SAME
    production comparison function ``--verify-only`` uses."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp6.generate_all(committed_root, qualify=False)
    gp6.generate_all(fresh_root, qualify=False)
    target = os.path.join(committed_root, "consume", "consume_summary.csv")
    with open(target, "ab") as handle:
        handle.write(b"corrupt\n")

    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE6_MODES)
    )
    assert any(
        message.startswith("consume:") and "content differs" in message
        for message in mismatches
    ), mismatches


def test_deterministic_phase6_generation(tmp_path):
    """Two independent Phase 6 generation runs must agree byte-for-byte on
    every CSV and field-for-field on every manifest."""
    run_a = str(tmp_path / "a")
    run_b = str(tmp_path / "b")
    gp6.generate_all(run_a, qualify=False)
    gp6.generate_all(run_b, qualify=False)
    assert gp2.verify_regeneration(run_a, run_b, list(PHASE6_MODES)) == []


def test_extra_committed_phase6_file_is_detected(tmp_path):
    """An extra file in a freshly generated tree must be reported."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp6.generate_all(committed_root, qualify=False)
    gp6.generate_all(fresh_root, qualify=False)
    with open(os.path.join(fresh_root, "consume", "stray.csv"), "w",
              encoding="utf-8") as handle:
        handle.write("stray\n")
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE6_MODES)
    )
    assert any("file set differs" in message for message in mismatches), mismatches


def test_generated_phase6_manifests_declare_the_phase6_dataset(tmp_path):
    """Every generated Phase 6 manifest must record ``dataset='phase6'`` and
    cite only ``consume_p6.*`` tolerance-policy keys."""
    out_root = str(tmp_path / "out")
    gp6.generate_all(out_root, qualify=False)
    for mode in PHASE6_MODES:
        path = os.path.join(out_root, mode, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["dataset"] == "phase6", mode
        assert manifest["tolerance_policy_reference"], mode
        for key in manifest["tolerance_policy_reference"]:
            assert key.startswith("consume_p6."), (mode, key)
        errors = validate_manifest(
            manifest, check_against_live_checkout=True,
            golden_dir=os.path.join(out_root, mode),
        )
        assert not errors, (mode, errors)


def test_harness_failure_raises(monkeypatch, tmp_path):
    """A harness invocation that exits nonzero must raise, never yield a
    golden built from a failed run."""
    class _FakeResult:
        returncode = 1
        stdout = "simulated stdout"
        stderr = "simulated stderr"

    monkeypatch.setattr(gp6, "run_harness", lambda *args, **kwargs: _FakeResult())
    with pytest.raises(RuntimeError, match="Phase 6 golden generation for mode="):
        gp6._generate_one("consume", str(tmp_path / "out"))


def test_manifest_mismatch_is_detected(tmp_path):
    """A manifest field altered after generation must be reported by the
    production comparison function."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp6.generate_all(committed_root, qualify=False)
    gp6.generate_all(fresh_root, qualify=False)
    path = os.path.join(committed_root, "consume", "consume.manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["build_type"] = "TamperedRelease"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE6_MODES)
    )
    assert any("manifest content differs" in message for message in mismatches), \
        mismatches


def test_missing_committed_phase6_file_is_detected(tmp_path):
    """A file missing from a freshly generated tree must be reported."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp6.generate_all(committed_root, qualify=False)
    gp6.generate_all(fresh_root, qualify=False)
    os.remove(os.path.join(fresh_root, "consume", "consume_components.csv"))
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE6_MODES)
    )
    assert any("file set differs" in message for message in mismatches), mismatches


def test_phase6_generation_never_touches_the_phase2_phase4_or_phase5_tree(tmp_path):
    """Generating Phase 6 into a temporary root must leave every committed
    Phase 2/Phase 4/Phase 5 file byte-identical - the four datasets are
    siblings, never merged."""
    before_p2 = _hash_tree(gp2.GOLDEN_ROOT)
    before_p4 = _hash_tree(gp4.GOLDEN_ROOT)
    before_p5 = _hash_tree(gp5.GOLDEN_ROOT)
    assert before_p2, "no committed Phase 2 tree to protect"
    assert before_p4, "no committed Phase 4 tree to protect"
    assert before_p5, "no committed Phase 5 tree to protect"
    gp6.generate_all(str(tmp_path / "out"), qualify=False)
    assert _hash_tree(gp2.GOLDEN_ROOT) == before_p2
    assert _hash_tree(gp4.GOLDEN_ROOT) == before_p4
    assert _hash_tree(gp5.GOLDEN_ROOT) == before_p5


def test_phase6_reuses_the_phase2_promotion_machinery():
    """The Phase 6 generator must REUSE the Phase 2 transactional
    promotion helpers, not re-implement a weaker copy of them."""
    assert gp6._promote is gp2._promote
    assert gp6._qualify_all is gp2._qualify_all
    assert gp6.verify_regeneration is gp2.verify_regeneration


def test_phase6_scenario_matrix_is_non_trivial_and_unique():
    """The consume mode must contribute a real, multi-row scenario matrix
    with unique case IDs covering both branches (equivalence + negative
    control)."""
    for mode in PHASE6_MODES:
        rows = phase6_rows(mode)
        assert len(rows) == 4, mode
        case_ids = [row[0] for row in rows]
        assert len(set(case_ids)) == len(case_ids), mode
        assert sum(1 for c in case_ids if c.startswith("default-equiv-")) == 3
        assert sum(1 for c in case_ids if c == "default-mismatch-control") == 1


def test_wrong_pinned_sha_fails_closed(monkeypatch):
    """``generate_all`` must fail closed before building, qualifying or
    generating against a checkout at any other SHA."""
    def _raise():
        raise ProvenanceError("simulated: checkout is not at the pinned SHA")

    monkeypatch.setattr(gp6, "check_pinned_sha", _raise)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ProvenanceError, match="pinned SHA"):
            gp6.generate_all(tmp, qualify=False)
