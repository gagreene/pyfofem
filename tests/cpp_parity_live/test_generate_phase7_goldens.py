#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_generate_phase7_goldens.py - Driver tests for the Phase 7 item E
additional-``run_burnup`` golden generator (``generate_phase7_goldens.py``).

The Phase 7 generator deliberately owns no promotion, locking or comparison
logic of its own: it imports and reuses the Phase 2 generator's
``_promote``/``_qualify_all``/``verify_regeneration`` verbatim, exactly as
Phase 4/5/6's generators do. Those mechanisms are already covered end to end
by ``test_generate_phase2_goldens.py``, so this module does NOT duplicate
them. What it covers is everything specific to the Phase 7 dataset: the
fail-closed pinned-SHA gate, deterministic generation of the rotten/sound-
mix/calm-wind/hot-ambient/long-residence-time ``consume`` scenario matrix,
detection of a corrupted/missing/extra/mismatched committed golden through
the SAME production comparison function, the ``dataset``/``consume_p7.*``
manifest fields, reuse (not re-implementation) of the promotion helpers,
and that generating Phase 7 never reads or writes the frozen Phase 2/4/5/6
trees.

Every test here drives the real compiled harness. Per the Phase 7
correction pass (2026-09-06), this module FAILS CLOSED (never skips)
when the live MSVC/CMake/Ninja toolchain is unavailable - Phase 7
explicitly forbids new skips; a missing toolchain in the acceptance/
full lane must be a loud, diagnosable failure (see
:func:`_require_toolchain`), not a silent skip. Every ``tmp_path`` use
below is also overridden (see :func:`tmp_path`) to a repository-local
directory, never the system/user temporary directory.

Function order: private helpers/fixtures first, then public test
functions, each group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import tests.cpp_parity_live.generate_phase2_goldens as gp2
import tests.cpp_parity_live.generate_phase4_goldens as gp4
import tests.cpp_parity_live.generate_phase5_goldens as gp5
import tests.cpp_parity_live.generate_phase6_goldens as gp6
import tests.cpp_parity_live.generate_phase7_goldens as gp7
from tests.cpp_parity_live._golden_manifest import ProvenanceError, validate_manifest
from tests.cpp_parity_live._harness_support import toolchain_status
from tests.cpp_parity_live._phase7_contract import (
    GOLDEN_ROOT as PHASE7_GOLDEN_ROOT,
    PHASE7_MODES,
    phase7_rows,
)
from tests.cpp_parity_live._scratch import scratch_tempdir

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
def _require_toolchain():
    """
    Fail closed (never skip) when the live MSVC/CMake/Ninja toolchain is
    unavailable. Phase 7 explicitly forbids new skips: a missing
    toolchain in the acceptance/full lane is a real gap that must be
    diagnosed and fixed, not silently hidden as a green "skipped" run.
    """
    ok, reason = toolchain_status()
    if not ok:
        pytest.fail(
            f"MSVC/CMake/Ninja toolchain unavailable: {reason} - this "
            "module drives the real compiled harness and does not skip "
            "(Phase 7 forbids new skips). Install/repair the toolchain, "
            "or run only the mocked-injection Phase 7 unit tests "
            "(tests/unit/test_run_burnup_cell_error_codes.py, "
            "test_fire_environment_bounds.py, test_gen_burnup_in_file.py, "
            "test_2d_input_regression.py, test_phase7_run_burnup_parity.py, "
            "test_phase7_golden_tracking.py) instead of this driver "
            "module.",
            pytrace=False,
        )


def test_committed_phase7_tree_is_present_and_validates():
    """The committed Phase 7 tree must exist and every manifest in it must
    validate against the live checkout."""
    for mode in PHASE7_MODES:
        mode_dir = os.path.join(PHASE7_GOLDEN_ROOT, mode)
        assert os.path.isdir(mode_dir), mode_dir
        path = os.path.join(mode_dir, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        errors = validate_manifest(
            manifest, check_against_live_checkout=True, golden_dir=mode_dir,
        )
        assert not errors, (mode, errors)


def test_corrupted_committed_phase7_golden_is_detected(tmp_path):
    """A byte-corrupted committed Phase 7 CSV must be detected by the SAME
    production comparison function ``--verify-only`` uses."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp7.generate_all(committed_root, qualify=False)
    gp7.generate_all(fresh_root, qualify=False)
    target = os.path.join(committed_root, "consume", "consume_summary.csv")
    with open(target, "ab") as handle:
        handle.write(b"corrupt\n")

    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE7_MODES)
    )
    assert any(
        message.startswith("consume:") and "content differs" in message
        for message in mismatches
    ), mismatches


def test_deterministic_phase7_generation(tmp_path):
    """Two independent Phase 7 generation runs must agree byte-for-byte on
    every CSV and field-for-field on every manifest."""
    run_a = str(tmp_path / "a")
    run_b = str(tmp_path / "b")
    gp7.generate_all(run_a, qualify=False)
    gp7.generate_all(run_b, qualify=False)
    assert gp2.verify_regeneration(run_a, run_b, list(PHASE7_MODES)) == []


def test_extra_committed_phase7_file_is_detected(tmp_path):
    """An extra file in a freshly generated tree must be reported."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp7.generate_all(committed_root, qualify=False)
    gp7.generate_all(fresh_root, qualify=False)
    with open(os.path.join(fresh_root, "consume", "stray.csv"), "w",
              encoding="utf-8") as handle:
        handle.write("stray\n")
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE7_MODES)
    )
    assert any("file set differs" in message for message in mismatches), mismatches


def test_generated_phase7_manifests_declare_the_phase7_dataset(tmp_path):
    """Every generated Phase 7 manifest must record ``dataset='phase7'`` and
    cite only ``consume_p7.*`` tolerance-policy keys."""
    out_root = str(tmp_path / "out")
    gp7.generate_all(out_root, qualify=False)
    for mode in PHASE7_MODES:
        path = os.path.join(out_root, mode, f"{mode}.manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["dataset"] == "phase7", mode
        assert manifest["tolerance_policy_reference"], mode
        for key in manifest["tolerance_policy_reference"]:
            assert key.startswith("consume_p7."), (mode, key)
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

    monkeypatch.setattr(gp7, "run_harness", lambda *args, **kwargs: _FakeResult())
    with pytest.raises(RuntimeError, match="Phase 7 golden generation for mode="):
        gp7._generate_one("consume", str(tmp_path / "out"))


def test_manifest_mismatch_is_detected(tmp_path):
    """A manifest field altered after generation must be reported by the
    production comparison function."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp7.generate_all(committed_root, qualify=False)
    gp7.generate_all(fresh_root, qualify=False)
    path = os.path.join(committed_root, "consume", "consume.manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["build_type"] = "TamperedRelease"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE7_MODES)
    )
    assert any("manifest content differs" in message for message in mismatches), \
        mismatches


def test_missing_committed_phase7_file_is_detected(tmp_path):
    """A file missing from a freshly generated tree must be reported."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gp7.generate_all(committed_root, qualify=False)
    gp7.generate_all(fresh_root, qualify=False)
    os.remove(os.path.join(fresh_root, "consume", "consume_components.csv"))
    mismatches = gp2.verify_regeneration(
        committed_root, fresh_root, list(PHASE7_MODES)
    )
    assert any("file set differs" in message for message in mismatches), mismatches


def test_phase7_generation_never_touches_the_phase2_phase4_phase5_or_phase6_tree(tmp_path):
    """Generating Phase 7 into a temporary root must leave every committed
    Phase 2/Phase 4/Phase 5/Phase 6 file byte-identical - the five datasets
    are siblings, never merged."""
    before_p2 = _hash_tree(gp2.GOLDEN_ROOT)
    before_p4 = _hash_tree(gp4.GOLDEN_ROOT)
    before_p5 = _hash_tree(gp5.GOLDEN_ROOT)
    before_p6 = _hash_tree(gp6.GOLDEN_ROOT)
    assert before_p2, "no committed Phase 2 tree to protect"
    assert before_p4, "no committed Phase 4 tree to protect"
    assert before_p5, "no committed Phase 5 tree to protect"
    assert before_p6, "no committed Phase 6 tree to protect"
    gp7.generate_all(str(tmp_path / "out"), qualify=False)
    assert _hash_tree(gp2.GOLDEN_ROOT) == before_p2
    assert _hash_tree(gp4.GOLDEN_ROOT) == before_p4
    assert _hash_tree(gp5.GOLDEN_ROOT) == before_p5
    assert _hash_tree(gp6.GOLDEN_ROOT) == before_p6


def test_phase7_reuses_the_phase2_promotion_machinery():
    """The Phase 7 generator must REUSE the Phase 2 transactional
    promotion helpers, not re-implement a weaker copy of them."""
    assert gp7._promote is gp2._promote
    assert gp7._qualify_all is gp2._qualify_all
    assert gp7.verify_regeneration is gp2.verify_regeneration


def test_phase7_scenario_matrix_is_non_trivial_and_unique():
    """The consume mode must contribute a real, 4-row scenario matrix with
    unique case IDs, one per targeted burnup branch."""
    for mode in PHASE7_MODES:
        rows = phase7_rows(mode)
        assert len(rows) == 4, mode
        case_ids = [row[0] for row in rows]
        assert len(set(case_ids)) == len(case_ids), mode
        assert set(case_ids) == {
            "rot-snd-mix", "calm-wind", "hot-amb-duff", "long-igtime",
        }


def test_wrong_pinned_sha_fails_closed(monkeypatch):
    """``generate_all`` must fail closed before building, qualifying or
    generating against a checkout at any other SHA."""
    def _raise():
        raise ProvenanceError("simulated: checkout is not at the pinned SHA")

    monkeypatch.setattr(gp7, "check_pinned_sha", _raise)
    with scratch_tempdir("generate_phase7_goldens", prefix="wrong-sha") as tmp:
        with pytest.raises(ProvenanceError, match="pinned SHA"):
            gp7.generate_all(tmp, qualify=False)


@pytest.fixture
def tmp_path(request):
    """
    Override pytest's built-in ``tmp_path`` fixture for every test in
    this module: a repository-local, collision-safe directory under
    ``tests/cpp_parity_live/_scratch.py``'s scratch root, never the
    system/user temporary directory, per the Phase 7 correction pass's
    explicit filesystem boundary (2026-09-06). Returns a real
    ``pathlib.Path`` so existing test bodies (``tmp_path / "committed"``,
    etc.) work completely unchanged. Removed on exit regardless of test
    outcome.

    :param request: The requesting test node (used only to namespace the
        directory name for readability under the shared scratch root).
    :return: Yields a ``pathlib.Path`` to the created directory.
    """
    with scratch_tempdir("generate_phase7_goldens", prefix=request.node.name) as path:
        yield Path(path)
