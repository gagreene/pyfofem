#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_generate_phase2_goldens.py - Driver tests for the Phase 2 golden
generator/promoter (``generate_phase2_goldens.py``), per the plan's
explicit requirement for tests covering: wrong pinned SHA, qualification
failure, harness failure, stale/extra/missing output, a corrupted
committed golden, a manifest mismatch, and deterministic successful
generation.

Most of these are exercised via monkeypatching (wrong SHA, qualification
failure, harness failure) so they run fast and do not depend on a
successful build; the promotion/staleness and determinism tests use the
real generator end-to-end (``qualify=False`` to skip the slow per-mode
pytest subprocess gate — a DIFFERENT thing than what these tests verify).

Function order: private helpers/fixtures first, then public test
functions, each group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

import tests.cpp_parity_live.generate_phase2_goldens as gpg
from tests.cpp_parity_live._golden_manifest import ProvenanceError
from tests.cpp_parity_live._harness_support import toolchain_status

pytestmark = pytest.mark.cpp_reference


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_toolchain():
    ok, reason = toolchain_status()
    if not ok:
        pytest.skip(f"MSVC/CMake/Ninja toolchain unavailable: {reason}")


def test_corrupted_committed_golden_is_detected(tmp_path):
    """A byte-corrupted committed golden CSV must be detected by the
    PRODUCTION comparison function (``verify_regeneration`` — the same
    one ``--verify-only`` uses), not a test-local ``filecmp`` shortcut."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gpg.generate_all(committed_root, qualify=False)
    gpg.generate_all(fresh_root, qualify=False)
    committed_csv = os.path.join(committed_root, "litter_eq", "litter_eq.csv")
    with open(committed_csv, "r+b") as f:
        data = f.read()
        f.seek(0)
        f.write(data.replace(b"ok", b"XX", 1) if b"ok" in data else data + b"corrupt")

    mismatches = gpg.verify_regeneration(committed_root, fresh_root, list(gpg.MODES))
    assert any(
        m.startswith("litter_eq:") and "content differs" in m for m in mismatches
    ), f"corrupted committed golden was not detected: {mismatches}"


def test_extra_committed_file_is_detected(tmp_path):
    """A stray extra file in the committed tree (not produced by fresh
    regeneration) must be detected by the production comparator."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gpg.generate_all(committed_root, qualify=False)
    gpg.generate_all(fresh_root, qualify=False)
    stray_path = os.path.join(committed_root, "litter_eq", "unexpected_extra.csv")
    with open(stray_path, "w") as f:
        f.write("not produced by regeneration")

    mismatches = gpg.verify_regeneration(committed_root, fresh_root, list(gpg.MODES))
    assert any(
        m.startswith("litter_eq:") and "file set differs" in m for m in mismatches
    ), f"extra committed file was not detected: {mismatches}"


def test_missing_committed_file_is_detected(tmp_path):
    """A file dropped from the committed tree that fresh regeneration
    still produces must be detected by the production comparator."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gpg.generate_all(committed_root, qualify=False)
    gpg.generate_all(fresh_root, qualify=False)
    os.remove(os.path.join(committed_root, "litter_eq", "litter_eq.csv"))

    mismatches = gpg.verify_regeneration(committed_root, fresh_root, list(gpg.MODES))
    assert any(
        m.startswith("litter_eq:") and "file set differs" in m for m in mismatches
    ), f"missing committed file was not detected: {mismatches}"


def test_deterministic_successful_generation(tmp_path):
    """Two independent generation runs (qualify=False, to isolate this
    from qualification-gate timing) produce a clean pass through the
    PRODUCTION comparison function — byte-identical scientific CSVs and
    field-identical manifests (excluding the fields that legitimately
    vary by run: timestamp, command, and pyfofem_dirty.porcelain)."""
    run_a = str(tmp_path / "a")
    run_b = str(tmp_path / "b")
    gpg.generate_all(run_a, qualify=False)
    gpg.generate_all(run_b, qualify=False)

    mismatches = gpg.verify_regeneration(run_a, run_b, list(gpg.MODES))
    assert mismatches == [], mismatches


def test_harness_failure_raises(monkeypatch, tmp_path):
    """A harness invocation that exits nonzero during generation must
    raise, not silently produce a golden from a failed run."""
    class _FakeResult:
        returncode = 1
        stdout = "simulated stdout"
        stderr = "simulated stderr"

    monkeypatch.setattr(gpg, "run_harness", lambda *a, **k: _FakeResult())
    with pytest.raises(RuntimeError, match="golden generation for mode="):
        gpg._generate_one("litter_eq", str(tmp_path / "out"))


def test_manifest_mismatch_is_detected(tmp_path):
    """A manifest field altered post-generation must be detected as a
    mismatch by the PRODUCTION comparison function against a freshly
    generated manifest."""
    committed_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    gpg.generate_all(committed_root, qualify=False)
    gpg.generate_all(fresh_root, qualify=False)
    manifest_path = os.path.join(committed_root, "litter_eq", "litter_eq.manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["upstream_cpp_sha"] = "f" * 40
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    mismatches = gpg.verify_regeneration(committed_root, fresh_root, list(gpg.MODES))
    assert any(
        m.startswith("litter_eq:") and "manifest content differs" in m for m in mismatches
    ), f"altered manifest field was not detected: {mismatches}"


def test_qualification_failure_raises_and_promotes_nothing(monkeypatch, tmp_path):
    """If the complete harness-contract qualification gate fails, no
    golden for ANY mode is promoted — a partial batch must never reach the
    committed tree. Qualification now runs ONCE for the whole module
    before any mode is generated (see item 4 of the Phase 2 correction
    pass), so a single simulated failure is sufficient to exercise this."""
    def _fail_qualification():
        raise RuntimeError("simulated qualification failure")

    monkeypatch.setattr(gpg, "_qualify_all", _fail_qualification)
    out_root = str(tmp_path / "committed")
    with pytest.raises(RuntimeError, match="simulated qualification failure"):
        gpg.generate_all(out_root, qualify=True)
    assert not os.path.isdir(out_root)


def test_stale_output_never_survives_promotion(monkeypatch, tmp_path):
    """_promote() must fully replace the destination directory — a stale
    file left over from an earlier run at the same path must not survive
    into a freshly promoted golden."""
    out_root = str(tmp_path / "committed")
    mode_dir = os.path.join(out_root, "litter_eq")
    os.makedirs(mode_dir)
    stale_path = os.path.join(mode_dir, "this_should_not_survive.txt")
    with open(stale_path, "w") as f:
        f.write("stale")

    fresh_root = str(tmp_path / "fresh")
    fresh_mode_dir = os.path.join(fresh_root, "litter_eq")
    os.makedirs(fresh_mode_dir)
    with open(os.path.join(fresh_mode_dir, "litter_eq.csv"), "w") as f:
        f.write("fresh content")

    monkeypatch.setattr(gpg, "_validate_staged_tree", lambda *_args: None)
    gpg._promote(fresh_root, out_root, ["litter_eq"])

    assert not os.path.isfile(stale_path), "stale file survived promotion"
    assert os.path.isfile(os.path.join(mode_dir, "litter_eq.csv"))


def test_promote_restores_prior_tree_on_mid_batch_failure(monkeypatch, tmp_path):
    """A failure copying ONE mode partway through a multi-mode promotion
    must leave the PREVIOUSLY COMMITTED tree fully intact for every mode
    in the batch — including the mode(s) already copied successfully
    before the failure — never a half-promoted mix of old and new."""
    out_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    committed_content = {}
    fresh_content = {}
    for mode in ("litter_eq", "mortality"):
        committed_dir = os.path.join(out_root, mode)
        os.makedirs(committed_dir)
        with open(os.path.join(committed_dir, f"{mode}.csv"), "w") as f:
            f.write(f"committed content for {mode}")
        committed_content[mode] = f"committed content for {mode}"

        fresh_dir = os.path.join(fresh_root, mode)
        os.makedirs(fresh_dir)
        with open(os.path.join(fresh_dir, f"{mode}.csv"), "w") as f:
            f.write(f"fresh content for {mode}")
        fresh_content[mode] = f"fresh content for {mode}"

    import shutil as shutil_module
    real_copytree = shutil_module.copytree

    def _fail_on_mortality(src, dst, *a, **k):
        if os.path.basename(dst) == "mortality":
            raise OSError("simulated copy failure for mortality")
        return real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(gpg.shutil, "copytree", _fail_on_mortality)

    with pytest.raises(OSError, match="simulated copy failure"):
        gpg._promote(fresh_root, out_root, ["litter_eq", "mortality"])

    # Every mode's committed content — including litter_eq, which copied
    # successfully BEFORE mortality's simulated failure — must be back
    # exactly as it was, not left as the partially-copied fresh content.
    for mode in ("litter_eq", "mortality"):
        with open(os.path.join(out_root, mode, f"{mode}.csv")) as f:
            assert f.read() == committed_content[mode], (
                f"{mode}: committed tree was not fully restored after a "
                "mid-batch promotion failure"
            )
    backup_root = out_root + ".promote_backup"
    assert not os.path.isdir(backup_root), "promotion backup directory was not cleaned up"


def test_promote_leaves_no_directory_when_nothing_was_previously_committed(monkeypatch, tmp_path):
    """A failure during the FIRST-EVER promotion (no prior committed tree
    to restore) must not leave a partially-copied directory behind."""
    out_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    fresh_dir = os.path.join(fresh_root, "litter_eq")
    os.makedirs(fresh_dir)
    with open(os.path.join(fresh_dir, "litter_eq.csv"), "w") as f:
        f.write("fresh content")

    def _always_fail(src, dst, *a, **k):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(gpg.shutil, "copytree", _always_fail)

    with pytest.raises(OSError, match="simulated copy failure"):
        gpg._promote(fresh_root, out_root, ["litter_eq"])

    assert not os.path.isdir(os.path.join(out_root, "litter_eq")), (
        "a partially-copied mode directory survived a first-promotion failure"
    )


def _assert_single_generation_no_mixing(mode_dir, expected_marker):
    """Every file under *mode_dir* must carry the SAME generation marker —
    proves a promotion never leaves a mixed old/new mode collection."""
    for name in os.listdir(mode_dir):
        with open(os.path.join(mode_dir, name)) as f:
            content = f.read()
        assert expected_marker in content, (
            f"{name} does not carry the expected marker {expected_marker!r} "
            f"(mixed old/new content survived promotion): {content!r}"
        )


def test_recover_interrupted_promotion_restores_old_root_when_swap_incomplete(tmp_path):
    """Simulates the hard-interruption point between the two swap renames:
    the old root was already moved aside to PROMOTE_OLD_SUFFIX, but the
    staging tree was never renamed into out_root's place (process died in
    between — no exception, no Python cleanup ever ran). Recovery must
    restore the old root exactly."""
    out_root = str(tmp_path / "committed")
    old = out_root + gpg.PROMOTE_OLD_SUFFIX
    os.makedirs(os.path.join(old, "litter_eq"))
    with open(os.path.join(old, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("OLD_MARKER content")
    staging = out_root + gpg.PROMOTE_STAGING_SUFFIX
    os.makedirs(os.path.join(staging, "litter_eq"))
    with open(os.path.join(staging, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("NEW_MARKER content — never installed")

    gpg._recover_interrupted_promotion(out_root)

    assert os.path.isdir(out_root)
    assert not os.path.isdir(old)
    assert not os.path.isdir(staging)
    _assert_single_generation_no_mixing(os.path.join(out_root, "litter_eq"), "OLD_MARKER")


def test_recover_interrupted_promotion_discards_redundant_backup_when_swap_completed(tmp_path):
    """Simulates the hard-interruption point AFTER both swap renames
    succeeded, but before the final old-backup cleanup ran. Recovery must
    discard the now-redundant backup and leave the (already correct) new
    root untouched."""
    out_root = str(tmp_path / "committed")
    os.makedirs(os.path.join(out_root, "litter_eq"))
    with open(os.path.join(out_root, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("NEW_MARKER content — already installed")
    old = out_root + gpg.PROMOTE_OLD_SUFFIX
    os.makedirs(os.path.join(old, "litter_eq"))
    with open(os.path.join(old, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("OLD_MARKER content — now redundant")

    gpg._recover_interrupted_promotion(out_root)

    assert os.path.isdir(out_root)
    assert not os.path.isdir(old)
    _assert_single_generation_no_mixing(os.path.join(out_root, "litter_eq"), "NEW_MARKER")


def test_recover_interrupted_promotion_discards_abandoned_staging_when_root_intact(tmp_path):
    """Simulates the hard-interruption point during staging BUILD/
    validation (before the swap even starts): out_root still holds its
    previous, untouched, valid content; a partial staging tree sits
    alongside it. Recovery must discard the abandoned staging tree and
    leave out_root exactly as it was."""
    out_root = str(tmp_path / "committed")
    os.makedirs(os.path.join(out_root, "litter_eq"))
    with open(os.path.join(out_root, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("OLD_MARKER content — untouched")
    staging = out_root + gpg.PROMOTE_STAGING_SUFFIX
    os.makedirs(os.path.join(staging, "litter_eq"))
    with open(os.path.join(staging, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("partial NEW content, never validated")

    gpg._recover_interrupted_promotion(out_root)

    assert os.path.isdir(out_root)
    assert not os.path.isdir(staging)
    _assert_single_generation_no_mixing(os.path.join(out_root, "litter_eq"), "OLD_MARKER")


def test_promote_self_heals_from_simulated_hard_interruption_before_proceeding(monkeypatch, tmp_path):
    """A NEW `_promote()` call, seeing the leftover state a prior hard
    interruption left behind (mid-swap: old moved aside, staging never
    installed), must self-heal via `_recover_interrupted_promotion` FIRST
    and then complete a fresh, correct promotion — not trip over the
    stale state or produce a mixed tree."""
    out_root = str(tmp_path / "committed")
    old = out_root + gpg.PROMOTE_OLD_SUFFIX
    os.makedirs(os.path.join(old, "litter_eq"))
    with open(os.path.join(old, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("OLD_MARKER content")

    fresh_root = str(tmp_path / "fresh")
    fresh_dir = os.path.join(fresh_root, "litter_eq")
    os.makedirs(fresh_dir)
    with open(os.path.join(fresh_dir, "litter_eq.csv"), "w") as f:
        f.write("NEW_MARKER content")

    monkeypatch.setattr(gpg, "_validate_staged_tree", lambda *_args: None)
    gpg._promote(fresh_root, out_root, ["litter_eq"])

    assert not os.path.isdir(old)
    assert not os.path.isdir(out_root + gpg.PROMOTE_STAGING_SUFFIX)
    _assert_single_generation_no_mixing(os.path.join(out_root, "litter_eq"), "NEW_MARKER")


def test_promote_restores_old_root_when_second_swap_rename_fails(monkeypatch, tmp_path):
    """If the SECOND swap rename (staging -> out_root) itself fails (e.g.
    a transient filesystem error), the already-renamed-aside old root must
    be restored immediately — out_root must never be left missing."""
    out_root = str(tmp_path / "committed")
    os.makedirs(os.path.join(out_root, "litter_eq"))
    with open(os.path.join(out_root, "litter_eq", "litter_eq.csv"), "w") as f:
        f.write("OLD_MARKER content")

    fresh_root = str(tmp_path / "fresh")
    fresh_dir = os.path.join(fresh_root, "litter_eq")
    os.makedirs(fresh_dir)
    with open(os.path.join(fresh_dir, "litter_eq.csv"), "w") as f:
        f.write("NEW_MARKER content")

    real_move = gpg.shutil.move
    staging_path = out_root + gpg.PROMOTE_STAGING_SUFFIX

    def _fail_only_on_staging_swap(src, dst, *a, **k):
        if src == staging_path:
            raise OSError("simulated failure renaming staging into place")
        return real_move(src, dst, *a, **k)

    monkeypatch.setattr(gpg.shutil, "move", _fail_only_on_staging_swap)
    monkeypatch.setattr(gpg, "_validate_staged_tree", lambda *_args: None)

    with pytest.raises(OSError, match="simulated failure renaming staging"):
        gpg._promote(fresh_root, out_root, ["litter_eq"])

    assert os.path.isdir(out_root), "out_root must never be left missing"
    assert not os.path.isdir(out_root + gpg.PROMOTE_OLD_SUFFIX)
    assert not os.path.isdir(staging_path)
    _assert_single_generation_no_mixing(os.path.join(out_root, "litter_eq"), "OLD_MARKER")


def test_promote_never_produces_mixed_old_new_mode_collection(monkeypatch, tmp_path):
    """Across two modes, a successful promotion must never leave one
    mode's fresh content alongside another mode's stale content — the
    swap is whole-tree, not per-mode."""
    out_root = str(tmp_path / "committed")
    for mode in ("litter_eq", "mortality"):
        d = os.path.join(out_root, mode)
        os.makedirs(d)
        with open(os.path.join(d, f"{mode}.csv"), "w") as f:
            f.write("OLD_MARKER content")

    fresh_root = str(tmp_path / "fresh")
    for mode in ("litter_eq", "mortality"):
        d = os.path.join(fresh_root, mode)
        os.makedirs(d)
        with open(os.path.join(d, f"{mode}.csv"), "w") as f:
            f.write("NEW_MARKER content")

    monkeypatch.setattr(gpg, "_validate_staged_tree", lambda *_args: None)
    gpg._promote(fresh_root, out_root, ["litter_eq", "mortality"])

    for mode in ("litter_eq", "mortality"):
        _assert_single_generation_no_mixing(os.path.join(out_root, mode), "NEW_MARKER")


def test_promoted_golden_is_readable_and_enumerable_from_fresh_subprocess(monkeypatch, tmp_path):
    """The Codex-demonstrated failure mode: files promoted via
    ``shutil.move`` carried a restrictive temp-dir ACL that locked out
    every principal but the generating account, so pytest collection from
    a DIFFERENT account/session failed with PermissionError. Proves the
    fix by actually opening and enumerating every promoted file from a
    brand-new subprocess (not just this same process/account)."""
    out_root = str(tmp_path / "committed")
    fresh_root = str(tmp_path / "fresh")
    fresh_dir = os.path.join(fresh_root, "litter_eq")
    os.makedirs(fresh_dir)
    with open(os.path.join(fresh_dir, "litter_eq.csv"), "w", encoding="utf-8") as f:
        f.write("case_id,outcome\nc1,ok\n")
    with open(os.path.join(fresh_dir, "litter_eq.manifest.json"), "w", encoding="utf-8") as f:
        f.write("{}")

    monkeypatch.setattr(gpg, "_validate_staged_tree", lambda *_args: None)
    gpg._promote(fresh_root, out_root, ["litter_eq"])

    mode_dir = os.path.join(out_root, "litter_eq")
    script = (
        "import os, sys\n"
        "mode_dir = sys.argv[1]\n"
        "names = sorted(os.listdir(mode_dir))\n"
        "assert names == ['litter_eq.csv', 'litter_eq.manifest.json'], names\n"
        "for name in names:\n"
        "    with open(os.path.join(mode_dir, name), encoding='utf-8') as f:\n"
        "        f.read()\n"
        "print('OK')\n"
    )
    from tests.cpp_parity_live._proc import run_bounded
    result = run_bounded(
        [sys.executable, "-c", script, mode_dir], timeout=30,
    )
    assert result.returncode == 0, (
        f"a fresh subprocess could not enumerate/read the promoted golden "
        f"directory (stdout={result.stdout!r} stderr={result.stderr!r})"
    )
    assert "OK" in result.stdout

    if os.name == "nt":
        icacls = run_bounded(["icacls", mode_dir], timeout=30)
        assert icacls.returncode == 0, icacls.stderr
        parent_icacls = run_bounded(["icacls", out_root], timeout=30)
        assert parent_icacls.returncode == 0, parent_icacls.stderr

        def _principals(icacls_output):
            # icacls prints "<path> <principal>:(perm)" on the first line
            # and "                 <principal>:(perm)" on continuations.
            principals = set()
            for line in icacls_output.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("Successfully"):
                    continue
                text = stripped.split(None, 1)[-1] if stripped == line.strip() else stripped
                if ":" in text:
                    principals.add(text.split(":", 1)[0].strip())
            return principals

        promoted_principals = _principals(icacls.stdout)
        parent_principals = _principals(parent_icacls.stdout)
        # The promoted directory must not be MORE restrictive than its own
        # parent — every principal the parent grants must still be present
        # on the freshly promoted directory (the exact ACL-inheritance
        # property a temp-dir-ACL `shutil.move` violated).
        missing = parent_principals - promoted_principals
        assert not missing, (
            f"promoted directory is missing principal(s) granted by its "
            f"parent: {missing} (promoted={promoted_principals!r}, "
            f"parent={parent_principals!r})"
        )


def test_wrong_pinned_sha_fails_closed(monkeypatch):
    """generate_all() must fail closed (never build/qualify/generate)
    against a checkout at any SHA other than the one pinned constant."""
    monkeypatch.setattr(gpg, "check_pinned_sha", lambda: (_ for _ in ()).throw(
        ProvenanceError("simulated: checkout is not at the pinned SHA")
    ))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ProvenanceError, match="pinned SHA"):
            gpg.generate_all(tmp, qualify=False)


def test_promotion_lock_rejects_a_live_concurrent_writer(tmp_path):
    out_root = str(tmp_path / "committed")
    os.makedirs(tmp_path, exist_ok=True)
    with gpg._promotion_lock(out_root):
        with pytest.raises(RuntimeError, match="another live generator"):
            with gpg._promotion_lock(out_root):
                pass
    assert not os.path.exists(out_root + gpg.PROMOTE_LOCK_SUFFIX)


def test_promotion_lock_recovers_a_dead_owner(tmp_path):
    out_root = str(tmp_path / "committed")
    lock_path = out_root + gpg.PROMOTE_LOCK_SUFFIX
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump({"pid": 2147483647, "create_time": 0.0}, f)
    with gpg._promotion_lock(out_root):
        assert os.path.isfile(lock_path)
    assert not os.path.exists(lock_path)


def test_validate_staged_tree_checks_complete_batch_and_manifests(monkeypatch, tmp_path):
    staging = str(tmp_path / "staging")
    modes = ["litter_eq", "mortality"]
    for mode in modes:
        mode_dir = os.path.join(staging, mode)
        os.makedirs(mode_dir)
        names = {
            f"{mode}_in.csv",
            f"{mode}.manifest.json",
            *(f"{mode}{suffix}.csv" for suffix in gpg.MODE_OUTPUT_SUFFIXES[mode]),
        }
        for name in names:
            with open(os.path.join(mode_dir, name), "w", encoding="utf-8") as f:
                f.write("{}" if name.endswith(".json") else "header\n")

    validated = []

    def _record_validation(manifest, *, check_against_live_checkout, golden_dir):
        validated.append((manifest, check_against_live_checkout, golden_dir))
        return []

    monkeypatch.setattr(gpg, "validate_manifest", _record_validation)
    gpg._validate_staged_tree(staging, modes)
    assert [os.path.basename(item[2]) for item in validated] == modes
    assert all(item[1] is True for item in validated)

    os.remove(os.path.join(staging, "mortality", "mortality_in.csv"))
    with pytest.raises(RuntimeError, match="staged file set must be exactly"):
        gpg._validate_staged_tree(staging, modes)
