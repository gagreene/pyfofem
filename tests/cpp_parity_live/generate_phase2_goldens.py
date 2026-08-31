#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_phase2_goldens.py - Generate the minimal Phase 2 qualifying golden
dataset (one per mode) with a full provenance manifest.

This is deliberately NOT a Phase 3-7 scientific scenario matrix: its job is
to establish the oracle/manifest infrastructure and produce one qualified,
fully-provenanced golden per mode, using the same canonical rows the
self-test matrix (``test_cpp_harness_contract.py``) already exercises and
qualifies.

Safety/gating contract:

- The pinned-SHA check (``_golden_manifest.check_pinned_sha``) runs before
  the overlay is (re)applied, before configure/build, and again inside
  every manifest build — a checkout at any other SHA fails closed at each
  of those points, not merely recorded afterward.
- The COMPLETE ``test_cpp_harness_contract.py`` module (every mode, run as
  a single real subprocess, unfiltered) must pass before ANY mode's golden
  is generated. This is a real, executed gate, not a docstring claim:
  :func:`_qualify_all` runs it once and raises on any failure, timeout, or
  zero tests collected (pytest exit code 5).
- Generation writes into a fresh temporary directory; the committed
  ``tests/test_data/test_golden_output/phase2/`` tree is only replaced as
  one recoverable whole-tree transaction, under an exclusive writer lock,
  after EVERY mode has generated successfully AND the complete staged
  batch has passed :func:`validate_manifest`. A failure
  partway through never leaves a stale/partial file in the committed tree,
  and no stale file from an earlier run can survive into a fresh
  generation (each mode's committed directory is fully removed and
  replaced, never merged into).

Usage:
    python tests/cpp_parity_live/generate_phase2_goldens.py
    python tests/cpp_parity_live/generate_phase2_goldens.py --verify-only

``--verify-only`` regenerates into a temporary directory and compares
against the committed goldens byte-for-byte (CSVs) / field-for-field
(manifests, excluding the fields that legitimately vary by run location:
``generated_utc`` and ``generating_command``, plus the path-keys — not
values — of the CSV-hash dicts, whose VALUES are still compared) without
overwriting anything, to prove deterministic regeneration (self-test
15/16 at the dataset level).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests._support import PROJECT_ROOT, TEST_GOLDEN_DIR
from tests.cpp_parity_live._golden_manifest import (
    build_manifest,
    check_pinned_sha,
    load_tolerance_policy,
    MODE_OUTPUT_SUFFIXES,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from tests.cpp_parity_live._harness_support import (
    BUILD_DIR,
    HARNESS_EXE,
    SPECIES_CSV,
    _msvc_env,
    ensure_built,
    run_harness,
    toolchain_status,
)
from tests.cpp_parity_live._proc import ProcTimeout, run_bounded
from tests.cpp_parity_live._output_contract import phase2_canonical_policy_keys
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

GOLDEN_ROOT = os.path.join(TEST_GOLDEN_DIR, "phase2")

#: Tolerance-policy keys applicable to the ONE canonical scenario each
#: mode's Phase 2 golden exercises (see tolerance_policy.json). These are
#: the scenarios actually generated below, not every key that exists.
_TOLERANCE_POLICY = load_tolerance_policy()
GOLDEN_TOLERANCE_KEYS = {
    mode: phase2_canonical_policy_keys(mode, _TOLERANCE_POLICY)
    for mode in MODES
}

#: The complete test_cpp_harness_contract.py module (all modes, all
#: branches — see item 4 of the Phase 2 correction pass) takes longer than
#: the old per-mode "-k <mode>" filter did; sized generously since a
#: false-positive timeout here would incorrectly block every mode's golden.
_QUALIFY_TIMEOUT_S = 900


def _build_flags_from_cache() -> str:
    """Read the EFFECTIVE compiler flags CMake actually used, from the real
    CMakeCache.txt — never a guessed/hardcoded string."""
    cache_path = os.path.join(BUILD_DIR, "CMakeCache.txt")
    flags = {}
    try:
        with open(cache_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for key in ("CMAKE_CXX_FLAGS:STRING=", "CMAKE_CXX_FLAGS_DEBUG:STRING=",
                            "CMAKE_BUILD_TYPE:STRING="):
                    if line.startswith(key):
                        flags[key.split(":")[0]] = line.strip().split("=", 1)[1]
    except OSError as exc:
        return f"<could not read {cache_path}: {exc}>"
    return " ".join(f"{k}={v}" for k, v in flags.items()) or "<no relevant flags found in CMakeCache.txt>"


def _compiler_identity() -> str:
    """Return cl.exe's own version banner (its first stderr line), using
    the vcvars64-sourced environment so this works regardless of whether
    the calling shell has cl.exe on PATH."""
    env = _msvc_env()
    if env is None:
        return "unknown (could not source MSVC environment)"
    cl_path = None
    for p in env.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, "cl.exe")
        if os.path.isfile(candidate):
            cl_path = candidate
            break
    if cl_path is None:
        return "unknown (cl.exe not found on sourced PATH)"
    try:
        out = run_bounded([cl_path], env=env, timeout=30).stderr or ""
    except (ProcTimeout, OSError) as exc:
        return f"unknown ({exc})"
    first_line = out.splitlines()[0] if out.splitlines() else "unknown"
    return first_line.strip()


def _generator_toolchain_identity() -> str:
    """Return real CMake + Ninja version strings — never generic prose."""
    env = _msvc_env()
    if env is None:
        return "unknown (could not source MSVC/CMake/Ninja environment)"
    parts = []
    for tool in ("cmake", "ninja"):
        path = shutil.which(tool, path=env.get("PATH", ""))
        if path is None:
            parts.append(f"{tool}=<not found on sourced PATH>")
            continue
        try:
            out = run_bounded([path, "--version"], env=env, timeout=30).stdout.strip()
        except (ProcTimeout, OSError) as exc:
            out = f"<error: {exc}>"
        parts.append(f"{tool}={out.splitlines()[0] if out else '<no output>'}")
    return "; ".join(parts)


def _generate_one(mode: str, out_dir: str) -> str:
    """Generate mode's golden + manifest into out_dir. Returns the
    manifest path. Raises RuntimeError on any failure."""
    m = MODES[mode]
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, mode)
    kw = {"species_csv": SPECIES_CSV} if m["needs_species"] else {}
    res = run_harness(mode, m["header"], [m["row"]], prefix,
                       output_suffixes=m["suffixes"], **kw)
    if res.returncode != 0:
        raise RuntimeError(
            f"golden generation for mode={mode!r} failed (rc={res.returncode}):\n"
            f"stdout={res.stdout}\nstderr={res.stderr}"
        )

    input_csv = [prefix + "_in.csv"]
    output_csvs = [
        prefix + suffix + ".csv" for suffix in m["suffixes"]
        if os.path.isfile(prefix + suffix + ".csv")
    ]
    side_files = {}
    if m["needs_species"]:
        side_files["species_table"] = SPECIES_CSV
    if mode == "consume":
        factor_csv = os.path.join(
            PROJECT_ROOT, "reference", "fofem_cpp", "FOF_UNIX", "Emission_Factors.csv"
        )
        if os.path.isfile(factor_csv):
            side_files["emission_factor_table"] = factor_csv

    manifest = build_manifest(
        harness_mode=mode,
        schema_version="1",
        compiler_identity=_compiler_identity(),
        generator_toolchain=_generator_toolchain_identity(),
        platform=platform.platform(),
        architecture=platform.machine(),
        build_type="Debug",
        build_flags=_build_flags_from_cache(),
        generating_command=(
            f"{HARNESS_EXE} {prefix}_in.csv {prefix}"
            + (f" --species-csv {SPECIES_CSV}" if m["needs_species"] else "")
        ),
        input_csv_paths=input_csv,
        output_csv_paths=output_csvs,
        tolerance_policy_keys=GOLDEN_TOLERANCE_KEYS[mode],
        side_files=side_files,
        now_utc_iso=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = os.path.join(out_dir, f"{mode}.manifest.json")
    write_manifest(manifest_path, manifest)

    errors = validate_manifest(manifest, check_against_live_checkout=True, golden_dir=out_dir)
    if errors:
        raise RuntimeError(
            f"generated manifest for mode={mode!r} failed validation:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
    return manifest_path


#: Sibling-of-out_root suffixes that together form the promotion state
#: machine (see :func:`_recover_interrupted_promotion`). They are fixed so
#: interrupted state is discoverable; :data:`PROMOTE_LOCK_SUFFIX` makes
#: those fixed names safe by admitting exactly one writer at a time.
PROMOTE_STAGING_SUFFIX = ".promote_staging"
PROMOTE_OLD_SUFFIX = ".promote_old"
PROMOTE_LOCK_SUFFIX = ".promote_lock"


def _promote(tmp_root: str, out_root: str, modes) -> None:
    """Promote one complete golden tree under an exclusive writer lock.

    :param tmp_root: Root containing freshly generated mode directories.
    :param out_root: Committed Phase 2 golden root to replace.
    :param modes: Exact mode collection that must be promoted.
    :returns: None.
    :raises RuntimeError: If another live generator owns the promotion
        lock or the staged tree fails validation.
    """
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    with _promotion_lock(out_root):
        _promote_locked(tmp_root, out_root, modes)


def _promote_locked(tmp_root: str, out_root: str, modes) -> None:
    """
    Whole-tree, transactional-and-recoverable (NOT literally atomic —
    Windows has no single-syscall directory-tree replace) promotion of
    every mode in *modes* from *tmp_root* into *out_root*.

    Sequence:

    1. :func:`_recover_interrupted_promotion` first, so a prior run's
       leftover state (caught exception OR a hard interruption/crash) is
       always resolved to one consistent tree before this run does
       anything.
    2. Build a COMPLETE staging tree at ``out_root + PROMOTE_STAGING_SUFFIX``
       (a SIBLING of *out_root*, under the same repo-tracked parent
       directory) via ``shutil.copytree`` for every mode — never
       ``shutil.move``/``os.rename`` of the *tmp_root* tree itself. On
       Windows, ``shutil.move`` of a directory (same-volume rename)
       carries the SOURCE directory's ACL into the destination verbatim —
       ``tmp_root`` is a ``tempfile.TemporaryDirectory()`` under a
       per-user ``%TEMP%`` whose inherited ACL does not grant the repo's
       other principals access (confirmed directly via ``icacls`` in an
       earlier correction pass: a moved directory came out readable only
       to the generating account, missing every principal the repo tree
       itself grants). A NEW directory created via ``copytree`` under the
       staging path instead inherits the STAGING PARENT's ACL (the same
       parent *out_root* itself lives in), which is what every other file
       in the checkout already has.
    3. Validate the COMPLETE staging tree (:func:`_validate_staged_tree`)
       BEFORE touching *out_root* at all.
    4. Swap the WHOLE root, not individual mode directories: rename
       *out_root* to ``out_root + PROMOTE_OLD_SUFFIX`` (if it exists),
       then rename the staging tree to *out_root*. Both are same-parent,
       same-volume renames — the staging tree's ACL (already correct,
       inherited at step 2) survives the rename unchanged.
    5. Remove the old-backup directory only now that the new tree is
       confirmed in place.

    On ANY failure (a caught exception, at any of the steps above), the
    staging tree is discarded and, if the swap had already renamed
    *out_root* aside but not yet completed, that backup is restored —
    the previously committed tree is left completely intact either way.
    Never merges: the staging tree is always built fresh and swapped in
    wholesale, so a stale file from an older golden can never survive
    alongside fresh output.

    A hard interruption (process killed, no exception ever raised) is
    handled by the SAME mechanism from the OUTSIDE: the directory layout
    itself (``out_root`` / ``out_root+PROMOTE_STAGING_SUFFIX`` /
    ``out_root+PROMOTE_OLD_SUFFIX``) IS the recovery record — there is no
    separate JSON/state file that could itself desync from what the
    filesystem actually did. The NEXT call to this function (or a direct
    call to :func:`_recover_interrupted_promotion`) resolves it.
    """
    _recover_interrupted_promotion(out_root)

    staging = out_root + PROMOTE_STAGING_SUFFIX
    old = out_root + PROMOTE_OLD_SUFFIX
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    if os.path.isdir(old):
        shutil.rmtree(old)

    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    os.makedirs(staging)
    try:
        for mode in modes:
            shutil.copytree(os.path.join(tmp_root, mode), os.path.join(staging, mode))
        _validate_staged_tree(staging, modes)

        if os.path.isdir(out_root):
            shutil.move(out_root, old)
        try:
            shutil.move(staging, out_root)
        except Exception:
            if os.path.isdir(old) and not os.path.isdir(out_root):
                shutil.move(old, out_root)
            raise
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
    except Exception:
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
        if os.path.isdir(old) and not os.path.isdir(out_root):
            shutil.move(old, out_root)
        raise


@contextmanager
def _promotion_lock(out_root: str):
    """Hold the single-writer lock for one golden-tree promotion.

    The lock records both PID and process creation time so a dead owner's
    lock is recoverable without mistaking a reused PID for the old process.

    :param out_root: Golden root whose sibling lock file is protected.
    :returns: A context manager yielding while this process owns the lock.
    :raises RuntimeError: If another live process owns the lock or a lock
        record is malformed and cannot be safely classified as stale.
    """
    lock_path = out_root + PROMOTE_LOCK_SUFFIX
    owner = {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock_path, encoding="utf-8") as f:
                    existing = json.load(f)
                process = psutil.Process(int(existing["pid"]))
                same_process = (
                    abs(process.create_time() - float(existing["create_time"]))
                    < 0.001
                )
            except psutil.NoSuchProcess:
                same_process = False
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot safely classify existing promotion lock "
                    f"{lock_path!r}: {exc}"
                ) from exc
            if same_process:
                raise RuntimeError(
                    f"another live generator owns promotion lock "
                    f"{lock_path!r} (pid={existing['pid']})"
                )
            os.unlink(lock_path)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(owner, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            break
    try:
        yield
    finally:
        try:
            with open(lock_path, encoding="utf-8") as f:
                current = json.load(f)
            if current == owner:
                os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _recover_interrupted_promotion(out_root: str) -> None:
    """
    Resolve any leftover state a PRIOR :func:`_promote` call left behind
    — whether it ended in a caught exception or a hard
    interruption/crash that never ran any Python cleanup code at all —
    back to exactly one consistent tree at *out_root*, before a new
    promotion begins.

    Purely directory-presence-driven (no separate state file to desync
    from reality):

    - ``out_root+PROMOTE_OLD_SUFFIX`` exists, *out_root* does NOT: the
      swap was interrupted between its two renames (old root moved
      aside, new root never installed) — restore the old root.
    - ``out_root+PROMOTE_OLD_SUFFIX`` exists, *out_root* ALSO exists: the
      swap itself completed; only the final backup-cleanup step didn't
      run — the backup is now redundant and is discarded.
    - ``out_root+PROMOTE_STAGING_SUFFIX`` exists (regardless of the above):
      an abandoned/no-longer-needed staging tree — always safe to
      discard, since *out_root* (restored above if necessary) is
      authoritative.

    :param out_root: The Phase 2 golden root this promotion targets.
    :return: None. Mutates the filesystem to resolve interrupted state.
    """
    staging = out_root + PROMOTE_STAGING_SUFFIX
    old = out_root + PROMOTE_OLD_SUFFIX

    if os.path.isdir(old) and not os.path.isdir(out_root):
        shutil.move(old, out_root)
    elif os.path.isdir(old) and os.path.isdir(out_root):
        shutil.rmtree(old, ignore_errors=True)

    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)


def _validate_staged_tree(staging: str, modes) -> None:
    """
    Validate a freshly built staging tree BEFORE it is swapped into
    *out_root*: exact mode/file sets plus every live manifest check.

    :param staging: Path to the staging tree (``out_root + PROMOTE_STAGING_SUFFIX``).
    :param modes: Mode names expected under *staging*.
    :raises RuntimeError: If the batch is incomplete, contains extras, or
        any staged manifest fails validation.
    """
    problems = []
    expected_modes = set(modes)
    actual_modes = {
        name for name in os.listdir(staging)
        if os.path.isdir(os.path.join(staging, name))
    }
    if actual_modes != expected_modes:
        problems.append(
            f"staged mode set must be exactly {sorted(expected_modes)}, "
            f"got {sorted(actual_modes)}"
        )
    for mode in modes:
        mode_dir = os.path.join(staging, mode)
        if not os.path.isdir(mode_dir):
            problems.append(f"{mode}: staged directory missing at {mode_dir!r}")
            continue
        expected_files = {
            f"{mode}_in.csv",
            f"{mode}.manifest.json",
            *(f"{mode}{suffix}.csv" for suffix in MODE_OUTPUT_SUFFIXES[mode]),
        }
        actual_files = {
            name for name in os.listdir(mode_dir)
            if os.path.isfile(os.path.join(mode_dir, name))
        }
        if actual_files != expected_files:
            problems.append(
                f"{mode}: staged file set must be exactly "
                f"{sorted(expected_files)}, got {sorted(actual_files)}"
            )
            continue
        manifest_path = os.path.join(mode_dir, f"{mode}.manifest.json")
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{mode}: cannot read staged manifest: {exc}")
            continue
        manifest_errors = validate_manifest(
            manifest, check_against_live_checkout=True, golden_dir=mode_dir,
        )
        problems.extend(f"{mode}: {error}" for error in manifest_errors)
    if problems:
        raise RuntimeError(
            "staged promotion tree failed validation before being swapped "
            "into the committed golden root:\n" + "\n".join(problems)
        )


def _qualify_all() -> None:
    """
    Run the COMPLETE ``test_cpp_harness_contract.py`` module once,
    unfiltered, before generating any mode's golden.

    A per-mode ``-k <mode>`` filter (the prior approach) silently skips
    every harness-contract test whose node ID contains no mode name (e.g.
    shared toolchain/fixture tests) — those tests never ran as part of the
    gate. Running the whole file once closes that gap. "Zero tests
    collected" is detected via pytest's own exit-code semantics (exit code
    5), not a stdout text search, since a reworded pytest summary line
    would silently defeat a text-search check.

    :raises RuntimeError: If qualification fails, times out, or collects
        zero tests.
    """
    test_file = os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", "test_cpp_harness_contract.py"
    )
    try:
        with tempfile.TemporaryDirectory(
                prefix=".phase2-qualification-", dir=PROJECT_ROOT,
        ) as pytest_temp:
            result = run_bounded(
                [
                    sys.executable, "-m", "pytest", "-q", test_file,
                    "--basetemp", os.path.join(pytest_temp, "basetemp"),
                    "-o", f"cache_dir={os.path.join(pytest_temp, 'cache')}",
                ],
                cwd=PROJECT_ROOT, timeout=_QUALIFY_TIMEOUT_S,
            )
    except ProcTimeout as exc:
        raise RuntimeError(
            f"self-test qualification timed out after {_QUALIFY_TIMEOUT_S}s: {exc}"
        )
    if result.returncode == 5:
        raise RuntimeError(
            "self-test qualification collected ZERO tests from "
            f"{test_file} (pytest exit code 5 — 'no tests collected') — "
            f"this would silently defeat the qualification gate:\n{result.stdout}"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"self-test qualification failed (pytest exit={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def generate_all(out_root: str, *, qualify: bool = True) -> None:
    """
    Generate every mode's golden + manifest, gated on pinned-SHA,
    build success, and (if *qualify*) that mode's own self-test subset —
    writing to a fresh temporary directory and promoting to *out_root*
    only after every mode succeeds.

    :param out_root: Destination directory (only touched after full
        success).
    :param qualify: If ``False``, skip the self-test subprocess gate
        (used only by this module's own driver tests, which need to
        exercise generation failure paths quickly and independently of
        the self-test suite's own correctness).
    """
    check_pinned_sha()
    ok, reason = toolchain_status()
    if not ok:
        raise RuntimeError(f"toolchain unavailable: {reason}")
    ok, reason = ensure_built()
    if not ok:
        raise RuntimeError(f"build failed: {reason}")

    if qualify:
        _qualify_all()

    with tempfile.TemporaryDirectory() as tmp_root:
        for mode in MODES:
            _generate_one(mode, os.path.join(tmp_root, mode))
        # Every mode succeeded and validated — promote as a batch. Do this
        # inside the `with` block so `tmp_root` still exists for
        # shutil.move's source side.
        _promote(tmp_root, out_root, list(MODES))


def verify_regeneration(committed_root: str, fresh_root: str, modes) -> List[str]:
    """
    Production comparison of a freshly generated golden tree in
    *fresh_root* against the committed tree in *committed_root* — the SAME
    function used by both ``--verify-only`` and
    ``test_generate_phase2_goldens.py``, so a corrupted/missing/extra/
    manifest-mismatched committed dataset is rejected the same way in both
    places (Phase 2 correction item 3: no test-local approximation).

    Every CSV is compared by SHA-256 (not ``filecmp.dircmp``'s shallow
    stat-based shortcut, which can pass on a changed file with an
    unchanged size/mtime). Every manifest field is compared directly
    except the two genuinely run-location-dependent fields
    (``generated_utc``, ``generating_command``) and
    ``pyfofem_dirty.porcelain`` (the one real, reproduced nondeterminism:
    generating into a fresh temp dir changes ``git status`` output itself,
    since ``tests/test_data/test_golden_output/phase2/`` is untracked —
    this is correct, real provenance per run, not a fair determinism
    target). Input/output CSV hash-map keys are basenames (stable by
    construction — see :func:`sha256_files_by_basename`) and are compared
    directly, not discarded or reduced to sorted values.

    :param committed_root: Directory holding the committed
        ``<mode>/`` subdirectories (e.g. ``GOLDEN_ROOT``).
    :param fresh_root: Directory holding a freshly regenerated
        ``<mode>/`` subdirectories to compare against.
    :param modes: Mode names to check.
    :return: List of human-readable mismatch strings; empty means the
        fresh regeneration matches the committed tree exactly.
    """
    mismatches: List[str] = []
    for mode in modes:
        committed = os.path.join(committed_root, mode)
        fresh = os.path.join(fresh_root, mode)
        if not os.path.isdir(committed):
            mismatches.append(f"{mode}: no committed golden at {committed}")
            continue
        if not os.path.isdir(fresh):
            mismatches.append(f"{mode}: no freshly generated golden at {fresh}")
            continue

        manifest_name = f"{mode}.manifest.json"
        committed_files = {f for f in os.listdir(committed) if f != manifest_name}
        fresh_files = {f for f in os.listdir(fresh) if f != manifest_name}
        missing = committed_files - fresh_files
        extra = fresh_files - committed_files
        if missing or extra:
            mismatches.append(
                f"{mode}: file set differs (missing from fresh regeneration="
                f"{sorted(missing)}, extra in fresh regeneration={sorted(extra)})"
            )

        for name in sorted(committed_files & fresh_files):
            c_path = os.path.join(committed, name)
            f_path = os.path.join(fresh, name)
            if sha256_file(c_path) != sha256_file(f_path):
                mismatches.append(f"{mode}: {name} content differs (SHA-256 mismatch)")

        committed_manifest_path = os.path.join(committed, manifest_name)
        fresh_manifest_path = os.path.join(fresh, manifest_name)
        if not os.path.isfile(committed_manifest_path):
            mismatches.append(f"{mode}: no committed manifest at {committed_manifest_path}")
            continue
        if not os.path.isfile(fresh_manifest_path):
            mismatches.append(f"{mode}: no freshly generated manifest at {fresh_manifest_path}")
            continue
        with open(committed_manifest_path, encoding="utf-8") as f:
            committed_manifest = json.load(f)
        with open(fresh_manifest_path, encoding="utf-8") as f:
            fresh_manifest = json.load(f)

        def _normalize(manifest):
            d = dict(manifest)
            d.pop("generated_utc", None)
            d.pop("generating_command", None)
            dirty = dict(d.get("pyfofem_dirty", {}))
            dirty.pop("porcelain", None)
            d["pyfofem_dirty"] = dirty
            return d

        if _normalize(committed_manifest) != _normalize(fresh_manifest):
            mismatches.append(
                f"{mode}: manifest content differs (fields excluded from "
                "this comparison: generated_utc, generating_command, "
                "pyfofem_dirty.porcelain)"
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                         help="Regenerate into a temp dir and diff against "
                              "the committed goldens instead of overwriting.")
    args = parser.parse_args()

    if args.verify_only:
        with tempfile.TemporaryDirectory() as tmp:
            generate_all(tmp)
            mismatches = verify_regeneration(GOLDEN_ROOT, tmp, list(MODES))
            if mismatches:
                print("DETERMINISM CHECK FAILED:")
                for msg in mismatches:
                    print(" -", msg)
                return 1
            print(
                "DETERMINISM CHECK PASSED: fresh regeneration is byte-identical "
                "(SHA-256 compared) to the committed goldens for every "
                "input/output CSV — no missing/extra files either — and "
                "every manifest field matches except generated_utc, "
                "generating_command, and pyfofem_dirty.porcelain."
            )
            return 0

    generate_all(GOLDEN_ROOT)
    print(f"Generated Phase 2 goldens under {GOLDEN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
