#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_golden_manifest.py - Provenance manifest builder/validator for the
C++-oracle golden datasets.

Two datasets exist (see :data:`VALID_DATASETS`): ``phase2``, the frozen
canonical one-row-per-mode dataset, and ``phase4``, the Tier-2 scenario
matrix. Both use the same six qualified harness modes, the same manifest
schema, and the same fail-closed checks; they differ only in which
tolerance-policy routes and generator-source files they cite. A manifest
with no ``dataset`` field is a ``phase2`` manifest — that field was added by
Phase 4 and is deliberately omitted for ``phase2`` so the already-committed
Phase 2 manifests remain byte-identical.

Every accepted golden dataset carries a manifest recording: the pinned
upstream C++ SHA (checked against a hardcoded constant, not merely
recorded), the full overlay directory's combined digest and per-file
hashes (all six maintained files under ``reference/fofem_cpp_overlay/``,
not only ``source/``), the harness mode/schema version, real
compiler/toolchain/platform/build identity, the exact generating command,
every input/output/side-file hash (repo-relative paths, re-verified
against the actual files at validation time — not just recorded), the UTC
generation timestamp, the pyfofem commit (and full working-tree dirty
status) used, documented expected divergences drawn from verified Gate 0
findings, and the specific tolerance-policy keys that apply. See
``development/plans/gate0/01-provenance.md`` for the original field list;
this module additionally enforces every one of those fields rather than
merely recording them.

Nothing here computes provenance from stale README prose — every field is
recomputed directly from the live checkout at generation/validation time.

Function order: private helpers first (``_dataset_of``,
``_is_safe_repo_relative_path``, ``_run_git``, alphabetized without their
leading underscore), then public top-level functions, alphabetized, per
AGENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from tests._support import CPP_REFERENCE_DIR, PROJECT_ROOT
from tests.cpp_parity_live._output_contract import (
    phase2_canonical_divergence_keys,
    phase2_canonical_policy_keys,
)
from tests.cpp_parity_live._proc import run_bounded

#: Bounded timeout (seconds) for every git subprocess this module spawns.
TIMEOUT_GIT_S = 30

#: The one pinned upstream C++ commit Phase 2 goldens may be generated
#: against. A live checkout at any other SHA must fail closed — recording
#: a different SHA in a manifest is not sufficient (Codex finding).
PINNED_UPSTREAM_SHA = "78f97f093ee7d1c77b3cd2622b2bd7248036c1e4"

OVERLAY_ROOT = os.path.join(PROJECT_ROOT, "reference", "fofem_cpp_overlay")

#: The exact, closed set of files the overlay maintains. Anything missing
#: or anything extra under OVERLAY_ROOT is a provenance error, not a
#: warning.
REQUIRED_OVERLAY_FILES = frozenset({
    "README.md",
    "patches/CMakeLists.remote_to_local.patch",
    "source/CMakeLists.txt",
    "source/FOFEM_CPP_CODEBASE.md",
    "source/FOF_UNIX/test_harness.cpp",
    "source/compile_test.bat",
})

#: Dataset a manifest belongs to when it carries no ``dataset`` field. The
#: committed Phase 2 manifests predate the field and must stay byte-identical,
#: so their dataset is inferred rather than recorded (Phase 4 addition).
DEFAULT_DATASET = "phase2"

#: Every golden dataset this module knows how to build/validate a manifest
#: for. ``phase2`` is the frozen canonical single-row-per-mode dataset;
#: ``phase4`` is the Tier-2 scenario-matrix dataset (see
#: ``_phase4_contract.py``), reusing the same six qualified harness modes;
#: ``phase5`` is the ``soil_campbell`` scenario-matrix dataset (see
#: ``_phase5_contract.py``) — the one dataset with its own harness mode
#: rather than reusing the Phase 2 six.
VALID_DATASETS = frozenset({"phase2", "phase4", "phase5"})

#: Human-readable label per dataset, used in error text only. Kept
#: separate from the dataset key so the Phase 2 wording that existing
#: approved tests assert on ("canonical Phase 2 scenario contract")
#: stays byte-stable while Phase 4/5 get their own labels.
DATASET_LABELS = {"phase2": "Phase 2", "phase4": "Phase 4", "phase5": "Phase 5"}

#: This module's own generation-time dependencies. Hashed into a manifest
#: whenever the parent repo is dirty (uncommitted), since a dirty-tree
#: generation cannot point at a stable commit for its own generator logic.
#: This list is the ``phase2`` dataset's list and is deliberately FROZEN:
#: the committed Phase 2 manifests cite it verbatim. Phase 4 declares its
#: own, larger list (see :func:`generator_source_files_for_dataset`).
GENERATOR_SOURCE_FILES = [
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "_golden_manifest.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "_harness_support.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "_output_contract.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "_proc.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "generate_phase2_goldens.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "test_cpp_harness_contract.py"),
    os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "tolerance_policy.json"),
]

TOLERANCE_POLICY_PATH = os.path.join(
    PROJECT_ROOT, "tests", "cpp_parity_live", "tolerance_policy.json"
)

VALID_HARNESS_MODES = frozenset({
    "consume", "litter_eq", "shrub_herb_eq", "mortality", "bark_thick", "canopy_cover",
    "soil_campbell",
})

#: The input/output schema version each harness mode declares. This is
#: PER MODE, not global: ``gate0/05-harness-contract.md`` declares each
#: mode's schema independently (§2 "Mode ``consume`` (schema v1)",
#: §5 "Mode ``mortality`` (schema v2)", ...), and revising one mode
#: must not silently redefine what an archived CSV of another mode's
#: v1 means. Mirrors ``test_harness.cpp``'s own ``MODES[]`` table,
#: whose ``schema_version`` field main() validates the magic line
#: against; ``test_cpp_harness_contract.py`` proves the two agree by
#: running the real binary, so this is not a silently-drifting copy.
#:
#: ``mortality`` is at v2 because Phase 4's correction pass added a
#: ``density_tpa`` column (``d_MIS.f_Den``, required by ``ValidInput``
#: at ``fof_mrt.cpp:1854-1856`` for every CroDam row) and renamed the
#: misnamed ``ckr_pct`` to ``ckr_rating``.
#:
#: ``soil_campbell`` (Phase 5) is at v1. Its schema appends three columns
#: (``duff_load_tac``, ``duff_consumed_pct``, ``duff_moist_pct``) beyond
#: the 13 the approved ``gate0/05-harness-contract.md`` §7 contract
#: listed — a Phase 5 item-1 audit finding, not a silent redesign:
#: ``SD_Init()`` (``fof_sd.cpp:258-261``) reads ``d_SI.f_DufLoaPre`` /
#: ``f_DufConPer`` / ``f_DufMoi`` on the Duff route, and ``SI_Init()``
#: (``fof_sh.cpp:221-231``) never initialises any of the three — without
#: them the Duff route ran on uninitialised memory. See the Phase 5
#: report for the full evidence trail.
MODE_SCHEMA_VERSIONS = {
    "consume": "1",
    "litter_eq": "1",
    "shrub_herb_eq": "1",
    "mortality": "2",
    "bark_thick": "1",
    "canopy_cover": "1",
    "soil_campbell": "1",
}

#: Every schema version any mode declares. Used only for the coarse
#: "is this a version string we know at all" check; the binding check
#: is the exact per-mode equality against :data:`MODE_SCHEMA_VERSIONS`.
VALID_SCHEMA_VERSIONS = frozenset(MODE_SCHEMA_VERSIONS.values())

#: Exact required output-file suffixes per mode for the ONE canonical
#: all-ok scenario Phase 2 actually generates (see
#: ``generate_phase2_goldens.GOLDEN_TOLERANCE_KEYS`` for the matching
#: scenario list). ``""`` means a single primary file named
#: ``<mode>.csv``; multiple entries are primary + secondary-fan-out (or,
#: for canopy_cover, primary + secondary-scientific-aggregate +
#: diagnostic-group-status). Duplicated here rather than imported from
#: ``test_cpp_harness_contract.MODES`` to avoid a circular import
#: (that module imports ``_harness_support``, which imports this module).
MODE_OUTPUT_SUFFIXES = {
    "consume": ("_summary", "_components"),
    "litter_eq": ("",),
    "shrub_herb_eq": ("",),
    "mortality": ("",),
    "bark_thick": ("",),
    "canopy_cover": ("_trees", "_stands", "_groups"),
    "soil_campbell": ("_summary", "_field"),
}

#: Modes whose golden requires exactly one species-table side file.
MODES_REQUIRING_SPECIES_TABLE = frozenset({"mortality", "bark_thick", "canopy_cover"})
#: Modes whose golden requires exactly one emission-factor-table side file.
MODES_REQUIRING_EMISSION_FACTOR_TABLE = frozenset({"consume"})

#: The exact, single tracked species table every species-driven mode's
#: golden must reference — not merely "a" species CSV.
EXPECTED_SPECIES_TABLE_REPO_PATH = "src/pyfofem/supporting_data/FOFEM6.7/FOF_SPP.CSV"
#: The exact, single approved emission-factor table `consume` must reference.
EXPECTED_EMISSION_FACTOR_TABLE_REPO_PATH = "reference/fofem_cpp/FOF_UNIX/Emission_Factors.csv"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_FIELDS = [
    "upstream_cpp_sha",
    "overlay_combined_digest",
    "overlay_file_digests",
    "harness_mode",
    "schema_version",
    "compiler_identity",
    "generator_toolchain",
    "platform",
    "architecture",
    "build_type",
    "build_flags",
    "generating_command",
    "input_csv_sha256",
    "output_csv_sha256",
    "side_file_sha256",
    "generated_utc",
    "pyfofem_commit",
    "pyfofem_dirty",
    "generator_source_sha256",
    "documented_expected_divergences",
    "tolerance_policy_reference",
]


class ProvenanceError(RuntimeError):
    """Raised when a fail-closed provenance precondition is violated."""


def _dataset_of(manifest: Dict[str, Any]) -> str:
    """
    Return the dataset a manifest belongs to.

    :param manifest: A manifest dict.
    :returns: The recorded ``dataset`` field, or :data:`DEFAULT_DATASET` when
        the field is absent (the frozen Phase 2 manifests predate it).
    """
    return str(manifest.get("dataset", DEFAULT_DATASET))


def _is_safe_repo_relative_path(rel_path: str) -> bool:
    """
    Return ``True`` iff *rel_path* is a plain, forward-slash repo-relative
    path that resolves inside :data:`~tests._support.PROJECT_ROOT` with no
    ``..`` traversal — rejects e.g. ``"../../etc/passwd"`` or an absolute
    path smuggled in as a "repo-relative" manifest field.
    """
    if not rel_path or os.path.isabs(rel_path):
        return False
    if "\\" in rel_path:
        return False
    normalized = os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))
    root = os.path.normpath(PROJECT_ROOT)
    return normalized == root or normalized.startswith(root + os.sep)


def _run_git(args: List[str]) -> str:
    """Run a git command via the bounded/tree-killing subprocess helper and
    return its stdout, raising like ``subprocess.check_output`` would on a
    nonzero exit (``run_bounded`` itself does not raise on nonzero exit)."""
    result = run_bounded(["git"] + args, timeout=TIMEOUT_GIT_S)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr}"
        )
    return result.stdout


def build_manifest(
        *,
        harness_mode: str,
        schema_version: str,
        compiler_identity: str,
        generator_toolchain: str,
        platform: str,
        architecture: str,
        build_type: str,
        build_flags: str,
        generating_command: str,
        input_csv_paths: List[str],
        output_csv_paths: List[str],
        tolerance_policy_keys: List[str],
        side_files: Optional[Dict[str, str]] = None,
        now_utc_iso: Optional[str] = None,
        dataset: str = DEFAULT_DATASET,
) -> Dict[str, Any]:
    """
    Build a complete provenance manifest for one golden-generation run.

    Calls :func:`check_pinned_sha` itself (fail-closed) before doing
    anything else.

    :param harness_mode: One of the six Phase 2 modes.
    :param schema_version: Input/output schema version string for
        *harness_mode*, the mode-specific value from
        :data:`MODE_SCHEMA_VERSIONS` (e.g. ``"1"`` for most modes,
        ``"2"`` for ``mortality`` since the Phase 4 correction pass) —
        never a single hardcoded value shared by every mode.
    :param compiler_identity: Real cl.exe version banner — never generic
        prose.
    :param generator_toolchain: Real CMake + Ninja version strings.
    :param platform: e.g. ``platform.platform()``.
    :param architecture: e.g. ``platform.machine()``.
    :param build_type: CMake build type used (``"Debug"``).
    :param build_flags: The EFFECTIVE compiler flags actually used, read
        from the real CMake cache — never guessed/hardcoded prose.
    :param generating_command: The exact command that produced this
        dataset (argv, joined).
    :param input_csv_paths: Absolute paths to every input CSV consumed.
    :param output_csv_paths: Absolute paths to every output CSV produced.
    :param tolerance_policy_keys: Dotted ``mode.scenario`` keys (from
        ``tolerance_policy.json``) applicable to this golden. Must be
        non-empty; ``documented_expected_divergences`` is derived from
        these, not supplied separately, so a golden can never carry an
        empty divergence list while claiming applicable, unresolved
        findings exist.
    :param side_files: Optional ``{role: absolute_path}`` map for side
        files whose identity matters (species table, emission-factor
        table) — hashed and recorded under ``side_file_sha256`` with both
        a repo-relative path and a digest.
    :param now_utc_iso: UTC timestamp to stamp (ISO-8601). Callers must
        supply this (this module never calls ``datetime.utcnow()`` itself
        so manifests stay reproducible/testable without wall-clock
        dependence); production callers pass a real timestamp.
    :param dataset: Which golden dataset this manifest belongs to — one of
        :data:`VALID_DATASETS`. ``"phase2"`` (the default) reproduces the
        original, frozen behaviour exactly and writes NO ``dataset`` field,
        so the committed Phase 2 manifests stay byte-identical; any other
        dataset records the field explicitly.
    :return: A manifest dict with every field in :data:`REQUIRED_FIELDS`
        populated.
    :raises ProvenanceError: Via :func:`check_pinned_sha`.
    :raises ValueError: If *tolerance_policy_keys* is empty or *dataset* is
        not a known dataset.
    """
    check_pinned_sha()
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"dataset {dataset!r} is not one of {sorted(VALID_DATASETS)}"
        )
    if not tolerance_policy_keys:
        raise ValueError(
            "tolerance_policy_keys must not be empty — every golden must "
            "cite the specific tolerance-policy keys that apply to it, not "
            "the whole harness-contract document"
        )

    policy = load_tolerance_policy()
    expected_policy_keys = canonical_policy_keys(dataset, harness_mode, policy)
    if tolerance_policy_keys != expected_policy_keys:
        raise ValueError(
            f"tolerance_policy_keys for {harness_mode!r} must exactly match "
            f"the canonical {DATASET_LABELS[dataset]} scenario contract; expected "
            f"{expected_policy_keys!r}, got {tolerance_policy_keys!r}"
        )
    divergences = divergences_for_keys(
        policy, canonical_divergence_keys(dataset, harness_mode)
    )

    overlay = compute_overlay_digests()
    side_files = side_files or {}
    dirty = git_dirty_status(PROJECT_ROOT)

    generator_source_sha256: Dict[str, str] = {}
    if dirty["dirty"]:
        generator_source_sha256 = sha256_files(
            generator_source_files_for_dataset(dataset)
        )

    manifest = {
        "upstream_cpp_sha": current_upstream_sha(),
        "overlay_combined_digest": overlay["combined"],
        "overlay_file_digests": overlay["per_file"],
        "harness_mode": harness_mode,
        "schema_version": schema_version,
        "compiler_identity": compiler_identity,
        "generator_toolchain": generator_toolchain,
        "platform": platform,
        "architecture": architecture,
        "build_type": build_type,
        "build_flags": build_flags,
        "generating_command": generating_command,
        "input_csv_sha256": sha256_files_by_basename(input_csv_paths),
        "output_csv_sha256": sha256_files_by_basename(output_csv_paths),
        "side_file_sha256": {
            role: {"path": to_repo_relative(path), "sha256": sha256_file(path)}
            for role, path in side_files.items()
        },
        "generated_utc": now_utc_iso,
        "pyfofem_commit": current_pyfofem_commit_safe(),
        "pyfofem_dirty": dirty,
        "generator_source_sha256": generator_source_sha256,
        "documented_expected_divergences": divergences,
        "tolerance_policy_reference": list(expected_policy_keys),
    }
    if dataset != DEFAULT_DATASET:
        manifest["dataset"] = dataset
    return manifest


def canonical_divergence_keys(dataset: str, mode: str) -> List[str]:
    """
    Return the dotted policy keys whose divergence status *dataset*'s
    manifest for *mode* must document.

    :param dataset: ``"phase2"`` or ``"phase4"``.
    :param mode: Harness mode name.
    :returns: Dotted ``<policy-section>.<route>`` keys, deterministically
        ordered.
    :raises KeyError: If *dataset* or *mode* has no contract.
    """
    if dataset == "phase2":
        return phase2_canonical_divergence_keys(mode)
    if dataset == "phase4":
        # Imported lazily: _phase4_contract imports the harness-contract
        # module, which imports _harness_support, which imports THIS module.
        from tests.cpp_parity_live._phase4_contract import phase4_divergence_keys
        return phase4_divergence_keys(mode)
    if dataset == "phase5":
        # Imported lazily for the same reason as phase4 above.
        from tests.cpp_parity_live._phase5_contract import phase5_divergence_keys
        return phase5_divergence_keys(mode)
    raise KeyError(f"unknown golden dataset: {dataset!r}")


def canonical_policy_keys(dataset: str, mode: str, policy: dict) -> List[str]:
    """
    Return every tolerance-policy key applicable to *dataset*'s golden for
    *mode*.

    :param dataset: ``"phase2"`` or ``"phase4"``.
    :param mode: Harness mode name.
    :param policy: Loaded tolerance-policy object.
    :returns: Dotted ``<policy-section>.<route>`` keys, deterministically
        ordered.
    :raises KeyError: If *dataset*, *mode*, or a configured route is absent.
    """
    if dataset == "phase2":
        return phase2_canonical_policy_keys(mode, policy)
    if dataset == "phase4":
        from tests.cpp_parity_live._phase4_contract import phase4_policy_keys
        keys = phase4_policy_keys(mode)
        for key in keys:
            section, _, route = key.partition(".")
            policy[section][route]
        return keys
    if dataset == "phase5":
        from tests.cpp_parity_live._phase5_contract import phase5_policy_keys
        keys = phase5_policy_keys(mode)
        for key in keys:
            section, _, route = key.partition(".")
            policy[section][route]
        return keys
    raise KeyError(f"unknown golden dataset: {dataset!r}")


def check_pinned_sha() -> None:
    """
    Fail closed unless the live C++ checkout is exactly at
    :data:`PINNED_UPSTREAM_SHA`.

    Must be called before applying the overlay, configuring/building,
    running qualification, or writing any golden output — a manifest
    recording a different SHA after the fact is not a substitute.

    :return: None.
    :raises ProvenanceError: If the live checkout's HEAD does not exactly
        equal :data:`PINNED_UPSTREAM_SHA`.
    """
    live = current_upstream_sha()
    if live != PINNED_UPSTREAM_SHA:
        raise ProvenanceError(
            f"pinned-SHA check failed: reference/fofem_cpp HEAD is {live}, "
            f"expected exactly {PINNED_UPSTREAM_SHA}. Refusing to apply the "
            "overlay, configure/build, qualify, or write golden output "
            "against an unpinned checkout."
        )


def compute_overlay_digests() -> Dict[str, Any]:
    """
    Hash every file under the maintained overlay directory
    (``reference/fofem_cpp_overlay/``) — README, patch, AND source, not
    only ``source/`` (Codex finding: the prior version only hashed
    ``source/``, silently excluding README.md and the patch from
    provenance).

    :return: ``{"per_file": {overlay_relpath: sha256}, "combined": sha256}``
        where ``combined`` is the SHA-256 of the sorted
        ``"overlay_relpath:sha256\\n"`` lines (order-independent of
        filesystem walk order). Keys are relative to ``OVERLAY_ROOT`` with
        forward slashes.
    """
    per_file: Dict[str, str] = {}
    for root, _, files in os.walk(OVERLAY_ROOT):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, OVERLAY_ROOT).replace(os.sep, "/")
            per_file[rel] = sha256_file(full)
    lines = sorted(f"{rel}:{h}" for rel, h in per_file.items())
    combined = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {"per_file": per_file, "combined": combined}


def current_pyfofem_commit() -> str:
    """Return the parent repo's current HEAD SHA (no dirty suffix — dirty
    status is recorded separately and in full via :func:`git_dirty_status`,
    since a suffix string cannot be validated/compared reliably)."""
    return _run_git(["-C", PROJECT_ROOT, "rev-parse", "HEAD"]).strip()


def current_pyfofem_commit_safe() -> str:
    """Wrapper kept separate so a git-less environment degrades to a clear
    sentinel instead of raising during manifest construction."""
    try:
        return current_pyfofem_commit()
    except Exception as exc:  # pragma: no cover - environment without git
        return f"<unavailable: {exc}>"


def current_upstream_sha() -> str:
    """Return the pinned C++ submodule checkout's current HEAD SHA."""
    return _run_git(["-C", CPP_REFERENCE_DIR, "rev-parse", "HEAD"]).strip()


#: Statuses that represent "no known problem" (either confirmed equivalent
#: or simply not yet evaluated) rather than an actual documented defect —
#: excluded from :func:`divergences_for_keys`'s output. Every OTHER status
#: (``known_divergent*``, ``python_side_unreachable``, etc.) is treated as
#: a real, worth-surfacing divergence/limitation.
_NON_DIVERGENT_STATUSES = frozenset({"verified", "unverified", "verified_equivalent"})


def divergences_for_keys(policy: Dict[str, Any], keys: List[str]) -> List[str]:
    """
    Build a human-readable divergence list from tolerance-policy *keys*
    (``"<mode>.<scenario>"``), for use as a manifest's
    ``documented_expected_divergences``.

    Only keys whose policy status represents an ACTUAL known
    defect/limitation are included — a key with status ``"verified"``,
    ``"unverified"``, or ``"verified_equivalent"`` is not a divergence (it
    is either confirmed matching or simply not yet evaluated) and is
    omitted. This means the returned list may legitimately be empty (e.g.
    every cited key is "verified") — that is not an error; the FULL set of
    cited policy keys (divergent or not) is separately recorded verbatim
    in the manifest's ``tolerance_policy_reference`` field.

    :param policy: The loaded tolerance policy (see
        :func:`load_tolerance_policy`).
    :param keys: Dotted ``mode.scenario`` keys into *policy*.
    :return: One string per DIVERGENT key: ``"<key>: <status> — "
        "<justification> (<traceability>)"``.
    :raises KeyError: If a key does not resolve in *policy*.
    """
    out = []
    for key in keys:
        mode, _, scenario = key.partition(".")
        entry = policy[mode][scenario]
        status = entry.get("status", "no_status_recorded")
        if status in _NON_DIVERGENT_STATUSES:
            continue
        out.append(
            f"{key}: {status} — {entry['justification']} "
            f"({entry['traceability']})"
        )
    return out


def generator_source_files_for_dataset(dataset: str) -> List[str]:
    """
    Return the absolute paths whose bytes determine *dataset*'s goldens.

    :param dataset: ``"phase2"``, ``"phase4"``, or ``"phase5"``.
    :returns: Absolute paths, in the dataset's declared order.
    :raises KeyError: If *dataset* is unknown.
    """
    if dataset == "phase2":
        return list(GENERATOR_SOURCE_FILES)
    if dataset == "phase4":
        from tests.cpp_parity_live._phase4_contract import (
            GENERATOR_SOURCE_FILES_RELATIVE,
        )
        return [
            os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
            for rel in GENERATOR_SOURCE_FILES_RELATIVE
        ]
    if dataset == "phase5":
        from tests.cpp_parity_live._phase5_contract import (
            GENERATOR_SOURCE_FILES_RELATIVE as _P5_FILES,
        )
        return [
            os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
            for rel in _P5_FILES
        ]
    raise KeyError(f"unknown golden dataset: {dataset!r}")


def git_dirty_status(repo_dir: str) -> Dict[str, Any]:
    """
    Detect staged, unstaged, AND untracked changes in *repo_dir*.

    ``git diff --quiet`` alone only detects unstaged changes to tracked
    files — it misses staged-but-uncommitted changes and untracked new
    files entirely (Codex finding). ``git status --porcelain`` covers all
    three.

    :param repo_dir: Path to a git working tree.
    :return: ``{"dirty": bool, "staged": bool, "unstaged": bool,
        "untracked": bool, "porcelain": str}``.
    """
    out = _run_git(["-C", repo_dir, "status", "--porcelain"])
    staged = False
    unstaged = False
    untracked = False
    for line in out.splitlines():
        if not line:
            continue
        index_state, worktree_state = line[0], line[1]
        if line.startswith("??"):
            untracked = True
            continue
        if index_state not in (" ",):
            staged = True
        if worktree_state not in (" ",):
            unstaged = True
    return {
        "dirty": bool(out.strip()),
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "porcelain": out,
    }


def load_tolerance_policy() -> Dict[str, Any]:
    """Load the centralized tolerance-policy JSON (see
    ``tests/cpp_parity_live/tolerance_policy.json``)."""
    with open(TOLERANCE_POLICY_PATH, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of *path*'s bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: List[str]) -> Dict[str, str]:
    """Return ``{repo_relative_path: sha256_hex}`` for each of *paths*."""
    return {to_repo_relative(p): sha256_file(p) for p in paths}


def sha256_files_by_basename(paths: List[str]) -> Dict[str, str]:
    """
    Return ``{basename: sha256_hex}`` for each of *paths*.

    Used for a golden's own input/output CSVs, which are always co-located
    with their manifest in the SAME directory — a basename is therefore a
    stable, portable identity, whereas :func:`to_repo_relative` would
    record a path that reaches outside the repo entirely whenever
    generation happens in a temporary directory before promotion (the
    normal case here; see ``generate_phase2_goldens.py``).

    :raises ValueError: If two of *paths* share a basename (the identity
        would be ambiguous).
    """
    out: Dict[str, str] = {}
    for p in paths:
        base = os.path.basename(p)
        if base in out:
            raise ValueError(f"duplicate basename among golden files: {base!r}")
        out[base] = sha256_file(p)
    return out


def to_repo_relative(path: str) -> str:
    """Return *path* relative to :data:`PROJECT_ROOT`, forward-slashed, so
    manifest keys are portable across machines/OSes."""
    rel = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    return rel.replace(os.sep, "/")


def validate_manifest(
        manifest: Any, *, check_against_live_checkout: bool = True,
        golden_dir: Optional[str] = None,
) -> List[str]:
    """
    Validate a manifest's structure, internal consistency, exact-value
    claims, and (optionally) every file digest it references against the
    files actually on disk right now.

    This is intentionally strict enough to catch a manifest whose output
    digests were replaced with zeros (a real gap Codex demonstrated in the
    prior version of this validator): every ``input_csv_sha256`` /
    ``output_csv_sha256`` / ``side_file_sha256`` entry is re-hashed from
    the real file on disk and compared, not merely checked for shape.

    :param manifest: A manifest dict (as produced by :func:`build_manifest`
        or loaded from JSON).
    :param check_against_live_checkout: If ``True``, also verify the
        recorded ``upstream_cpp_sha``/``overlay_combined_digest``/
        ``overlay_file_digests`` and every referenced input/output/side
        file against the live checkout and disk.
    :param golden_dir: Directory the manifest's OWN input/output CSVs
        (keyed by basename — see :func:`sha256_files_by_basename`) live
        in. Required for the input/output on-disk re-hash check when
        *check_against_live_checkout* is ``True`` — if omitted in that
        case, this now FAILS CLOSED with an explicit
        ``"golden_dir is required"`` error rather than silently skipping
        the input/output authentication (side_file_sha256, which carries
        its own repo-relative paths, is still checked against
        :data:`~tests._support.PROJECT_ROOT` regardless of *golden_dir*).

    The dataset a manifest belongs to is read from its own ``dataset``
    field (absent means ``"phase2"``), and every dataset-scoped contract —
    the exact tolerance-policy key set, the derived expected-divergence
    list, and the exact ``generator_source_sha256`` key set — is resolved
    against THAT dataset. A manifest naming an unknown dataset is rejected
    outright rather than validated against the wrong contract.

    :return: List of human-readable error strings; empty means valid.
    """
    errors: List[str] = []

    if not isinstance(manifest, dict):
        return [f"manifest is not a JSON object: {type(manifest)!r}"]

    for field in REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"missing required field: {field}")
    if errors:
        # Structural errors make every check below meaningless.
        return errors

    # --- type / format / allowed-value checks ---
    dataset = _dataset_of(manifest)
    if dataset not in VALID_DATASETS:
        # Every downstream contract (policy keys, divergences, generator
        # sources) is dataset-scoped, so an unknown dataset makes the rest
        # meaningless — fail here rather than validate against the wrong
        # contract.
        return [
            f"dataset {dataset!r} is not one of {sorted(VALID_DATASETS)}"
        ]
    if manifest["harness_mode"] not in VALID_HARNESS_MODES:
        errors.append(
            f"harness_mode {manifest['harness_mode']!r} not one of "
            f"{sorted(VALID_HARNESS_MODES)}"
        )
    if manifest["schema_version"] not in VALID_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version {manifest['schema_version']!r} not one of "
            f"{sorted(VALID_SCHEMA_VERSIONS)}"
        )
    elif manifest["harness_mode"] in MODE_SCHEMA_VERSIONS:
        # The binding check: a version string that is valid for SOME
        # mode is still wrong provenance if it is not this mode's own
        # declared version.
        expected_version = MODE_SCHEMA_VERSIONS[manifest["harness_mode"]]
        if manifest["schema_version"] != expected_version:
            errors.append(
                f"schema_version is {manifest['schema_version']!r}, but "
                f"mode {manifest['harness_mode']!r} declares "
                f"{expected_version!r}"
            )
    if not _GIT_SHA_RE.match(str(manifest["upstream_cpp_sha"])):
        errors.append(
            f"upstream_cpp_sha {manifest['upstream_cpp_sha']!r} is not a "
            "40-hex-character git SHA"
        )
    elif manifest["upstream_cpp_sha"] != PINNED_UPSTREAM_SHA:
        errors.append(
            f"upstream_cpp_sha is {manifest['upstream_cpp_sha']}, but the "
            f"only approved pinned SHA is {PINNED_UPSTREAM_SHA} — a "
            "manifest recording a different SHA is not sufficient "
            "provenance"
        )
    if not _SHA256_RE.match(str(manifest["overlay_combined_digest"])):
        errors.append("overlay_combined_digest is not a 64-hex-char SHA-256")

    if not isinstance(manifest["overlay_file_digests"], dict) or not manifest["overlay_file_digests"]:
        errors.append("overlay_file_digests must be a non-empty object")
    else:
        recorded_set = set(manifest["overlay_file_digests"].keys())
        missing = REQUIRED_OVERLAY_FILES - recorded_set
        extra = recorded_set - REQUIRED_OVERLAY_FILES
        if missing:
            errors.append(f"overlay_file_digests missing required files: {sorted(missing)}")
        if extra:
            errors.append(f"overlay_file_digests has unexpected extra files: {sorted(extra)}")
        for rel, digest in manifest["overlay_file_digests"].items():
            if not _SHA256_RE.match(str(digest)):
                errors.append(f"overlay_file_digests[{rel!r}] is not a 64-hex-char SHA-256")
        # Internal consistency: the recorded combined digest must equal
        # the recomputation from the recorded per-file digests, regardless
        # of the live checkout.
        lines = sorted(f"{rel}:{h}" for rel, h in manifest["overlay_file_digests"].items())
        recomputed_combined = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        if recomputed_combined != manifest["overlay_combined_digest"]:
            errors.append(
                "overlay_combined_digest does not match the recomputation "
                "from the manifest's own overlay_file_digests (internally "
                "inconsistent manifest)"
            )

    for field in ("input_csv_sha256", "output_csv_sha256"):
        val = manifest[field]
        if not isinstance(val, dict) or not val:
            errors.append(f"{field} must be a non-empty object")
        else:
            for rel, digest in val.items():
                if not _SHA256_RE.match(str(digest)):
                    errors.append(f"{field}[{rel!r}] is not a 64-hex-char SHA-256")

    # --- exact per-mode input/output file contract (Codex finding: a
    # missing, extra, or renamed output must be rejected, not merely a
    # digest mismatch on the files that happen to be present) ---
    mode = manifest["harness_mode"]
    if (
            mode in MODE_OUTPUT_SUFFIXES
            and isinstance(manifest["input_csv_sha256"], dict)
            and isinstance(manifest["output_csv_sha256"], dict)
    ):
        expected_outputs = {f"{mode}{suffix}.csv" for suffix in MODE_OUTPUT_SUFFIXES[mode]}
        actual_outputs = set(manifest["output_csv_sha256"].keys())
        missing_out = expected_outputs - actual_outputs
        extra_out = actual_outputs - expected_outputs
        if missing_out:
            errors.append(f"output_csv_sha256 missing required files for mode {mode!r}: {sorted(missing_out)}")
        if extra_out:
            errors.append(f"output_csv_sha256 has unexpected extra files for mode {mode!r}: {sorted(extra_out)}")

        expected_inputs = {f"{mode}_in.csv"}
        actual_inputs = set(manifest["input_csv_sha256"].keys())
        if actual_inputs != expected_inputs:
            errors.append(
                f"input_csv_sha256 for mode {mode!r} must be exactly "
                f"{sorted(expected_inputs)}, got {sorted(actual_inputs)}"
            )

    # --- exact side-file contract per mode ---
    if not isinstance(manifest["side_file_sha256"], dict):
        errors.append("side_file_sha256 must be an object")
    else:
        side = manifest["side_file_sha256"]
        expected_roles = set()
        if mode in MODES_REQUIRING_SPECIES_TABLE:
            expected_roles.add("species_table")
        if mode in MODES_REQUIRING_EMISSION_FACTOR_TABLE:
            expected_roles.add("emission_factor_table")
        actual_roles = set(side.keys())
        missing_roles = expected_roles - actual_roles
        extra_roles = actual_roles - expected_roles
        if missing_roles:
            errors.append(f"side_file_sha256 missing required role(s) for mode {mode!r}: {sorted(missing_roles)}")
        if extra_roles:
            errors.append(
                f"side_file_sha256 has unexpected role(s) for mode {mode!r} "
                f"(modes without side files must not carry any; a species-only "
                f"mode must not also carry an emission-factor entry, etc.): {sorted(extra_roles)}"
            )

        for role, rec in side.items():
            if not isinstance(rec, dict) or "path" not in rec or "sha256" not in rec:
                errors.append(
                    f"side_file_sha256[{role!r}] must be an object with "
                    "'path' and 'sha256'"
                )
                continue
            if not _SHA256_RE.match(str(rec["sha256"])):
                errors.append(f"side_file_sha256[{role!r}]['sha256'] is not a valid SHA-256")
            if not _is_safe_repo_relative_path(str(rec["path"])):
                errors.append(
                    f"side_file_sha256[{role!r}]['path'] {rec['path']!r} is not a "
                    "safe repo-relative path (absolute, backslash, or traversal "
                    "outside the repository)"
                )
                continue
            if role == "species_table" and rec["path"] != EXPECTED_SPECIES_TABLE_REPO_PATH:
                errors.append(
                    f"side_file_sha256['species_table']['path'] must be exactly "
                    f"{EXPECTED_SPECIES_TABLE_REPO_PATH!r}, got {rec['path']!r}"
                )
            if role == "emission_factor_table" and rec["path"] != EXPECTED_EMISSION_FACTOR_TABLE_REPO_PATH:
                errors.append(
                    f"side_file_sha256['emission_factor_table']['path'] must be "
                    f"exactly {EXPECTED_EMISSION_FACTOR_TABLE_REPO_PATH!r}, got {rec['path']!r}"
                )

    policy = None
    if not isinstance(manifest["tolerance_policy_reference"], list) or not manifest["tolerance_policy_reference"]:
        errors.append("tolerance_policy_reference must be a non-empty list of policy keys")
    else:
        try:
            policy = load_tolerance_policy()
        except Exception as exc:  # pragma: no cover
            errors.append(f"could not load tolerance_policy.json for cross-check: {exc}")
            policy = None
        if policy is not None:
            for key in manifest["tolerance_policy_reference"]:
                key_mode, _, scenario = str(key).partition(".")
                if key_mode not in policy or scenario not in policy.get(key_mode, {}):
                    errors.append(f"tolerance_policy_reference key {key!r} not found in tolerance_policy.json")
            if mode in VALID_HARNESS_MODES:
                expected_policy_keys = canonical_policy_keys(dataset, mode, policy)
                if manifest["tolerance_policy_reference"] != expected_policy_keys:
                    errors.append(
                        f"tolerance_policy_reference for mode {mode!r} must "
                        f"exactly match the canonical {DATASET_LABELS[dataset]} scenario "
                        f"contract: expected {expected_policy_keys!r}, got "
                        f"{manifest['tolerance_policy_reference']!r}"
                    )

    if not isinstance(manifest["documented_expected_divergences"], list):
        errors.append("documented_expected_divergences must be a list")
    else:
        for entry in manifest["documented_expected_divergences"]:
            if not isinstance(entry, str) or not entry:
                errors.append(
                    f"documented_expected_divergences entry {entry!r} must be "
                    "a non-empty string"
                )
        # Deliberately NOT required to be non-empty: a golden whose cited
        # tolerance_policy_reference keys are all "verified"/"unverified"/
        # "verified_equivalent" has no actual divergence to document — see
        # divergences_for_keys()'s _NON_DIVERGENT_STATUSES filter.
        if policy is not None and mode in VALID_HARNESS_MODES:
            expected_divergences = divergences_for_keys(
                policy, canonical_divergence_keys(dataset, mode)
            )
            if manifest["documented_expected_divergences"] != expected_divergences:
                errors.append(
                    "documented_expected_divergences must equal the "
                    "scenario-applicable divergences derived from "
                    f"tolerance_policy.json; expected {expected_divergences!r}, "
                    f"got {manifest['documented_expected_divergences']!r}"
                )

    if not isinstance(manifest.get("pyfofem_dirty"), dict) or "dirty" not in manifest.get("pyfofem_dirty", {}):
        errors.append("pyfofem_dirty must be an object with at least a 'dirty' key")
    elif manifest["pyfofem_dirty"].get("dirty") and not manifest.get("generator_source_sha256"):
        errors.append(
            "pyfofem_dirty.dirty is true but generator_source_sha256 is "
            "empty — a dirty-tree generation must hash its own generator "
            "source"
        )

    if errors:
        return errors

    if not check_against_live_checkout:
        return errors

    # --- live-checkout / on-disk cross-checks ---
    try:
        live_sha = current_upstream_sha()
    except Exception as exc:  # pragma: no cover
        errors.append(f"could not read live upstream SHA for cross-check: {exc}")
        live_sha = None
    if live_sha is not None and manifest["upstream_cpp_sha"] != live_sha:
        errors.append(
            f"upstream_cpp_sha mismatch: manifest says {manifest['upstream_cpp_sha']}, "
            f"live checkout is {live_sha}"
        )

    live_overlay = compute_overlay_digests()
    if manifest["overlay_combined_digest"] != live_overlay["combined"]:
        errors.append(
            "overlay_combined_digest mismatch: manifest says "
            f"{manifest['overlay_combined_digest']}, live overlay tree "
            f"hashes to {live_overlay['combined']}"
        )
    recorded_files = set(manifest["overlay_file_digests"].keys())
    live_files = set(live_overlay["per_file"].keys())
    if recorded_files != live_files:
        errors.append(
            f"overlay file set mismatch: manifest has {sorted(recorded_files)}, "
            f"live tree has {sorted(live_files)}"
        )
    for rel in recorded_files & live_files:
        if manifest["overlay_file_digests"][rel] != live_overlay["per_file"][rel]:
            errors.append(
                f"overlay file {rel!r} digest mismatch: manifest says "
                f"{manifest['overlay_file_digests'][rel]}, live file hashes to "
                f"{live_overlay['per_file'][rel]}"
            )

    def _check_referenced_file(path: str, recorded_digest: str, label: str) -> None:
        if not os.path.isfile(path):
            errors.append(f"{label} {path!r} referenced by manifest does not exist on disk")
            return
        actual = sha256_file(path)
        if actual != recorded_digest:
            errors.append(
                f"{label} {path!r} digest mismatch: manifest says "
                f"{recorded_digest}, actual file hashes to {actual}"
            )

    if golden_dir is None:
        errors.append(
            "golden_dir is required when check_against_live_checkout=True — "
            "omitting it previously caused input/output CSV authentication "
            "to be silently skipped. Pass the directory containing this "
            "golden's own input/output CSVs, or call with "
            "check_against_live_checkout=False if no on-disk directory "
            "exists to check against."
        )
    else:
        for base, digest in manifest["input_csv_sha256"].items():
            _check_referenced_file(os.path.join(golden_dir, base), digest, "input CSV")
        for base, digest in manifest["output_csv_sha256"].items():
            _check_referenced_file(os.path.join(golden_dir, base), digest, "output CSV")

    for role, rec in manifest["side_file_sha256"].items():
        _check_referenced_file(
            os.path.join(PROJECT_ROOT, rec["path"]), rec["sha256"], f"side file {role!r}"
        )

    # --- generator_source_sha256: exact key set + live rehash ---
    expected_gen_keys = {
        to_repo_relative(p)
        for p in generator_source_files_for_dataset(dataset)
    }
    gen_sha = manifest.get("generator_source_sha256") or {}
    if gen_sha:
        actual_gen_keys = set(gen_sha.keys())
        if actual_gen_keys != expected_gen_keys:
            errors.append(
                "generator_source_sha256 key set does not match "
                f"GENERATOR_SOURCE_FILES exactly: missing "
                f"{sorted(expected_gen_keys - actual_gen_keys)}, "
                f"extra {sorted(actual_gen_keys - expected_gen_keys)}"
            )
        for rel in actual_gen_keys & expected_gen_keys:
            _check_referenced_file(
                os.path.join(PROJECT_ROOT, rel), gen_sha[rel], f"generator source {rel!r}"
            )

    return errors


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    """Write *manifest* as pretty-printed, deterministically-ordered JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
