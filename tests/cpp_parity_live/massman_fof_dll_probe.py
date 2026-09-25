#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
massman_fof_dll_probe.py - Tracked, independently-reproducible, fail-closed
diagnostic build/link/run probe for the pinned
``reference/fofem_cpp/FOF_DLL/`` Massman HMV solver. Evidence source for
findings F-55/F-57/F-58 (``development/plans/gate0/04-findings.md``).

Run manually (NOT part of ``--suite core``/``--suite full``, and NOT
collected by plain ``pytest`` -- this module's filename deliberately does
not match pytest's ``test_*.py`` discovery pattern, and it does its own
compile/link/run rather than asserting anything):

    python -m tests.cpp_parity_live.massman_fof_dll_probe

Prints a single JSON object to stdout and exits 0 only when every
structural stage completed AND both runs produced IDENTICAL parsed
results and diagnostic counts (schema-valid; see item 7 below for exactly
what "identical" means here -- it is NaN-aware semantic equality of the
parsed fields, not a raw-stdout byte comparison), reaching
``stage == "complete"``; exits 1 with an
``"error"``/``"stage"`` pair describing exactly which check failed
otherwise. Scientific non-finiteness (NaN/Inf in the solver's own output)
is NOT a structural failure -- a run that completes cleanly but produces
non-finite output still reaches ``stage == "complete"``, with that fact
recorded explicitly via ``solver_output_finite``/the per-field
``*_any_nan``/``*_any_inf`` flags. Only genuine probe-mechanics failures
(bad build, unparseable output, non-reproducible runs, a dirty FOF_DLL
checkout, ...) prevent ``stage == "complete"``.

What this does, in order:

1. Verifies :data:`FOF_DLL_DIR` is git-clean (no tracked modification,
   staged change, deletion, or untracked file) via ``git -C
   reference/fofem_cpp status --porcelain -- FOF_DLL`` -- scoped
   specifically to ``FOF_DLL/`` so it does NOT reject the separately
   expected, pre-existing ``FOF_UNIX``/overlay dirtiness elsewhere in the
   submodule.
2. Fail-closed pinned-upstream-SHA check (:func:`tests.cpp_parity_live.
   _golden_manifest.check_pinned_sha`) -- refuses to build against any
   checkout other than the one pinned SHA.
3. Computes SHA-256 digests for the driver source, this probe script
   itself, every selected ``FOF_DLL/*.cpp`` file, the two directly-
   consumed headers (``BMSoil.h``/``HTAA.h``), and one aggregate digest
   covering every file under ``FOF_DLL/`` (not just the selected ones) --
   so a caller can verify exactly which bytes were compiled without
   re-running the probe.
4. Discovers the MSVC/cl.exe toolchain the same way the live C++ parity
   harness does (:func:`tests.cpp_parity_live._harness_support._msvc_env`),
   and records cl.exe's own version banner and the Visual Studio
   installation path used -- never a generic "some compiler" claim.
5. Compiles+links :data:`DRIVER_SOURCE` against :data:`FOF_DLL_SOURCES`
   into a uniquely-named scratch directory rooted under THIS repository
   (:func:`_scratch_root`, verified to be a child of ``PROJECT_ROOT``
   before anything is created or deleted) -- never
   ``reference/fofem_cpp/`` or ``reference/fofem_cpp_overlay/``, and never
   the system/user temp directory. Fails closed if either compile/link
   returns nonzero OR either compiler stream (stdout or stderr) contains
   an ``LNK4006``/``LNK4221`` duplicate-symbol warning.
6. Runs the resulting binary TWICE, each with a bounded timeout, full
   descendant-process-tree cleanup
   (:func:`tests.cpp_parity_live._proc.run_bounded`), and stdin explicitly
   closed (``stdin=subprocess.DEVNULL``) -- the pinned source's own
   ``Mylongjmp()`` diagnostic path calls ``getchar()`` on certain
   numeric-anomaly detections (see ``FOF_DLL/BM_Util.cpp``), so an
   inherited stdin could otherwise hang forever.
7. Strictly parses and validates each run's single machine-readable
   ``PROBE_RESULT|...`` line against an exact required-field schema (see
   ``massman_fof_dll_probe_driver.cpp``'s own docstring for the field
   list) -- absent, duplicated, missing/duplicated/malformed/out-of-domain
   fields, a nonzero process return code, an unparseable/undecodable
   stream, or a nonpositive ``hta_count``/a set ``hta_get_failed`` flag
   all fail closed. The two runs' PARSED RESULTS (compared with NaN-aware
   equality via :func:`_values_equal` -- two genuinely-NaN values in the
   same field count as equal, since IEEE 754 NaN is never equal to itself
   under plain ``==``) and their divide-by-zero diagnostic counts must
   then be identical, or this also fails closed. This is semantic
   parsed-result equality, NOT a byte-for-byte comparison of the raw
   captured stdout (which may legitimately differ in incidental ways
   this probe does not care about, e.g. pointer-adjacent debug output the
   driver does not itself emit).
8. Deletes the scratch build directory (its own context-manager cleanup)
   and reports everything as one JSON object.

Function order: private helpers first, then public functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

from tests._support import CPP_REFERENCE_DIR, PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import (
    PINNED_UPSTREAM_SHA,
    ProvenanceError,
    check_pinned_sha,
    sha256_file,
)
from tests.cpp_parity_live._harness_support import _msvc_env, _vs_installation_path
from tests.cpp_parity_live._proc import ProcTimeout, run_bounded

FOF_DLL_DIR = os.path.join(CPP_REFERENCE_DIR, "FOF_DLL")
DRIVER_SOURCE = os.path.join(
    PROJECT_ROOT, "tests", "cpp_parity_live", "massman_fof_dll_probe_driver.cpp"
)
PROBE_SCRIPT = os.path.abspath(__file__)

#: Two headers the driver directly #includes -- hashed separately from the
#: aggregate FOF_DLL digest so a caller can see exactly what the driver's
#: compiled interface was, without needing to recompute the aggregate.
CONSUMED_HEADERS = ("BMSoil.h", "HTAA.h")

#: Wall-clock bound (seconds) for the compile+link step. A cold rebuild of
#: 80 small translation units on this toolchain has been observed to take
#: well under 60s; generously doubled.
TIMEOUT_COMPILE_S = 180
#: Wall-clock bound (seconds) for running the compiled probe binary.
#: FOF_DLL/BM_Util.cpp's Mylongjmp() diagnostic path can call getchar();
#: this probe always closes stdin (see run_probe()) so that path can only
#: ever return immediately (EOF), never actually block -- this timeout is
#: strictly a defense-in-depth backstop, not the expected exit path.
TIMEOUT_RUN_S = 60
#: Wall-clock bound (seconds) for the FOF_DLL git-cleanliness check.
TIMEOUT_GIT_S = 30

#: The exact, ordered set of pinned FOF_DLL/*.cpp files this probe links
#: against. Resolved empirically from REAL compiler/linker output (never
#: guessed): starting from HMV_Model's directly-called functions and
#: iterating on real LNK2019/LNK2001 "unresolved external symbol" errors
#: until zero remained. 80 files, zero duplicate-symbol (LNK4006/LNK4221)
#: warnings. Three genuine, source-verified surprises this closure search
#: surfaced (kept here, not silently normalized away):
#:   - Quincy1G.cpp is linked ONLY for the global-variable storage it
#:     defines (den0/thetai/tempai/parden/psini/... plus ~60 auxiliary
#:     hydraulic-model constants) -- its own Quincy1G() function body is
#:     never invoked by the real call path (Soil_Model_Data_Files_HMV.cpp
#:     calls WesternUS01(bmi), with the Quincy1G(bmi) call commented out).
#:   - BoundaryU.cpp/BoundaryUBFD.cpp/BoundaryLHB17dBFD.cpp are three
#:     mutually-exclusive ALTERNATE upper-boundary-condition
#:     implementations, but each owns storage for a distinct subset of the
#:     bcQ/bcva/bcta/bcra/forIR/eta4/expp/ttme/force/js arrays that the one
#:     boundary function actually called (BoundarydBFD, from
#:     HMV_Model.cpp:69) only extern-declares -- so all three sibling
#:     files must be linked purely for storage, even though only one
#:     boundary FUNCTION ever executes.
#:   - global.cpp supplies the small set of true physical-constant globals
#:     (pres0/temp0/grav/rgas/stefbol/h2omol/psi0/temp00/h2opsir/denair0/
#:     diff0/diffH0) that Physical_Constants_HMV.cpp only assigns to via
#:     extern, never defines itself.
FOF_DLL_SOURCES: tuple = (
    "HMV_Model.cpp",
    "Model_Switch_HMV.cpp",
    "Physical_Constants_HMV.cpp",
    "Soil_Time_Depth_Param_HMV.cpp",
    "Soil_Model_Data_Files_HMV.cpp",
    "WesternUS01.cpp",
    "BoundarydBFD.cpp",
    "SolveHMV.cpp",
    "CrankNicolson.cpp",
    "Model_Param.cpp",
    "HTAA.cpp",
    "LoadQuincy1.cpp",
    "BM_Util.cpp",
    "PSINi.cpp",
    "Matrix.cpp",
    "GenThomas.cpp",
    "BoundaryIR_T1.cpp",
    "BoundaryIR.cpp",
    "BoundaryU.cpp",
    "BoundaryUBFD.cpp",
    "BoundaryLHB17dBFD.cpp",
    "Water_Vapor_Constants.cpp",
    "Water_Vapor_Constants_0.cpp",
    "Water_Vapor_Constants_2.cpp",
    "Liquid_Water_Constants.cpp",
    "Liquid_Water_Constants_2.cpp",
    "Dry_Air_Constants_HMV.cpp",
    "PsinTmpnT.cpp",
    "AccumulateHMV.cpp",
    "caleta4.cpp",
    "calAwa.cpp",
    "calAwaP.cpp",
    "calConCoef2.cpp",
    "calConCoef5.cpp",
    "calEBCN.cpp",
    "calQHCN.cpp",
    "calSw.cpp",
    "calVsourceGNRa.cpp",
    "calconHMVll.cpp",
    "calconRAD.cpp",
    "calcondry.cpp",
    "calcpaHMVNR.cpp",
    "calcsHMV.cpp",
    "calcsHMVnT.cpp",
    "calden2HMV.cpp",
    "caldiffHMVNRa.cpp",
    "caldiseq.cpp",
    "caldryvis.cpp",
    "calepssurfHMV.cpp",
    "calgascomb.cpp",
    "calhydrauKF.cpp",
    "calhydrauVA.cpp",
    "calmulaHMV.cpp",
    "calmulaWHMV.cpp",
    "calprofP.cpp",
    "calpsinProf.cpp",
    "calrhev.cpp",
    "calstefan1NR.cpp",
    "calsurfdT.cpp",
    "caltempkHMV.cpp",
    "calthetaCSr.cpp",
    "calthetaFYr.cpp",
    "caluHMV.cpp",
    "calvaporHMV.cpp",
    "calxhiv1.cpp",
    "densatHMV.cpp",
    "harmean.cpp",
    "rhoveqHMV.cpp",
    "vapdiffEHMV.cpp",
    "vaporTempdiff.cpp",
    "EBcoef.cpp",
    "global.cpp",
    "Quincy1G.cpp",
    "calmaxdenwHMV.cpp",
    "calmaxmulaHMV.cpp",
    "calmaxmulaWHMV.cpp",
    "calcp0HMV.cpp",
    "Vmult5.cpp",
    "calparx.cpp",
    "densatmaxHMV.cpp",
)

#: Per-field metric suffixes the driver reports for each of the four
#: measured quantities (heat, moist, psin, time). Kept as a single source
#: of truth for both the required-field schema and result comparison.
_FIELD_GROUPS: Tuple[str, ...] = ("heat", "moist", "psin", "time")
_FIELD_METRICS: Tuple[str, ...] = (
    "total", "finite", "any_nan", "any_inf", "first", "last",
)
_SCALAR_INT_FIELDS: Tuple[str, ...] = ("hmv_rc", "hta_count", "hta_layers")
_SCALAR_BOOL_FIELDS: Tuple[str, ...] = ("hta_get_failed",)

_DIVIDE_BY_ZERO_RE = re.compile(r"Divide by Zero:\s*:\s*(\S+)")
_PROBE_RESULT_RE = re.compile(r"^PROBE_RESULT\|(.+)$", re.MULTILINE)


def _aggregate_fof_dll_digest() -> str:
    """
    Return one deterministic SHA-256 digest covering every file under
    :data:`FOF_DLL_DIR` (not only the compiled-in subset), so a caller can
    detect ANY change anywhere in the directory, not just to files this
    probe happens to compile.

    :return: Hex digest of the sorted ``"<repo-relative-posix-path>\\0
        <file-sha256>\\n"`` entries for every file under ``FOF_DLL/``.
    """
    entries = []
    for dirpath, _dirnames, filenames in os.walk(FOF_DLL_DIR):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, FOF_DLL_DIR).replace(os.sep, "/")
            entries.append((rel, sha256_file(full)))
    entries.sort(key=lambda pair: pair[0])
    digest = hashlib.sha256()
    for rel, file_hash in entries:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cl_exe_path(env: Dict[str, str]) -> Optional[str]:
    """Return the absolute path to ``cl.exe`` on *env*'s ``PATH``, or
    ``None`` if not found (Windows resolves an executable name against the
    CALLING process's PATH, not a ``env=`` dict handed to ``Popen`` --
    an absolute path is required)."""
    for p in env.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, "cl.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _compiler_identity(env: Dict[str, str], cl_path: str) -> str:
    """Return ``cl.exe``'s own version banner (its first stderr line when
    invoked with no arguments) -- never a generic "MSVC" claim."""
    try:
        out = run_bounded([cl_path], env=env, timeout=30).stderr or ""
    except (ProcTimeout, OSError) as exc:
        return f"unknown ({exc})"
    lines = out.splitlines()
    return lines[0].strip() if lines else "unknown (no banner output)"


def _count_divide_by_zero(text: str) -> Dict[str, int]:
    """Count the pinned source's own runtime ``xZero()`` diagnostics
    (``BM_Util.cpp``), by the variable name each call site names."""
    return dict(Counter(_DIVIDE_BY_ZERO_RE.findall(text)))


def _duplicate_symbol_warnings(*streams: str) -> List[str]:
    """Return every ``LNK4006``/``LNK4221`` duplicate-symbol warning line
    found in ANY of *streams* (both compiler stdout and stderr must be
    checked -- MSVC's exact stream choice for a given diagnostic is not a
    contract this probe should rely on)."""
    warnings: List[str] = []
    for stream in streams:
        for line in (stream or "").splitlines():
            if "LNK4006" in line or "LNK4221" in line:
                warnings.append(line)
    return warnings


def _fof_dll_git_status(
        *, repo_dir: str = CPP_REFERENCE_DIR, pathspec: str = "FOF_DLL",
) -> Dict[str, object]:
    """
    Check that *pathspec* under *repo_dir* has no tracked modification,
    staged change, deletion, or untracked file. Defaults to
    :data:`FOF_DLL_DIR`'s repo-relative pathspec (``"FOF_DLL"``) scoped
    within the real ``reference/fofem_cpp`` submodule -- so the separately
    expected, pre-existing ``FOF_UNIX``/overlay dirtiness elsewhere in
    that submodule is never rejected. *repo_dir*/*pathspec* are
    overridable so tests can point this at a disposable, process-owned
    repository instead (see ``test_massman_fof_dll_probe.py``'s dedicated
    ``_fof_dll_git_status`` tests) rather than the real submodule.

    :param repo_dir: Git repository root to check.
    :param pathspec: Pathspec (relative to *repo_dir*) to scope the check
        to.
    :return: ``{"clean": bool, "porcelain": str}`` on a successful git
        invocation, or ``{"clean": False, "error": str}`` if git itself
        could not be run/failed.
    """
    try:
        result = run_bounded(
            ["git", "-C", repo_dir, "status", "--porcelain", "--", pathspec],
            timeout=TIMEOUT_GIT_S,
        )
    except (ProcTimeout, OSError) as exc:
        return {"clean": False, "error": f"git status invocation failed: {exc}"}
    if result.returncode != 0:
        return {
            "clean": False,
            "error": f"git status exited {result.returncode}: {result.stderr}",
        }
    porcelain = result.stdout
    return {"clean": porcelain.strip() == "", "porcelain": porcelain}


def _parse_probe_result(stdout: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """
    Strictly parse and validate exactly one ``PROBE_RESULT|...`` line
    against the driver's exact required-field schema.

    :param stdout: The probe binary's captured stdout.
    :return: ``(fields, None)`` on success, or ``(None, reason)`` if the
        line is absent, duplicated, or any field is missing, duplicated,
        malformed, or outside its allowed domain.
    """
    matches = _PROBE_RESULT_RE.findall(stdout)
    if not matches:
        return None, "PROBE_RESULT line absent from probe stdout"
    if len(matches) > 1:
        return None, f"PROBE_RESULT line duplicated ({len(matches)} occurrences)"

    required = set(_SCALAR_INT_FIELDS) | set(_SCALAR_BOOL_FIELDS) | {"errmes"}
    for group in _FIELD_GROUPS:
        for metric in _FIELD_METRICS:
            required.add(f"{group}_{metric}")

    raw: Dict[str, str] = {}
    for part in matches[0].split("|"):
        if "=" not in part:
            return None, f"malformed field (no '='): {part!r}"
        key, _, value = part.partition("=")
        if key in raw:
            return None, f"duplicated field key: {key!r}"
        raw[key] = value

    missing = required - set(raw)
    extra = set(raw) - required
    if missing:
        return None, f"missing required field(s): {sorted(missing)}"
    if extra:
        return None, f"unexpected extra field(s): {sorted(extra)}"

    fields: Dict[str, object] = {"errmes": raw["errmes"]}

    for key in _SCALAR_INT_FIELDS:
        try:
            fields[key] = int(raw[key])
        except ValueError:
            return None, f"field {key!r} is not a valid integer: {raw[key]!r}"

    for key in _SCALAR_BOOL_FIELDS:
        try:
            value = int(raw[key])
        except ValueError:
            return None, f"field {key!r} is not a valid integer: {raw[key]!r}"
        if value not in (0, 1):
            return None, f"field {key!r} is not 0/1: {value!r}"
        fields[key] = value

    for group in _FIELD_GROUPS:
        total_key = f"{group}_total"
        finite_key = f"{group}_finite"
        try:
            total = int(raw[total_key])
            finite = int(raw[finite_key])
        except ValueError:
            return None, f"field {total_key!r}/{finite_key!r} is not a valid integer"
        if total < 0 or finite < 0 or finite > total:
            return None, (
                f"field {group!r} counts out of domain: total={total} finite={finite}"
            )
        fields[total_key] = total
        fields[finite_key] = finite

        for bool_metric in ("any_nan", "any_inf"):
            key = f"{group}_{bool_metric}"
            try:
                value = int(raw[key])
            except ValueError:
                return None, f"field {key!r} is not a valid integer: {raw[key]!r}"
            if value not in (0, 1):
                return None, f"field {key!r} is not 0/1: {value!r}"
            fields[key] = value

        for float_metric in ("first", "last"):
            key = f"{group}_{float_metric}"
            try:
                fields[key] = float(raw[key])
            except ValueError:
                return None, f"field {key!r} is not a valid float: {raw[key]!r}"

    # Cross-field invariants the driver's own measurement loop guarantees
    # by construction -- enforced here so a malformed/impossible
    # combination (e.g. a total that doesn't match hta_count*hta_layers,
    # or "finite" and "any_nan" both claiming the same sample) fails
    # closed rather than being silently accepted as a well-formed result.
    if fields["hta_count"] <= 0:
        return None, f"hta_count is not positive: {fields['hta_count']!r}"
    if fields["hta_layers"] <= 0:
        return None, f"hta_layers is not positive: {fields['hta_layers']!r}"

    expected_total = fields["hta_count"] * fields["hta_layers"]
    for group in _FIELD_GROUPS:
        total = fields[f"{group}_total"]
        finite = fields[f"{group}_finite"]
        any_nan = fields[f"{group}_any_nan"]
        any_inf = fields[f"{group}_any_inf"]
        first = fields[f"{group}_first"]
        last = fields[f"{group}_last"]

        if total != expected_total:
            return None, (
                f"field {group!r}_total={total} does not equal "
                f"hta_count*hta_layers={expected_total}"
            )
        if finite == total and (any_nan or any_inf):
            return None, (
                f"field {group!r} claims finite==total ({finite}) but also "
                f"sets any_nan={any_nan}/any_inf={any_inf}"
            )
        if (any_nan or any_inf) and not (finite < total):
            return None, (
                f"field {group!r} sets any_nan={any_nan}/any_inf={any_inf} "
                f"but finite={finite} is not less than total={total}"
            )
        if not (any_nan or any_inf) and finite != total:
            return None, (
                f"field {group!r} sets neither any_nan nor any_inf, but "
                f"finite={finite} does not equal total={total}"
            )
        if finite == total and not (math.isfinite(first) and math.isfinite(last)):
            return None, (
                f"field {group!r} claims every sample is finite "
                f"(finite==total=={total}) but first={first!r}/last={last!r} "
                "is not finite"
            )

    return fields, None


def _scratch_root() -> str:
    """
    Return (creating if needed) a directory reserved for this probe's own
    disposable build artifacts, rooted under this repository's own tree
    (never the system/user temp directory) so every byte this probe
    writes stays inside the checkout, per the authorized filesystem
    boundary. Verified via :func:`_verify_under_project_root` before use.
    """
    root = os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", ".massman_probe_scratch"
    )
    _verify_under_project_root(root)
    os.makedirs(root, exist_ok=True)
    return root


def _values_equal(fields_a: Dict[str, object], fields_b: Dict[str, object]) -> bool:
    """Return whether two parsed :func:`_parse_probe_result` dicts are
    identical in every field (used to require exact run-to-run
    determinism -- see :func:`run_probe`).

    A plain ``==`` on the two dicts would be WRONG here: IEEE 754 NaN is
    never equal to itself (``float('nan') != float('nan')``), so two
    runs that both genuinely produced NaN in the same field would
    otherwise be reported as non-deterministic, which is false. Two NaN
    values in the same field count as equal for this determinism check;
    every other value (including +/-inf, which DOES compare equal to
    itself) uses ordinary equality."""
    if set(fields_a) != set(fields_b):
        return False
    for key, value_a in fields_a.items():
        value_b = fields_b[key]
        if isinstance(value_a, float) and isinstance(value_b, float):
            if math.isnan(value_a) and math.isnan(value_b):
                continue
        if value_a != value_b:
            return False
    return True


def _verify_under_project_root(path: str) -> str:
    """
    Resolve *path* and raise ``RuntimeError`` unless it is
    :data:`PROJECT_ROOT` itself or a descendant of it -- called before
    this probe creates OR deletes anything, so a bug can never make it
    operate outside the checkout.

    :param path: Path to verify (need not yet exist).
    :return: The resolved (``os.path.realpath``) absolute path.
    :raises RuntimeError: If *path* resolves outside :data:`PROJECT_ROOT`.
    """
    real = os.path.realpath(path)
    root_real = os.path.realpath(PROJECT_ROOT)
    if real != root_real and not real.startswith(root_real + os.sep):
        raise RuntimeError(
            f"refusing to operate outside PROJECT_ROOT: {real!r} is not "
            f"under {root_real!r}"
        )
    return real


def main() -> int:
    """CLI entry point: run the probe, print its report as JSON, and
    return a process exit code reflecting whether every stage completed
    (not whether the pinned SOLVER itself reported success -- see F-58,
    nor whether its output was finite -- see ``solver_output_finite``)."""
    report = run_probe()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("stage") == "complete" else 1


def run_probe() -> Dict[str, object]:
    """
    Run the full fail-closed probe (FOF_DLL cleanliness -> pinned-SHA
    check -> provenance digests -> toolchain discovery -> compile -> link
    -> two bounded runs -> strict schema validation -> cross-run equality)
    and return a JSON-serializable report dict. Never raises for an
    ordinary build/run/validation failure -- those are reported via the
    dict's own ``"error"``/``"stage"`` keys so a caller always gets a
    complete, inspectable report; only a genuinely unexpected internal
    exception propagates.

    :return: Report dict; ``report["stage"] == "complete"`` iff every
        structural check passed (scientific non-finiteness is NOT a
        structural failure -- see ``solver_output_finite``).
    """
    report: Dict[str, object] = {
        "pinned_upstream_sha": PINNED_UPSTREAM_SHA,
        "fof_dll_source_count": len(FOF_DLL_SOURCES),
        "fof_dll_sources": list(FOF_DLL_SOURCES),
    }

    fof_dll_status = _fof_dll_git_status()
    report["fof_dll_git_status"] = fof_dll_status
    if not fof_dll_status.get("clean"):
        report["stage"] = "fof_dll_cleanliness_check"
        report["error"] = (
            "reference/fofem_cpp/FOF_DLL has uncommitted/untracked changes: "
            f"{fof_dll_status}"
        )
        return report

    try:
        check_pinned_sha()
    except ProvenanceError as exc:
        report["stage"] = "pinned_sha_check"
        report["error"] = str(exc)
        return report
    report["pinned_sha_check"] = "PASSED"

    missing = [
        f for f in FOF_DLL_SOURCES
        if not os.path.isfile(os.path.join(FOF_DLL_DIR, f))
    ]
    missing_headers = [
        h for h in CONSUMED_HEADERS
        if not os.path.isfile(os.path.join(FOF_DLL_DIR, h))
    ]
    if missing or missing_headers:
        report["stage"] = "source_closure_check"
        report["error"] = (
            f"missing pinned FOF_DLL source file(s): {missing}; "
            f"missing header(s): {missing_headers}"
        )
        return report

    report["provenance"] = {
        "driver_source_sha256": sha256_file(DRIVER_SOURCE),
        "probe_script_sha256": sha256_file(PROBE_SCRIPT),
        "fof_dll_source_sha256": {
            f: sha256_file(os.path.join(FOF_DLL_DIR, f)) for f in FOF_DLL_SOURCES
        },
        "consumed_header_sha256": {
            h: sha256_file(os.path.join(FOF_DLL_DIR, h)) for h in CONSUMED_HEADERS
        },
        "fof_dll_directory_aggregate_sha256": _aggregate_fof_dll_digest(),
    }

    env = _msvc_env()
    if env is None:
        report["stage"] = "toolchain_discovery"
        report["error"] = "could not source the MSVC/CMake/Ninja build environment"
        return report
    cl_path = _cl_exe_path(env)
    if cl_path is None:
        report["stage"] = "toolchain_discovery"
        report["error"] = "cl.exe not found on the sourced MSVC PATH"
        return report
    report["vs_installation_path"] = _vs_installation_path()
    report["compiler_identity"] = _compiler_identity(env, cl_path)
    report["cl_exe_path"] = cl_path

    scratch_root = _scratch_root()
    with tempfile.TemporaryDirectory(
        prefix="probe_", dir=scratch_root,
    ) as tmp_dir:
        _verify_under_project_root(tmp_dir)
        out_exe = os.path.join(tmp_dir, "probe.exe")
        source_paths = [os.path.join(FOF_DLL_DIR, f) for f in FOF_DLL_SOURCES]
        compile_flags = [
            "/nologo", "/EHsc", "/W3", "/wd4578", "/D_CRT_SECURE_NO_WARNINGS",
            "/I", FOF_DLL_DIR,
        ]
        compile_cmd = (
            [cl_path] + compile_flags + [DRIVER_SOURCE] + source_paths
            + ["/Fe:" + out_exe, "/Fo:" + tmp_dir + os.sep]
        )
        report["compile_flags"] = compile_flags
        try:
            compile_result = run_bounded(
                compile_cmd, cwd=tmp_dir, env=env, timeout=TIMEOUT_COMPILE_S,
            )
        except ProcTimeout as exc:
            report["stage"] = "compile_link"
            report["error"] = f"compile/link timed out: {exc}"
            return report
        report["compile_returncode"] = compile_result.returncode
        duplicate_symbol_warnings = _duplicate_symbol_warnings(
            compile_result.stdout, compile_result.stderr
        )
        report["duplicate_symbol_warning_count"] = len(duplicate_symbol_warnings)
        report["duplicate_symbol_warnings"] = duplicate_symbol_warnings
        if compile_result.returncode != 0 or duplicate_symbol_warnings:
            report["stage"] = "compile_link"
            report["error"] = (
                "compile/link failed"
                if compile_result.returncode != 0
                else "duplicate-symbol warning(s) present"
            )
            report["compile_stdout_tail"] = "\n".join(
                (compile_result.stdout or "").splitlines()[-60:]
            )
            report["compile_stderr_tail"] = "\n".join(
                (compile_result.stderr or "").splitlines()[-60:]
            )
            return report
        if not os.path.isfile(out_exe):
            report["stage"] = "compile_link"
            report["error"] = "compile/link reported success but the binary is missing"
            return report

        run_results: List[Dict[str, object]] = []
        for run_index in range(2):
            try:
                run_result = run_bounded(
                    [out_exe], cwd=tmp_dir, env=env, timeout=TIMEOUT_RUN_S,
                    stdin=subprocess.DEVNULL,
                )
            except ProcTimeout as exc:
                report["stage"] = "run"
                report["error"] = f"probe binary run #{run_index} timed out: {exc}"
                return report
            except (UnicodeDecodeError, OSError) as exc:
                report["stage"] = "run"
                report["error"] = (
                    f"probe binary run #{run_index} output could not be decoded: {exc}"
                )
                return report
            if run_result.returncode != 0:
                report["stage"] = "run"
                report["error"] = (
                    f"probe binary run #{run_index} exited "
                    f"{run_result.returncode} (nonzero)"
                )
                report["run_stdout_tail"] = "\n".join(
                    (run_result.stdout or "").splitlines()[-60:]
                )
                return report
            duplicate_symbol_at_runtime = _duplicate_symbol_warnings(
                run_result.stdout, run_result.stderr
            )
            if duplicate_symbol_at_runtime:
                report["stage"] = "run"
                report["error"] = (
                    f"probe binary run #{run_index} emitted duplicate-symbol "
                    f"warning text at runtime: {duplicate_symbol_at_runtime}"
                )
                return report
            fields, parse_error = _parse_probe_result(run_result.stdout or "")
            if fields is None:
                report["stage"] = "run"
                report["error"] = (
                    f"probe binary run #{run_index} result failed schema "
                    f"validation: {parse_error}"
                )
                report["run_stdout_tail"] = "\n".join(
                    (run_result.stdout or "").splitlines()[-60:]
                )
                return report
            # A nonpositive hta_count/hta_layers, or a total mismatched
            # against hta_count*hta_layers, is already rejected by
            # _parse_probe_result()'s own cross-field validation above --
            # fields is only ever non-None here once those hold.
            if fields["hta_get_failed"]:
                report["stage"] = "run"
                report["error"] = (
                    f"probe binary run #{run_index} reported hta_get_failed=1 "
                    "(HTA_Get returned an unexpected failure)"
                )
                return report
            divide_by_zero = _count_divide_by_zero(run_result.stdout or "")
            run_results.append({
                "run_index": run_index,
                "process_returncode": run_result.returncode,
                "divide_by_zero_diagnostic_counts": divide_by_zero,
                "probe_result": fields,
            })
        report["runs"] = run_results

        if not _values_equal(
            run_results[0]["probe_result"], run_results[1]["probe_result"]
        ) or run_results[0]["divide_by_zero_diagnostic_counts"] != run_results[1][
            "divide_by_zero_diagnostic_counts"
        ]:
            report["stage"] = "run"
            report["error"] = "the two runs did not produce identical results"
            return report
        report["runs_identical"] = True

        # tmp_dir and every compiled .obj/.exe under it are deleted here by
        # TemporaryDirectory's own context-manager exit -- never inside
        # reference/fofem_cpp/ or reference/fofem_cpp_overlay/, and no
        # file under either was ever opened for writing by this probe.

    canonical = run_results[0]["probe_result"]
    solver_output_finite = all(
        canonical[f"{group}_finite"] == canonical[f"{group}_total"]
        for group in _FIELD_GROUPS
    )
    report["solver_output_finite"] = solver_output_finite
    report["stage"] = "complete"
    return report


if __name__ == "__main__":
    sys.exit(main())
