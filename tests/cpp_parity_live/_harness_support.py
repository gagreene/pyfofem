#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_harness_support.py - Drive the live fofem_test C++ harness from pytest.

Locates the MSVC/CMake/Ninja toolchain (bundled inside a Visual Studio
Build Tools install, discovered via ``vswhere.exe`` — matches the manual
toolchain-discovery steps used to build this harness during Phase 2
development; nothing here is a hardcoded personal path), verifies the C++
checkout is at the one pinned SHA (fail-closed — a different checkout must
never build/configure/qualify/generate against), builds ``fofem_test.exe``
if missing or stale, and provides a thin Python wrapper around invoking it:
correct working directory (``FOF_UNIX/``, so ``NES_Read("")``/
``Emission_Factors.csv`` resolve — fof_nes.cpp:277), exact CLI argument
construction, and CSV output parsing.

This module intentionally does nothing on a non-Windows host or a Windows
host without the VC.Tools.x86.x64 component: :func:`toolchain_status`
reports why, and callers use that to skip cleanly rather than fail — the
harness itself is Windows/MSVC-only by construction (Phase 2 scope), not a
missing-infrastructure gap on a machine that does have the toolchain.

Every subprocess call in this module is routed through
:func:`tests.cpp_parity_live._proc.run_bounded`, which kills the FULL
descendant process tree on timeout (plain ``subprocess.run(timeout=...)``
does not guarantee this on Windows — see ``_proc.py``): a hung
compiler-discovery, configure, build, or harness invocation fails
deterministically rather than hanging the caller or leaving zombie
descendants.

Diagnostic harness override: set ``FOFEM_TEST_HARNESS_EXE`` to the
absolute path of an alternate compiled binary (e.g. the AddressSanitizer
diagnostic build at ``reference/fofem_cpp/build_diag/fofem_test.exe``) to
make :func:`run_harness` invoke it instead of the default golden/release
build. The override is validated on every call (:func:`resolve_harness_exe`
raises :class:`HarnessConfigError` if it does not point at a real file) —
never silently falls back.

Function order: top-level functions are alphabetized, private-then-public,
per AGENTS.md. ``HarnessResult`` (a type, not a function) is declared
ahead of the alphabetized blocks since it is this module's return-type
declaration, analogous to a module constant.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from tests._support import CPP_REFERENCE_DIR, PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import ProvenanceError, check_pinned_sha
from tests.cpp_parity_live._proc import ProcTimeout, run_bounded

FOF_UNIX_DIR = os.path.join(CPP_REFERENCE_DIR, "FOF_UNIX")
BUILD_DIR = os.path.join(CPP_REFERENCE_DIR, "build")
HARNESS_EXE = os.path.join(BUILD_DIR, "fofem_test.exe")
HARNESS_SOURCE = os.path.join(FOF_UNIX_DIR, "test_harness.cpp")
SPECIES_CSV = os.path.join(
    PROJECT_ROOT, "src", "pyfofem", "supporting_data", "FOFEM6.7", "FOF_SPP.CSV"
)
PREPARE_SCRIPT = os.path.join(PROJECT_ROOT, "tests", "prepare_cpp_reference.py")

#: Environment variable that, if set, overrides which compiled harness
#: binary :func:`run_harness` invokes (see module docstring).
HARNESS_EXE_OVERRIDE_ENV_VAR = "FOFEM_TEST_HARNESS_EXE"

_VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

#: Bounded timeouts (seconds) for every subprocess this module spawns.
#: Chosen generously (a cold ASan-instrumented rebuild can take real time)
#: but finite: a hang must fail loudly, never block the caller forever.
TIMEOUT_DISCOVERY_S = 30
TIMEOUT_ENV_SOURCE_S = 30
TIMEOUT_PREPARE_S = 60
TIMEOUT_CONFIGURE_S = 120
TIMEOUT_BUILD_S = 600
TIMEOUT_HARNESS_RUN_S = 60

_env_cache: Dict[str, Optional[Dict[str, str]]] = {}


class HarnessConfigError(RuntimeError):
    """Raised when :data:`HARNESS_EXE_OVERRIDE_ENV_VAR` is set to a path
    that is not a real, usable file — never silently ignored/fallen back
    from."""


class HarnessTimeout(RuntimeError):
    """Raised when the harness process (and its full descendant tree, via
    :func:`tests.cpp_parity_live._proc.run_bounded`) exceeded
    :data:`TIMEOUT_HARNESS_RUN_S`."""


class HarnessResult:
    """Result of one ``fofem_test`` invocation."""

    def __init__(self, returncode: int, stdout: str, stderr: str,
                 output_files: Dict[str, str]):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        #: Maps a logical suffix (e.g. ``""``, ``"_summary"``, ``"_trees"``)
        #: to the CSV rows found at ``<prefix><suffix>.csv``, if that file
        #: exists. Empty dict if the harness exited before writing anything.
        self.output_files = output_files

    def rows(self, suffix: str = "") -> List[Dict[str, str]]:
        return self._parsed.get(suffix, [])

    def stdout_field(self, key: str) -> Optional[str]:
        """Extract a ``KEY=value`` diagnostic line from stdout (e.g.
        ``SPECIES_TABLE_SHA256``)."""
        prefix = key + "="
        for line in self.stdout.splitlines():
            if line.startswith(prefix):
                return line[len(prefix):]
        return None


def _msvc_env() -> Optional[Dict[str, str]]:
    """
    Return the environment produced by sourcing vcvars64.bat with CMake and
    Ninja (bundled inside the VS install, not on PATH by default) prepended
    to PATH — cached per process since sourcing it is not free.
    """
    if "env" in _env_cache:
        return _env_cache["env"]

    ok, _ = toolchain_status()
    if not ok:
        _env_cache["env"] = None
        return None

    vs_path = _vs_installation_path()
    vcvars64 = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    cmake_bin_dir = os.path.join(
        vs_path, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "CMake", "bin"
    )
    ninja_dir = os.path.join(
        vs_path, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja"
    )

    # A batch FILE, not a one-line `cmd /c "... && ..."` string: cmd.exe's
    # single-line parser can mis-tokenize the literal parentheses in
    # "Program Files (x86)" inside an `&&`-chained command list (observed
    # directly: "The system cannot find the path specified" even though
    # every path is correct). A .bat file is parsed line-by-line and does
    # not hit this.
    marker = "___FOFEM_ENV_START___"
    bat_content = (
        "@echo off\r\n"
        f'set "PATH={cmake_bin_dir};{ninja_dir};%PATH%"\r\n'
        f'call "{vcvars64}" >nul 2>&1\r\n'
        f"echo {marker}\r\n"
        "set\r\n"
    )
    fd, bat_path = tempfile.mkstemp(suffix=".bat")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(bat_content)
        out = run_bounded(["cmd.exe", "/c", bat_path], timeout=TIMEOUT_ENV_SOURCE_S).stdout
    except (ProcTimeout, Exception):
        _env_cache["env"] = None
        return None
    finally:
        os.remove(bat_path)

    if marker not in out:
        _env_cache["env"] = None
        return None

    env: Dict[str, str] = {}
    for line in out.split(marker, 1)[1].splitlines():
        line = line.strip("\r")
        if "=" in line:
            k, _, v = line.partition("=")
            if k:
                env[k] = v

    if not env or shutil.which("cmake", path=env.get("PATH", "")) is None:
        _env_cache["env"] = None
        return None

    _env_cache["env"] = env
    return env


def _vs_installation_path() -> Optional[str]:
    if os.name != "nt" or not os.path.isfile(_VSWHERE):
        return None
    try:
        out = run_bounded(
            [
                _VSWHERE,
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            timeout=TIMEOUT_DISCOVERY_S,
        ).stdout.strip()
    except (ProcTimeout, Exception):
        return None
    return out or None


def ensure_built() -> Tuple[bool, str]:
    """
    Ensure ``fofem_test.exe`` exists, reflecting the current overlay source.

    Checks the pinned C++ SHA (fail-closed) BEFORE reapplying the overlay
    or configuring/building anything. Reapplies the overlay
    (non-destructively; never refreshes/resets the submodule — see
    ``tests/prepare_cpp_reference.py``), then configures and builds only
    the ``fofem_test`` target.

    :return: ``(True, "")`` on success; ``(False, reason)`` otherwise.
    """
    ok, reason = toolchain_status()
    if not ok:
        return False, reason

    try:
        check_pinned_sha()
    except ProvenanceError as exc:
        return False, str(exc)

    env = _msvc_env()
    if env is None:
        return False, "failed to source the MSVC/CMake/Ninja build environment"

    # Windows CreateProcess resolves the executable name against the
    # *calling* process's PATH, not the `env=` block handed to Popen — an
    # absolute path is required, or "cmake" silently resolves against
    # os.environ instead of the vcvars64-augmented env (or isn't found at
    # all if this Python process has no cmake on its own PATH).
    cmake_exe = shutil.which("cmake", path=env.get("PATH", ""))
    if cmake_exe is None:
        return False, "cmake not found on the sourced MSVC/CMake/Ninja PATH"

    try:
        r = run_bounded(
            [sys.executable, PREPARE_SCRIPT], cwd=PROJECT_ROOT, timeout=TIMEOUT_PREPARE_S,
        )
    except ProcTimeout as exc:
        return False, f"prepare_cpp_reference.py timed out: {exc}"
    if r.returncode != 0:
        return False, f"prepare_cpp_reference.py failed:\n{r.stdout}\n{r.stderr}"

    try:
        configure = run_bounded(
            [cmake_exe, "-S", CPP_REFERENCE_DIR, "-B", BUILD_DIR, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Debug"],
            cwd=CPP_REFERENCE_DIR, env=env, timeout=TIMEOUT_CONFIGURE_S,
        )
    except ProcTimeout as exc:
        return False, f"cmake configure timed out: {exc}"
    if configure.returncode != 0:
        return False, f"cmake configure failed:\n{configure.stdout}\n{configure.stderr}"

    try:
        build = run_bounded(
            [cmake_exe, "--build", BUILD_DIR, "--target", "fofem_test"],
            cwd=CPP_REFERENCE_DIR, env=env, timeout=TIMEOUT_BUILD_S,
        )
    except ProcTimeout as exc:
        return False, f"cmake --build timed out: {exc}"
    if build.returncode != 0:
        return False, f"cmake --build failed:\n{build.stdout}\n{build.stderr}"

    if not os.path.isfile(HARNESS_EXE):
        return False, f"build reported success but {HARNESS_EXE} is missing"

    return True, ""


def resolve_harness_exe() -> str:
    """
    Return the compiled harness binary :func:`run_harness` should invoke:
    :data:`HARNESS_EXE` (the default golden/release build) unless
    :data:`HARNESS_EXE_OVERRIDE_ENV_VAR` is set, in which case that path is
    validated and used instead — e.g. to point at the separate ASan
    diagnostic build for the diagnostic-matrix re-run.

    :return: Absolute path to the harness binary to invoke.
    :raises HarnessConfigError: If the override env var is set but does not
        point at a real file.
    """
    override = os.environ.get(HARNESS_EXE_OVERRIDE_ENV_VAR)
    if override is None:
        return HARNESS_EXE
    if not os.path.isfile(override):
        raise HarnessConfigError(
            f"{HARNESS_EXE_OVERRIDE_ENV_VAR}={override!r} does not exist or "
            "is not a file — refusing to silently fall back to the default "
            "harness binary"
        )
    return override


def run_harness(
        mode: str,
        header: List[str],
        rows: List[List[str]],
        out_prefix: str,
        schema_version: str = "1",
        species_csv: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        output_suffixes: Tuple[str, ...] = ("",),
        magic_override: Optional[str] = None,
        header_override: Optional[List[str]] = None,
) -> "HarnessResult":
    """
    Write an input CSV for *mode* and invoke ``fofem_test`` (or its
    :func:`resolve_harness_exe` override) against it.

    Every declared output path (``<out_prefix><suffix>.csv`` for each of
    *output_suffixes*) is removed BEFORE invocation (a stale file from an
    earlier run at the same prefix can never be mistaken for this run's
    output) and removed again if the call times out or raises unexpectedly
    (a partially-written file from a killed process can never be mistaken
    for complete output).

    :param mode: Harness mode name (used in the magic line unless
        *magic_override* is given).
    :param header: Column header for *mode* (ignored if *header_override*
        is given).
    :param rows: List of row field-lists, written comma-joined verbatim
        (deliberately not CSV-quoted — the contract's input format has no
        quoting, so a malformed-input self-test can inject raw commas).
    :param out_prefix: Absolute output-prefix path passed to the harness.
    :param schema_version: Value written on the magic line.
    :param species_csv: If given, appended as ``--species-csv <path>``.
    :param extra_args: Additional raw CLI arguments appended after
        ``--species-csv`` (if any) — used by CLI self-tests.
    :param output_suffixes: Which ``<out_prefix><suffix>.csv`` files to try
        parsing after the run (missing files are simply absent from
        ``output_files``/``rows()``).
    :param magic_override: Raw magic line to write instead of the derived
        ``#fofem-harness,<mode>,<schema_version>`` (for malformed-input
        self-tests).
    :param header_override: Raw header line fields to write instead of
        *header* (for malformed-header self-tests).
    :return: A populated :class:`HarnessResult`.
    :raises HarnessTimeout: If the harness process exceeds
        :data:`TIMEOUT_HARNESS_RUN_S` — the full descendant process tree is
        already killed (via :func:`tests.cpp_parity_live._proc.run_bounded`)
        and any partial output removed before this is raised.
    :raises HarnessConfigError: Via :func:`resolve_harness_exe`.
    """
    in_path = out_prefix + "_in.csv"
    lines = []
    lines.append(magic_override if magic_override is not None
                 else f"#fofem-harness,{mode},{schema_version}")
    lines.append(",".join(header_override if header_override is not None else header))
    for row in rows:
        lines.append(",".join(row))
    with open(in_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    output_paths = [out_prefix + suffix + ".csv" for suffix in output_suffixes]
    for p in output_paths:
        if os.path.isfile(p):
            os.remove(p)

    harness_exe = resolve_harness_exe()
    args = [harness_exe, in_path, out_prefix]
    if species_csv is not None:
        args += ["--species-csv", species_csv]
    if extra_args:
        args += extra_args

    try:
        proc = run_bounded(args, cwd=FOF_UNIX_DIR, timeout=TIMEOUT_HARNESS_RUN_S)
    except ProcTimeout as exc:
        for p in output_paths:
            if os.path.isfile(p):
                os.remove(p)
        raise HarnessTimeout(
            f"fofem_test timed out (mode={mode!r}, out_prefix={out_prefix!r}): {exc}"
        ) from exc
    except Exception:
        for p in output_paths:
            if os.path.isfile(p):
                os.remove(p)
        raise

    # Round 4 correction item 5 raised "a process-level nonzero return
    # must never leave accepted output", reasoning that expected model
    # errors are always zero-exit. Direct evidence contradicts that
    # premise for THIS harness: row 12's "row unexpectedly errors" family
    # (test_row12_row_unexpectedly_errors_bark_thick/canopy_cover/mortality)
    # deliberately returns NONZERO **and** writes a complete, valid output
    # row whose own `outcome` column says "unexpected_failure" — that is
    # the intended signal (loud on two channels at once, not silent), and
    # test_harness.cpp's CsvWriter already fails closed on row-width/
    # write/flush/close failure (round-3 hardening), so any file that
    # survives a NORMAL process exit (any return code) is, by the C++
    # side's own contract, already complete — there is no observed
    # "genuinely partial file after a normal nonzero exit" case to guard
    # against here. A blanket wipe-on-nonzero-return was tried and
    # reverted after directly reproducing that it deletes real,
    # already-tested, intentionally-preserved diagnostic output (see the
    # three tests above, which failed with IndexError once the wipe was
    # added). Left AS-IS pending explicit user sign-off — see the round 4
    # correction report's item 5 write-up. The ProcTimeout/Exception
    # branches above still remove output when the process was killed or
    # genuinely failed to complete, which is the case this module can
    # actually observe a partial/incomplete file from.

    output_files: Dict[str, List[Dict[str, str]]] = {}
    for suffix, path in zip(output_suffixes, output_paths):
        if os.path.isfile(path):
            with open(path, newline="", encoding="utf-8") as f:
                output_files[suffix] = list(csv.DictReader(f))

    result = HarnessResult(proc.returncode, proc.stdout, proc.stderr, {})
    result._parsed = output_files
    return result


def toolchain_status() -> Tuple[bool, str]:
    """
    Report whether the MSVC/CMake/Ninja toolchain needed to build and run
    the live C++ harness is available on this host.

    :return: ``(True, "")`` if available; ``(False, reason)`` otherwise.
    """
    if os.name != "nt":
        return False, "the fofem_test C++ harness build is Windows/MSVC-only"
    vs_path = _vs_installation_path()
    if vs_path is None:
        return False, (
            "vswhere.exe did not find a Visual Studio Build Tools install "
            "with the VC.Tools.x86.x64 component"
        )
    vcvars64 = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(vcvars64):
        return False, f"vcvars64.bat not found under {vs_path}"
    return True, ""
