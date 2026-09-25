#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_massman_fof_dll_probe.py - Contract tests for
``massman_fof_dll_probe.py``'s fail-closed orchestration
(:func:`tests.cpp_parity_live.massman_fof_dll_probe.run_probe`), using
injected/mocked :class:`~tests.cpp_parity_live._proc.BoundedResult`-shaped
values. **None of these tests compiles or links FOF_DLL** - every
``run_bounded`` call ``run_probe()`` makes is monkeypatched with a stub
that never invokes a real compiler/linker/binary, so this module is safe
to run on any machine, with or without the MSVC toolchain, and is
registered in ``CORE_TESTS`` (fast, no live C++ build).

Real compile/link/run evidence lives in ``massman_fof_dll_probe.py``
itself (run manually - see its own module docstring); this module only
proves the SURROUNDING fail-closed logic reacts correctly to every
structural failure mode the Phase 6 probe-hardening pass requires, using
the exact same parsing/validation/equality code paths a real run would
exercise.

Every ordinary orchestration test below (see ``_install_stub()``) mocks
``check_pinned_sha()`` and ``_fof_dll_git_status()`` in addition to
toolchain discovery and ``run_bounded()`` - no real ``git`` process runs
at all for those tests, so they cannot fail due to a machine's Git
``safe.directory``/ownership configuration, the real submodule's live
state, or MSVC availability (a real failure this fixes: an earlier
version of this module left both real, and a CI account whose Git
configuration refused to treat ``reference/fofem_cpp`` as safe made every
one of these tests fail at the FOF_DLL-cleanliness stage before
exercising the behaviour each test actually names). ``_fof_dll_git_status()``
itself has dedicated tests further down that exercise its REAL behaviour,
using either injected ``run_bounded`` results or a disposable,
process-owned Git repository created via ``tmp_path`` - never the real
submodule and never global Git configuration.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

import pytest

from tests.cpp_parity_live import massman_fof_dll_probe as probe


class _FakeResult:
    """Minimal stand-in for :class:`tests.cpp_parity_live._proc.BoundedResult`."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _default_field_line(**overrides: str) -> str:
    """
    Build a well-formed ``PROBE_RESULT|...`` line matching the driver's
    exact schema, with sane all-finite defaults for a 2-sample/1-layer
    scenario. Individual key=value pairs can be overridden, added, or
    (by passing ``None``) removed, to construct each fail-closed scenario.

    :param overrides: ``field_name=value_string`` overrides; ``field_name
        =None`` removes that field entirely (to test a missing-field
        scenario).
    :return: The full ``PROBE_RESULT|...`` line (no trailing newline).
    """
    fields: Dict[str, str] = {
        "hmv_rc": "1",
        "errmes": "",
        "hta_count": "2",
        "hta_layers": "1",
        "hta_get_failed": "0",
    }
    for group in ("heat", "moist", "psin", "time"):
        fields.update({
            f"{group}_total": "2",
            f"{group}_finite": "2",
            f"{group}_any_nan": "0",
            f"{group}_any_inf": "0",
            f"{group}_first": "1.0",
            f"{group}_last": "2.0",
        })
    for key, value in overrides.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = value
    return "PROBE_RESULT|" + "|".join(f"{k}={v}" for k, v in fields.items())


def _install_stub(monkeypatch, *, compile_result: _FakeResult,
                   run_results: List[_FakeResult],
                   capture_run_kwargs: Optional[List[Dict[str, Any]]] = None):
    """
    Monkeypatch EVERY dependency :func:`~massman_fof_dll_probe.run_probe`
    needs to reach the compile/run stages, so these "ordinary
    orchestration" tests exercise ONLY the branch they name, with no
    dependency on git ownership/``safe.directory`` configuration, the
    real submodule's live state, or the MSVC toolchain: ``check_pinned_sha``
    and ``_fof_dll_git_status`` are replaced with a no-op success and a
    canned clean result respectively (real ``git`` invocations for THOSE
    checks were the exact cause of a real failure this fixes -- a
    different git ownership configuration in one CI account made
    ``_fof_dll_git_status()`` fail before any of these tests' own named
    behaviour was ever reached); toolchain discovery and ``run_bounded``
    (for the compile/run stages only) are replaced with deterministic
    stubs. File hashing/provenance digests are NOT mocked -- they are
    plain Python file I/O over files this checkout already has on disk,
    with no git/ownership dependency, so mocking them would not remove
    any real environment dependency.

    Dedicated ``_fof_dll_git_status()`` tests (see below) exercise THAT
    function's own real behaviour separately, via injected ``run_bounded``
    results or a disposable, process-owned repository -- never the real
    submodule and never global git configuration.

    The stubbed ``run_bounded`` treats its FIRST call as "compile" (and
    creates the empty file named by the command's own ``/Fe:`` argument,
    so ``run_probe()``'s own "binary is missing" check does not spuriously
    fire) and every subsequent call as one binary "run", returning
    *run_results* in order.

    :param capture_run_kwargs: If given, every RUN call's keyword
        arguments (e.g. ``stdin=``) are appended here in order, for tests
        that need to inspect exactly what was passed to ``run_bounded``.
    """
    monkeypatch.setattr(probe, "check_pinned_sha", lambda: None)
    monkeypatch.setattr(
        probe, "_fof_dll_git_status", lambda: {"clean": True, "porcelain": ""},
    )
    monkeypatch.setattr(probe, "_msvc_env", lambda: {"PATH": ""})
    monkeypatch.setattr(probe, "_cl_exe_path", lambda env: "C:\\fake\\cl.exe")
    monkeypatch.setattr(probe, "_compiler_identity", lambda env, cl_path: "fake compiler")
    monkeypatch.setattr(probe, "_vs_installation_path", lambda: "C:\\fake\\vs")

    call_index = {"n": 0}

    def _stub(cmd, **kwargs):
        n = call_index["n"]
        call_index["n"] += 1
        if n == 0:
            for arg in cmd:
                if isinstance(arg, str) and arg.startswith("/Fe:"):
                    open(arg[len("/Fe:"):], "wb").close()
            return compile_result
        if capture_run_kwargs is not None:
            capture_run_kwargs.append(kwargs)
        return run_results[n - 1]

    monkeypatch.setattr(probe, "run_bounded", _stub)


def test_all_finite_count_consistent_result_is_accepted():
    """Item 2 positive case: a count-consistent (``total ==
    hta_count*hta_layers``), fully-finite line parses with no error at
    all - the cross-field invariants below must not reject a genuinely
    well-formed result."""
    fields, error = probe._parse_probe_result(_default_field_line())
    assert error is None
    assert fields is not None
    assert fields["heat_total"] == fields["hta_count"] * fields["hta_layers"]


def test_contradictory_finite_and_inf_flag_fails_closed():
    """Item 2: a field claiming every sample is finite (``finite ==
    total``) while ALSO setting ``any_inf=1`` is an impossible
    combination and must fail closed."""
    fields, error = probe._parse_probe_result(_default_field_line(heat_any_inf="1"))
    assert fields is None
    assert "any_inf" in error


def test_contradictory_finite_and_nan_flag_fails_closed():
    """Item 2: the same impossible combination for ``any_nan``."""
    fields, error = probe._parse_probe_result(_default_field_line(heat_any_nan="1"))
    assert fields is None
    assert "any_nan" in error


def test_duplicate_symbol_warning_in_stderr_fails_closed(monkeypatch):
    """Item 3: an LNK4221 warning appearing only in STDERR (not stdout)
    must still fail closed - both streams must be checked."""
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(
            returncode=0, stdout="", stderr="warning LNK4221: duplicate\r\n"
        ),
        run_results=[],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert report["duplicate_symbol_warning_count"] == 1


def test_duplicate_symbol_warning_in_stdout_fails_closed(monkeypatch):
    """Item 3: a duplicate-symbol linker warning must fail closed even
    though the linker itself reported success (returncode 0)."""
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(
            returncode=0, stdout="warning LNK4006: X already defined\r\n"
        ),
        run_results=[],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert report["duplicate_symbol_warning_count"] == 1


def test_fof_dll_git_status_accepts_a_clean_disposable_repository(tmp_path):
    """Item 1: exercises the REAL ``_fof_dll_git_status()`` (not mocked)
    against a disposable Git repository this test creates and owns via
    ``pytest``'s own ``tmp_path`` fixture - never the real submodule,
    never global Git configuration (no ``safe.directory`` entry is
    touched). A freshly-initialized, fully-committed repository must be
    reported clean."""
    repo = tmp_path / "repo"
    fof_dll = repo / "FOF_DLL"
    fof_dll.mkdir(parents=True)
    (fof_dll / "x.cpp").write_text("// nothing\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test",
         "add", "-A"],
        cwd=repo, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test",
         "commit", "-q", "-m", "initial"],
        cwd=repo, check=True,
    )
    status = probe._fof_dll_git_status(repo_dir=str(repo), pathspec="FOF_DLL")
    assert status["clean"] is True


def test_fof_dll_git_status_detects_an_untracked_file_in_a_disposable_repository(tmp_path):
    """Companion to the clean-repository test above: an untracked file
    under the checked pathspec, in an otherwise-committed disposable
    repository, must be reported as NOT clean."""
    repo = tmp_path / "repo"
    fof_dll = repo / "FOF_DLL"
    fof_dll.mkdir(parents=True)
    (fof_dll / "x.cpp").write_text("// nothing\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test",
         "add", "-A"],
        cwd=repo, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test",
         "commit", "-q", "-m", "initial"],
        cwd=repo, check=True,
    )
    (fof_dll / "new_untracked.cpp").write_text("// new\n", encoding="utf-8")
    status = probe._fof_dll_git_status(repo_dir=str(repo), pathspec="FOF_DLL")
    assert status["clean"] is False
    assert "new_untracked.cpp" in status["porcelain"]


def test_fof_dll_git_status_reports_a_nonzero_git_exit_as_unclean(monkeypatch):
    """Item 1: an injected ``run_bounded`` result - no real git process -
    proves a nonzero git exit code (e.g. exactly the "dubious ownership"
    refusal a differently-configured Git install can produce) is treated
    as NOT clean, with the failure recorded, never silently treated as
    clean."""
    monkeypatch.setattr(
        probe, "run_bounded",
        lambda cmd, **kwargs: _FakeResult(
            returncode=128, stderr="fatal: detected dubious ownership",
        ),
    )
    status = probe._fof_dll_git_status()
    assert status["clean"] is False
    assert "128" in status["error"]


def test_fof_dll_git_status_reports_uncleanliness_from_injected_run_bounded_output(monkeypatch):
    """Item 1: the injected-``run_bounded``-results style of testing
    ``_fof_dll_git_status()`` directly, with no real git process at all."""
    monkeypatch.setattr(
        probe, "run_bounded",
        lambda cmd, **kwargs: _FakeResult(returncode=0, stdout=" M FOF_DLL/x.cpp\n"),
    )
    status = probe._fof_dll_git_status()
    assert status["clean"] is False
    assert "x.cpp" in status["porcelain"]


def test_hta_count_nonpositive_fails_closed(monkeypatch):
    """Item 3: hta_count <= 0 must fail closed (a nonpositive count means
    no sample was ever actually inspected)."""
    line = _default_field_line(hta_count="0")
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "hta_count" in report["error"]


def test_hta_get_failed_flag_fails_closed(monkeypatch):
    """Item 3: the driver's own hta_get_failed=1 flag (HTA_Get returned
    an unexpected failure) must fail closed."""
    line = _default_field_line(hta_get_failed="1")
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "hta_get_failed" in report["error"]


def test_hta_layers_nonpositive_fails_closed():
    """Item 2: ``hta_layers`` <= 0 (zero OR negative) must fail closed -
    a nonpositive layer count means no sample was ever actually
    inspected, symmetric with the ``hta_count`` check above."""
    for value in ("0", "-1"):
        fields, error = probe._parse_probe_result(_default_field_line(hta_layers=value))
        assert fields is None
        assert "hta_layers" in error


def test_mismatched_group_total_fails_closed():
    """Item 2: a field's ``total`` that does not equal
    ``hta_count * hta_layers`` must fail closed, even though every
    OTHER individual-field domain check on it passes."""
    fields, error = probe._parse_probe_result(_default_field_line(heat_total="99"))
    assert fields is None
    assert "heat" in error
    assert "hta_count*hta_layers" in error


def test_nonzero_compile_returncode_fails_closed(monkeypatch):
    """Item 3: a nonzero compile/link return code must fail closed."""
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=2, stdout="error C2065"),
        run_results=[],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert report["compile_returncode"] == 2


def test_nonzero_run_returncode_fails_closed(monkeypatch):
    """Item 3: a nonzero binary-execution return code must fail closed."""
    line = _default_field_line()
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=1, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "nonzero" in report["error"]


def test_probe_result_absent_fails_closed(monkeypatch):
    """Item 3: no PROBE_RESULT line at all must fail closed - the
    original bug this test guards against would have let two absent
    results parse as ``None == None`` and report ``runs_identical=True``."""
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout="no result line here\n")],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "schema validation" in report["error"]


def test_probe_result_duplicated_field_fails_closed(monkeypatch):
    """Item 3: the same field key appearing twice in one PROBE_RESULT
    line must fail closed."""
    line = _default_field_line() + "|hmv_rc=1"
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "schema validation" in report["error"]


def test_probe_result_duplicated_line_fails_closed(monkeypatch):
    """Item 3: TWO PROBE_RESULT lines in one run's output must fail
    closed, not silently pick the first/last one."""
    line = _default_field_line()
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line + "\n" + line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "schema validation" in report["error"]


def test_probe_result_missing_field_fails_closed(monkeypatch):
    """Item 3: a required field missing from the line must fail closed."""
    line = _default_field_line(hta_layers=None)
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "schema validation" in report["error"]


def test_probe_result_out_of_domain_field_fails_closed(monkeypatch):
    """Item 3: finite > total is outside the field's allowed domain and
    must fail closed."""
    line = _default_field_line(heat_finite="99")
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[_FakeResult(returncode=0, stdout=line)],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "schema validation" in report["error"]


def test_runs_that_differ_fail_closed(monkeypatch):
    """Item 3: two runs producing genuinely different results (here,
    different finite hmv_rc values) must fail closed."""
    line_a = _default_field_line(hmv_rc="1")
    line_b = _default_field_line(hmv_rc="0")
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[
            _FakeResult(returncode=0, stdout=line_a),
            _FakeResult(returncode=0, stdout=line_b),
        ],
    )
    report = probe.run_probe()
    assert report["stage"] != "complete"
    assert "did not produce identical results" in report["error"]


def test_runs_with_matching_nan_fields_are_identical_and_complete(monkeypatch):
    """Regression test for the exact bug this hardening pass fixed: two
    runs that both genuinely produce NaN in the same field must be
    considered IDENTICAL (NaN is never equal to itself under plain
    ``==``, so a naive dict comparison would wrongly report these two
    equal-in-substance runs as non-deterministic). ``solver_output_finite``
    must be ``False`` since not every value was finite."""
    line = _default_field_line(
        heat_finite="0", heat_any_nan="1", heat_first="nan", heat_last="nan",
    )
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[
            _FakeResult(returncode=0, stdout=line),
            _FakeResult(returncode=0, stdout=line),
        ],
    )
    report = probe.run_probe()
    assert report["stage"] == "complete"
    assert report["runs_identical"] is True
    assert report["solver_output_finite"] is False


def test_stdin_is_closed_for_binary_execution(monkeypatch):
    """Item 2 regression test: every binary-execution ``run_bounded`` call
    must explicitly request ``stdin=subprocess.DEVNULL`` - the pinned
    source's own diagnostic ``getchar()`` path (``FOF_DLL/BM_Util.cpp``)
    must never be able to inherit and block on this process's stdin."""
    line = _default_field_line()
    captured: List[Dict[str, Any]] = []
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[
            _FakeResult(returncode=0, stdout=line),
            _FakeResult(returncode=0, stdout=line),
        ],
        capture_run_kwargs=captured,
    )
    report = probe.run_probe()
    assert report["stage"] == "complete"
    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs.get("stdin") == subprocess.DEVNULL


def test_successful_all_finite_run_reports_solver_output_finite_true(monkeypatch):
    """Positive-path contract: an all-finite fabricated result reaches
    ``stage == "complete"`` with ``solver_output_finite=True``."""
    line = _default_field_line()
    _install_stub(
        monkeypatch,
        compile_result=_FakeResult(returncode=0),
        run_results=[
            _FakeResult(returncode=0, stdout=line),
            _FakeResult(returncode=0, stdout=line),
        ],
    )
    report = probe.run_probe()
    assert report["stage"] == "complete"
    assert report["solver_output_finite"] is True


def test_values_equal_treats_different_finite_values_as_unequal():
    """Companion to the NaN-equality test below: ordinary differing
    finite values must still compare unequal."""
    assert probe._values_equal({"x": 1.0}, {"x": 2.0}) is False


def test_values_equal_treats_nan_as_equal_to_nan():
    """Class (b) unit test for :func:`massman_fof_dll_probe._values_equal`
    directly (not through the full ``run_probe()`` orchestration): two
    dicts differing only by both holding a genuinely fresh (non-cached)
    NaN object in the same field must compare equal."""
    a = {"x": float("nan"), "y": 1.0}
    b = {"x": float(str(float("nan"))), "y": 1.0}
    assert a["x"] is not b["x"]
    assert probe._values_equal(a, b) is True


def test_verify_under_project_root_accepts_a_child_path():
    """Item 1 contract: a path actually under PROJECT_ROOT is accepted
    and returned resolved."""
    import os
    child = os.path.join(probe.PROJECT_ROOT, "tests", "cpp_parity_live")
    resolved = probe._verify_under_project_root(child)
    assert resolved.startswith(os.path.realpath(probe.PROJECT_ROOT))


def test_verify_under_project_root_rejects_an_outside_path():
    """Item 1 contract: a path outside PROJECT_ROOT (here, the real
    system temp directory) must be rejected, proving the authorized
    filesystem boundary is actually enforced, not merely documented."""
    import tempfile
    with pytest.raises(RuntimeError, match="outside PROJECT_ROOT"):
        probe._verify_under_project_root(tempfile.gettempdir())
