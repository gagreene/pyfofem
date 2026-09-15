#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase8_operational_hardening.py - Phase 8 item G: operational
hardening.

Covers the operational properties item G names that are NOT already
covered elsewhere in this repository, citing the existing coverage
rather than duplicating it:

- **Timeout behavior / child-process cleanup for bounded subprocesses**:
  already comprehensively covered by
  ``tests/unit/test_proc.py::test_run_bounded_kills_the_full_process_tree_on_timeout``
  and its sibling tests (real process-tree kill on timeout, survivor/
  access-denied reporting, stdin handling). ``run_fofem_emissions()``
  itself exposes no timeout parameter for its internal
  ``ProcessPoolExecutor`` dispatch (see
  ``test_phase8_serial_parallel_equivalence.py``'s own module docstring
  for the full explanation) - not re-derived here.
- **Child-process cleanup for ``ProcessPoolExecutor``**: covered in
  ``test_phase8_serial_parallel_equivalence.py``.
- **Deterministic repeated execution under a serial call**: proven here
  directly (the parallel case is covered in
  ``test_phase8_serial_parallel_equivalence.py``).
- **Existing Phase 2-7 golden/manifests remain valid and deterministic**:
  requires the live MSVC/CMake/Ninja build (``ensure_built()``), so it
  is exercised directly in the Phase 8 acceptance audit's own
  ``--verify-only`` gates, not duplicated as a new CORE test here (doing
  so would break CORE's no-toolchain-required guarantee) - the EXISTING
  ``tests/cpp_parity_live/test_generate_phase{2,4,5,6,7}_goldens.py``
  modules already provide this as real pytest nodes under
  ``--suite full``.

**Correction pass (2026-09-11, third round) - item 2 (debris
postconditions after a timeout).** ``_assert_no_debris_and_clean_pytest_run()``
previously took its post-run snapshots only after ``run_bounded()``
returned NORMALLY; a genuine hang (caught as ``ProcTimeout``) skipped
both post-run snapshots and every debris comparison entirely, silently
permitting debris created before the hang to escape detection -
contradicting the helper's own claim to check debris "regardless of the
subprocess's own outcome." The helper now catches ``ProcTimeout``
around the ``run_bounded()`` call, still takes both post-run snapshots
and runs the comparison unconditionally, and either re-raises the
original ``ProcTimeout`` unchanged (no debris found) or raises a
combined ``AssertionError ... from`` the original ``ProcTimeout``
(debris found alongside the timeout, both causes preserved via
exception chaining). Two new regression tests prove both outcomes using
a synthetic hanging pytest target (never a genuinely slow scientific
call): one that creates real scoped debris before hanging (debris
detected, ``ProcTimeout`` preserved as ``__cause__``, timed-out process
tree confirmed terminated, deliberately created debris removed after
the assertion) and one that hangs without creating any debris (the bare
``ProcTimeout`` propagates unchanged).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import re
import sys
import time
import uuid
import warnings

import numpy as np
import psutil
import pytest

from pyfofem import run_fofem_emissions
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import git_safe_directory_value
from tests.cpp_parity_live._phase8_driver_support import run_phase8_batch
from tests.cpp_parity_live._proc import BoundedResult, ProcTimeout, pids_alive, run_bounded
from tests.cpp_parity_live._scratch import scratch_tempdir

#: A generous, evidence-based bound for a small representative burnup
#: call. Measured wall-clock for this exact call (5 cells, serial burnup)
#: run through the bounded Phase 8 driver subprocess (interpreter
#: startup + import included) was ~1.6s on the development machine -
#: this bound gives roughly a 10x safety margin, not an arbitrary tight
#: threshold. (Correction pass 2026-09-11, item 5: this bound is now
#: the DRIVER SUBPROCESS's own timeout, not merely a post-hoc elapsed-
#: time check on an in-process call, so exceeding it actually terminates
#: the call - see
#: ``test_bounded_representative_burnup_call_exceeding_its_bound_produces_proctimeout_with_confirmed_cleanup``.)
_REPRESENTATIVE_RUN_BOUND_S = 15.0

#: Bound for every ``git``/``pytest`` subprocess this module spawns.
_SUBPROCESS_TIMEOUT_S = 60.0

#: The OTHER 6 Phase 8 test modules, used for the order-independence and
#: no-checkout-debris checks below. Deliberately EXCLUDES this very file
#: (``test_phase8_operational_hardening.py``) - including it would make
#: these two tests spawn a subprocess that re-runs this same test, which
#: would spawn another subprocess that re-runs it again, unboundedly
#: (a real bug caught during authoring, before it could hang the suite;
#: the excluded module's own tests are still fully exercised whenever the
#: real, non-recursive test run collects and executes this file directly).
_PHASE8_SIBLING_MODULE_PATHS = [
    "tests/unit/test_phase8_array_isolation.py",
    "tests/unit/test_phase8_serial_parallel_equivalence.py",
    "tests/unit/test_phase8_mortality_facade.py",
    "tests/unit/test_phase8_moisture_regime_integration.py",
    "tests/unit/test_phase8_unit_system_contract.py",
    "tests/unit/test_phase8_runner_completeness.py",
]


def _git(*args: str, cwd: str = PROJECT_ROOT) -> BoundedResult:
    """
    Run a bounded ``git`` subprocess, qualified with a per-command
    ``-c safe.directory=<forward-slash form of cwd>`` override so it
    succeeds regardless of the running account's global Git
    configuration or the repository's file ownership - never written to
    any config file. Mirrors the identical helper already established in
    ``test_phase4/5/6/7_golden_tracking.py``.

    :param args: ``git`` subcommand and arguments.
    :param cwd: Working directory for the subprocess.
    :returns: The :class:`~tests.cpp_parity_live._proc.BoundedResult`.
    """
    safe_directory = git_safe_directory_value(cwd)
    return run_bounded(
        ["git", "-c", f"safe.directory={safe_directory}", *args],
        timeout=_SUBPROCESS_TIMEOUT_S, cwd=cwd,
    )


def _assert_no_debris_and_clean_pytest_run(args: list, *, timeout: float) -> BoundedResult:
    """
    Run ``pytest`` (via a bounded subprocess) against *args*, checking
    for Phase-8-owned debris REGARDLESS of the subprocess's own outcome
    - the debris comparison always runs, even when the subprocess fails
    OR TIMES OUT - and only THEN requiring a clean (``0``) return code
    (skipped entirely on a timeout, which has no return code), so a
    failing sibling test, and now a HANGING sibling test, can never be
    masked as an accepted outcome (Phase 8 correction pass item 2: the
    prior version accepted ``returncode in (0, 1)``, which silently
    accepted a real sibling failure; Phase 8 THIRD correction pass item
    2: the prior version took its post-run snapshots only after
    ``run_bounded()`` returned NORMALLY, so a genuine hang - caught by
    :class:`~tests.cpp_parity_live._proc.ProcTimeout` - skipped both
    post-run snapshots and every debris comparison entirely, silently
    permitting pre-hang debris to escape detection).

    Two independent debris checks are performed, each scoped to what it
    can actually prove (item 8):

    - ``git status --porcelain`` before/after, covering Git-VISIBLE
      changes anywhere in the checkout (staged/unstaged/untracked) -
      this canNOT see a new file that matches a ``.gitignore`` rule.
    - A real filesystem inventory of every file under ``tests/unit/``
      and ``tests/cpp_parity_live/`` (the only directories Phase 8 test
      code lives in) before/after, which DOES include ignored files
      (e.g. a stray ``__pycache__``/``.pyc``, or a ``.pytest_cache``
      directory) - the actual scope this test can honestly claim to
      have inventoried, not the entire checkout.

    The subprocess itself runs with ``PYTHONDONTWRITEBYTECODE=1`` and
    ``-p no:cacheprovider`` (suppressing ``.pyc``/``.pytest_cache``
    creation at the source, rather than only detecting it after the
    fact) and an explicit, repository-local ``--basetemp`` under the
    gitignored scratch root - safe here because none of the arguments
    this function is ever called with is the pre-existing Phase 4
    regression test whose own precondition requires ``tmp_path`` to sit
    OUTSIDE any Git repository at all (that precondition is why the
    established convention avoids a repo-local basetemp for the FULL
    all-phase suite, not for this narrow, known argument list).

    **On a timeout** (``run_bounded()`` raises ``ProcTimeout``): the
    ``with scratch_tempdir(...)`` block's own ``__exit__`` still runs
    (ordinary context-manager semantics - the exception propagates
    THROUGH the ``with`` statement, which cleans up the basetemp
    directory before letting it continue upward), so basetemp cleanup
    is unaffected. The timeout is then CAUGHT here (not allowed to
    propagate past this function unexamined) so the post-run snapshots
    and debris comparison still run unconditionally. If debris is found
    alongside the timeout, both causes remain diagnostically visible: a
    combined ``AssertionError`` describing the debris is raised
    ``... from`` the original ``ProcTimeout`` (so the timeout survives
    as ``__cause__``, printed in the traceback, never silently
    dropped). If no debris is found, the original ``ProcTimeout`` is
    re-raised UNCHANGED - a timeout that leaves the checkout genuinely
    clean is reported exactly as it would have been before this
    function existed, not wrapped or reworded.

    :param args: Extra ``pytest`` CLI arguments (test file paths).
    :param timeout: Seconds before the subprocess's process tree is
        killed.
    :return: The :class:`~tests.cpp_parity_live._proc.BoundedResult`,
        for the caller's own additional assertions (e.g. inspecting
        ``stdout`` for a pass/xfail summary). Never returned on a
        timeout (an exception is always raised in that case instead).
    :raises AssertionError: If either debris check finds a difference
        (whether the subprocess finished normally, failed, or timed
        out), or if a normally-finished subprocess did not exit ``0``.
    :raises tests.cpp_parity_live._proc.ProcTimeout: If the subprocess
        timed out and left the checkout genuinely clean (no debris).
    """
    git_before = _git('status', '--porcelain').stdout
    files_before = _snapshot_scoped_files()

    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = None
    timeout_exc = None
    with scratch_tempdir('phase8_operational_hardening_basetemp') as basetemp:
        try:
            result = run_bounded(
                [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                 f'--basetemp={basetemp}'] + list(args),
                timeout=timeout, cwd=PROJECT_ROOT, env=env,
            )
        except ProcTimeout as exc:
            timeout_exc = exc

    git_after = _git('status', '--porcelain').stdout
    files_after = _snapshot_scoped_files()

    debris_messages = []
    if git_before != git_after:
        debris_messages.append(
            'running pytest changed the real checkout git status (staged/unstaged/'
            f'untracked):\nbefore:\n{git_before}\nafter:\n{git_after}'
        )
    if files_before != files_after:
        debris_messages.append(
            'running pytest left new/removed file(s) under tests/unit/ or '
            f'tests/cpp_parity_live/ (ignored-file-inclusive inventory): '
            f'new={sorted(files_after - files_before)}, '
            f'removed={sorted(files_before - files_after)}'
        )

    if timeout_exc is not None:
        if debris_messages:
            raise AssertionError(
                'the pytest subprocess timed out AND left debris (both causes '
                'preserved - see the chained ProcTimeout above for the timeout '
                'itself):\n' + '\n'.join(debris_messages)
            ) from timeout_exc
        raise timeout_exc

    assert not debris_messages, '\n'.join(debris_messages)
    assert result.returncode == 0, (
        f'the pytest subprocess itself failed (rc={result.returncode}) - a '
        'failing test must not be masked by the debris check.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
    return result


def _pass_summary(stdout: str) -> str:
    """
    Extract the pass/xfail/fail counts from pytest's final summary line
    (e.g. ``'44 passed, 2 xfailed'`` from ``'44 passed, 2 xfailed in
    18.77s'``), deliberately dropping the elapsed-time suffix, which
    varies run-to-run even with identical outcomes.

    :param stdout: Captured pytest stdout.
    :return: The summary line with any trailing ``' in <N>s'`` removed.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else ''
    return re.sub(r'\s+in\s+[\d.]+s(\s*\([^)]*\))?$', '', last)


def _snapshot_scoped_files() -> frozenset:
    """
    Walk ``tests/unit/`` and ``tests/cpp_parity_live/`` (the only
    directories Phase 8 test code lives in) and return every file path
    found, INCLUDING files a ``.gitignore`` rule would hide from ``git
    status`` (e.g. ``__pycache__``/``.pyc``, a stray
    ``.pytest_cache``) - the honest, actually-inventoried scope for the
    "no generated debris" claim (Phase 8 correction pass item 8: ``git
    status --porcelain`` alone cannot see an ignored file at all).

    :return: Frozenset of paths relative to :data:`PROJECT_ROOT`,
        forward-slash separated.
    """
    scoped_roots = (
        os.path.join(PROJECT_ROOT, 'tests', 'unit'),
        os.path.join(PROJECT_ROOT, 'tests', 'cpp_parity_live'),
    )
    found = set()
    for root_dir in scoped_roots:
        for root, _dirs, files in os.walk(root_dir):
            for name in files:
                full = os.path.join(root, name)
                found.add(os.path.relpath(full, PROJECT_ROOT).replace(os.sep, '/'))
    return frozenset(found)


def test_bounded_representative_burnup_call_completes_within_a_generous_measured_bound():
    """A small, representative serial burnup call (5 cells) must
    complete within a generous, evidence-based bound.

    **Correction pass (2026-09-11), item 5**: this now runs the call
    through :func:`~tests.cpp_parity_live._phase8_driver_support.
    run_phase8_batch` - the SAME bounded Phase 8 driver subprocess
    architecture the rest of this suite uses - with
    :data:`_REPRESENTATIVE_RUN_BOUND_S` as the driver's own timeout, with
    closed stdin and full process-tree cleanup on expiry. The prior
    version ran the call in-process and only checked elapsed time AFTER
    it returned, so it could neither terminate nor diagnose a genuine
    hang; see
    ``test_bounded_representative_burnup_call_exceeding_its_bound_produces_proctimeout_with_confirmed_cleanup``
    for the deterministic proof that exceeding this exact bound actually
    raises ``ProcTimeout`` with confirmed process-tree termination."""
    result = run_phase8_batch(
        dict(
            litter=[1.0, 2.0, 3.0, 4.0, 5.0], duff=[0.0] * 5, duff_depth=[0.0] * 5,
            herb=[0.0] * 5, shrub=[0.0] * 5, crown_foliage=[0.0] * 5, crown_branch=[0.0] * 5,
            pct_crown_burned=[0.0] * 5, region=['InteriorWest'] * 5,
            use_burnup=True, num_workers=1, moisture_regime='Dry', units='Imperial',
            hfi=[500.0] * 5,
        ),
        timeout=_REPRESENTATIVE_RUN_BOUND_S,
    )
    assert np.all(np.asarray(result['BurnupError']) == 0)


def test_bounded_representative_burnup_call_exceeding_its_bound_produces_proctimeout_with_confirmed_cleanup():
    """Deterministic timeout-path regression for item 5: forcing the
    driver to sleep past :data:`_REPRESENTATIVE_RUN_BOUND_S` (via the
    test-only ``PHASE8_DRIVER_SIMULATE_SLEEP_S`` hook, never a genuinely
    slow scientific call) must raise ``ProcTimeout`` reporting CONFIRMED
    termination, with zero PIDs still alive after a short grace wait -
    proving the representative-call bound above is a real, enforced
    subprocess timeout, not merely a value compared against after the
    fact."""
    env = dict(os.environ)
    env['PHASE8_DRIVER_SIMULATE_SLEEP_S'] = str(_REPRESENTATIVE_RUN_BOUND_S * 4)
    with pytest.raises(ProcTimeout) as exc_info:
        run_phase8_batch({'litter': 1.0}, timeout=1.5, env=env)
    message = str(exc_info.value)
    assert 'CONFIRMED terminated' in message, message

    pid_list = re.search(r'\[([0-9,\s]*)\]', message).group(1)
    pids = [int(p) for p in re.findall(r'\d+', pid_list)]
    assert pids
    deadline = time.time() + 5
    remaining = pids
    while time.time() < deadline:
        remaining = pids_alive(pids)
        if not remaining:
            break
        time.sleep(0.2)
    assert remaining == [], f'process(es) survived the driver timeout kill: {remaining}'


def test_git_helper_succeeds_against_the_real_project_under_forced_ownership_mistrust(monkeypatch):
    """Hostile Git ownership support (established in the Phase 7
    acceptance-gate correction and extended across every phase's own
    golden-tracking module) must hold for this module's own ``_git()``
    helper too: a read-only ``rev-parse`` against the real project
    repository must succeed even when Git's dubious-ownership check is
    forced via ``GIT_TEST_ASSUME_DIFFERENT_OWNER=1`` with blank,
    repository-local global/system config."""
    with scratch_tempdir('phase8_operational_hardening_ownership') as base:
        blank_global = os.path.join(base, 'blank_global.gitconfig')
        blank_system = os.path.join(base, 'blank_system.gitconfig')
        open(blank_global, 'w', encoding='utf-8').close()
        open(blank_system, 'w', encoding='utf-8').close()
        monkeypatch.setenv('GIT_TEST_ASSUME_DIFFERENT_OWNER', '1')
        monkeypatch.setenv('GIT_CONFIG_GLOBAL', blank_global)
        monkeypatch.setenv('GIT_CONFIG_SYSTEM', blank_system)

        result = _git('rev-parse', 'HEAD')
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.strip()) == 40


def test_no_checkout_local_generated_debris_check_rejects_a_failing_run_while_still_checking_cleanup(monkeypatch):
    """Regression proof for Phase 8 correction pass item 2: a
    deliberately-failing synthetic test must be REJECTED by
    ``_assert_no_debris_and_clean_pytest_run`` (never silently accepted
    the way the prior ``returncode in (0, 1)`` version would have),
    while the debris comparison itself is still proven to have actually
    executed - via a call-count spy on ``_snapshot_scoped_files``,
    which must fire exactly twice (before AND after) even though the
    subprocess fails."""
    import tests.unit.test_phase8_operational_hardening as this_module

    call_count = {'n': 0}
    real_snapshot = this_module._snapshot_scoped_files

    def _spy():
        call_count['n'] += 1
        return real_snapshot()

    monkeypatch.setattr(this_module, '_snapshot_scoped_files', _spy)

    with scratch_tempdir('phase8_operational_hardening_synthetic_failure') as synth_dir:
        synth_path = os.path.join(synth_dir, 'test_synthetic_failure.py')
        with open(synth_path, 'w', encoding='utf-8') as fh:
            fh.write(
                'def test_deliberately_fails():\n'
                '    assert False, "synthetic failure for Phase 8 regression coverage"\n'
            )
        with pytest.raises(AssertionError, match='pytest subprocess itself failed'):
            _assert_no_debris_and_clean_pytest_run([synth_path], timeout=60.0)

    assert call_count['n'] == 2, (
        f'expected the debris snapshot to be taken exactly twice (before and '
        f'after) even though the subprocess failed, got {call_count["n"]}'
    )


def test_no_checkout_local_generated_debris_check_reports_debris_and_the_original_proctimeout_together_when_a_hanging_run_leaves_debris():
    """Regression proof for Phase 8 THIRD correction pass item 2: a
    synthetic test that creates a real, scoped debris file and then
    HANGS (never a genuinely slow scientific call) must be caught by the
    real bounded timeout path, and - unlike the prior version, which
    would have skipped both post-run snapshots and every debris
    comparison entirely on a timeout - the debris must still be
    DETECTED, with the original ``ProcTimeout`` preserved as the
    combined ``AssertionError``'s ``__cause__`` (never silently
    dropped), and the timed-out process tree confirmed fully terminated.
    The deliberately created debris file is removed after the assertion
    regardless of outcome, so this regression test itself leaves nothing
    behind."""
    marker_name = f'.phase8_op3_debris_probe_{uuid.uuid4().hex}.tmp'
    marker_path = os.path.join(PROJECT_ROOT, 'tests', 'unit', marker_name)
    try:
        with scratch_tempdir('phase8_operational_hardening_timeout_with_debris') as synth_dir:
            synth_path = os.path.join(synth_dir, 'test_synthetic_hang_with_debris.py')
            with open(synth_path, 'w', encoding='utf-8') as fh:
                fh.write(
                    'import time\n'
                    f'open({marker_path!r}, "w").close()\n'
                    'def test_hangs_after_creating_debris():\n'
                    '    time.sleep(9999)\n'
                )
            with pytest.raises(AssertionError) as exc_info:
                _assert_no_debris_and_clean_pytest_run([synth_path], timeout=20.0)

        message = str(exc_info.value)
        assert 'timed out AND left debris' in message, message
        assert marker_name in message, (
            f'expected the debris message to name the new file {marker_name!r}: {message}'
        )
        assert isinstance(exc_info.value.__cause__, ProcTimeout), (
            f'expected the original ProcTimeout to be preserved as __cause__, got '
            f'{exc_info.value.__cause__!r}'
        )
        timeout_message = str(exc_info.value.__cause__)
        assert 'CONFIRMED terminated' in timeout_message, timeout_message

        pid_list = re.search(r'\[([0-9,\s]*)\]', timeout_message).group(1)
        pids = [int(p) for p in re.findall(r'\d+', pid_list)]
        assert pids
        deadline = time.time() + 5
        remaining = pids
        while time.time() < deadline:
            remaining = pids_alive(pids)
            if not remaining:
                break
            time.sleep(0.2)
        assert remaining == [], f'process(es) survived the timeout kill: {remaining}'
    finally:
        if os.path.exists(marker_path):
            os.remove(marker_path)
        assert not os.path.exists(marker_path)


def test_no_checkout_local_generated_debris_check_reraises_the_original_proctimeout_unchanged_when_a_hanging_run_leaves_no_debris():
    """Companion to the debris-plus-timeout test above: a synthetic test
    that HANGS but creates no debris at all must surface the ORIGINAL
    ``ProcTimeout`` unchanged (never wrapped in an ``AssertionError``) -
    proving the debris comparison genuinely runs and genuinely finds
    nothing, rather than the timeout being reported some other way."""
    with scratch_tempdir('phase8_operational_hardening_timeout_no_debris') as synth_dir:
        synth_path = os.path.join(synth_dir, 'test_synthetic_hang_no_debris.py')
        with open(synth_path, 'w', encoding='utf-8') as fh:
            fh.write(
                'import time\n'
                'def test_hangs_without_creating_debris():\n'
                '    time.sleep(9999)\n'
            )
        with pytest.raises(ProcTimeout) as exc_info:
            _assert_no_debris_and_clean_pytest_run([synth_path], timeout=2.0)
    assert 'CONFIRMED terminated' in str(exc_info.value), str(exc_info.value)


def test_phase8_modules_create_no_checkout_local_generated_debris():
    """Running the 6 sibling Phase 8 test modules must leave the real
    checkout's Git-visible status UNCHANGED and create no new/removed
    file (tracked or ignored) under ``tests/unit/``/
    ``tests/cpp_parity_live/`` - and the subprocess itself must have
    actually SUCCEEDED (a failing sibling test can never be masked as
    an accepted outcome here - see the regression test above proving
    this rejection actually works)."""
    _assert_no_debris_and_clean_pytest_run(_PHASE8_SIBLING_MODULE_PATHS, timeout=180.0)


def test_phase8_modules_pass_in_both_forward_and_reversed_file_order():
    """Running the 6 sibling Phase 8 test modules in forward order and then in
    REVERSED order must both produce the SAME pass/xfail summary line -
    proving no test in this new module set depends on execution order or
    leaks state into a sibling module.

    **Correction pass (2026-09-11), item 4**: both runs now go through
    the SAME fail-closed ``_assert_no_debris_and_clean_pytest_run``
    infrastructure every other subprocess pytest call in this module
    uses (managed repository-local ``--basetemp``, ``-p
    no:cacheprovider``, ``PYTHONDONTWRITEBYTECODE=1``, and a
    before/after Git-visible + ignored-file debris check) - the prior
    version called :func:`~tests.cpp_parity_live._proc.run_bounded`
    directly, twice, with none of those protections."""
    forward = _assert_no_debris_and_clean_pytest_run(_PHASE8_SIBLING_MODULE_PATHS, timeout=180.0)
    reversed_result = _assert_no_debris_and_clean_pytest_run(
        list(reversed(_PHASE8_SIBLING_MODULE_PATHS)), timeout=180.0,
    )
    assert _pass_summary(forward.stdout) == _pass_summary(reversed_result.stdout), (
        f'forward: {_pass_summary(forward.stdout)!r} vs reversed: {_pass_summary(reversed_result.stdout)!r}'
    )


def test_repeated_serial_burnup_calls_are_deterministic():
    """Two identical serial (``num_workers=1``) calls must produce
    bit-for-bit identical results - the parallel case is covered
    separately in ``test_phase8_serial_parallel_equivalence.py``."""
    litter = np.array([1.0, 2.0, 3.0])
    zeros = np.zeros(3)
    kwargs = dict(
        litter=litter, duff=zeros, duff_depth=zeros, herb=zeros, shrub=zeros,
        crown_foliage=zeros, crown_branch=zeros, pct_crown_burned=zeros,
        region=np.array(['InteriorWest'] * 3),
        use_burnup=True, num_workers=1, moisture_regime='Dry', units='Imperial',
        hfi=np.full(3, 500.0),
    )
    first = run_fofem_emissions(**kwargs)
    second = run_fofem_emissions(**kwargs)
    for key in ('LitCon', 'FlaDur', 'SmoDur', 'BurnupError'):
        np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(second[key]))


def test_scratch_tempdir_leaves_no_file_handle_or_directory_residue():
    """Repository-local scratch infrastructure (reused, not duplicated,
    per the Phase 8 constraint to use it for new helpers) must clean up
    completely on both a normal exit and an exception - a file opened
    and closed inside the scratch context must not survive it, and the
    scratch directory itself must not exist afterward."""
    captured_path = None
    with scratch_tempdir('phase8_operational_hardening_handles') as scratch_dir:
        captured_path = scratch_dir
        file_path = os.path.join(scratch_dir, 'probe.txt')
        with open(file_path, 'w', encoding='utf-8') as fh:
            fh.write('phase8 handle probe')
        assert os.path.isfile(file_path)
    assert not os.path.exists(captured_path), 'scratch directory was not cleaned up on normal exit'

    captured_path_2 = None
    with pytest.raises(RuntimeError):
        with scratch_tempdir('phase8_operational_hardening_handles') as scratch_dir_2:
            captured_path_2 = scratch_dir_2
            raise RuntimeError('simulated failure inside the scratch context')
    assert not os.path.exists(captured_path_2), 'scratch directory was not cleaned up after an exception'


def test_warning_behavior_is_silent_for_a_nominal_burnup_call():
    """A nominal, fully-valid burnup call must produce ZERO Python
    warnings - establishing a clean baseline so a future regression that
    introduces spurious warnings (e.g. a NumPy ``RuntimeWarning`` from an
    unguarded divide) is detectable. (Known, pre-existing warning-
    producing cases - e.g. F-29's ``calc_crown_length_vol_scorched``
    zero-crown-depth division - are pinned separately in
    ``tests/unit/test_tree_flame_source_relations.py`` and not
    re-derived here.)"""
    litter = np.array([1.0, 2.0])
    zeros = np.zeros(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run_fofem_emissions(
            litter=litter, duff=zeros, duff_depth=zeros, herb=zeros, shrub=zeros,
            crown_foliage=zeros, crown_branch=zeros, pct_crown_burned=zeros,
            region=np.array(['InteriorWest'] * 2),
            use_burnup=True, num_workers=1, moisture_regime='Dry', units='Imperial',
            hfi=np.full(2, 500.0),
        )
    assert not caught, f'expected zero warnings for a nominal call, got: {[str(w.message) for w in caught]}'
