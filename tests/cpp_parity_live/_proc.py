#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_proc.py - Bounded subprocess execution with real, VERIFIED process-tree
termination on timeout.

``subprocess.run(timeout=...)`` only terminates the direct child process on
Windows; a Ninja build, a compiler invocation, or a pytest subprocess can
leave live descendants (cl.exe, link.exe, worker processes) behind after
the timeout fires, since Windows has no SIGKILL-to-process-group
equivalent by default. Every Phase 2 subprocess call
(``_harness_support.py``, ``_golden_manifest.py``,
``generate_phase2_goldens.py``, ``test_cpp_harness_contract.py``) goes
through :func:`run_bounded` instead, which enumerates the descendants that
exist at timeout and attempts to kill that snapshot plus the parent via
``psutil``. A process can race by spawning another descendant between the
snapshot and the kill, so this is a bounded best-effort teardown with
explicit post-kill accounting, not an operating-system-level containment
guarantee.

**The accurate guarantee** (round 4 correction item 5 — a prior round's
docstrings overclaimed unconditional success): every targeted process is
sent ``kill()``, and :func:`_kill_tree` then WAITS and RE-CHECKS each one.
A ``psutil.AccessDenied`` on the kill attempt, or a process still alive
after the wait, is recorded rather than silently swallowed — see
:class:`KillResult`. :func:`run_bounded` never claims "was killed"
unconditionally in its :class:`ProcTimeout` message when survivors or
access-denied PIDs remain; it reports them explicitly so a caller can
tell the difference between "confirmed fully torn down" and "kill
attempted, outcome uncertain for PID N". Verified by a real test that
spawns a child which itself spawns a grandchild and confirms both are
CONFIRMED GONE (not merely requested to die) afterward — see
``test_cpp_harness_contract.py::test_run_bounded_kills_the_full_process_tree_on_timeout``.

Function order: private helpers first, then public functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import subprocess
import time
from typing import Dict, List, Optional, Sequence

import psutil


class BoundedResult:
    """Result of one :func:`run_bounded` call."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class KillResult:
    """
    Outcome of one :func:`_kill_tree` call — the honest accounting a
    caller needs to tell "confirmed fully torn down" apart from "kill
    attempted, outcome uncertain".

    :ivar targeted: Every PID a kill was attempted on.
    :ivar access_denied: PIDs where ``kill()`` itself raised
        ``psutil.AccessDenied`` (the kill request was refused, not merely
        slow).
    :ivar survivors: PIDs that were still running after waiting up to
        the caller's timeout for the kill to take effect — confirmed via
        a re-check, not assumed.
    """

    def __init__(self, targeted: List[int], access_denied: List[int], survivors: List[int]):
        self.targeted = targeted
        self.access_denied = access_denied
        self.survivors = survivors

    @property
    def fully_terminated(self) -> bool:
        """``True`` only if every targeted PID is confirmed gone — no
        access-denied refusals and no surviving processes."""
        return not self.access_denied and not self.survivors


class ProcTimeout(RuntimeError):
    """Raised by :func:`run_bounded` when the process exceeded its
    timeout and its descendant-tree kill was attempted — see the message
    (and the module docstring) for whether termination was actually
    CONFIRMED for every PID or whether some are access-denied/surviving."""


def _kill_tree(pid: int, *, wait_s: float = 5.0) -> KillResult:
    """
    Kill *pid* and every live descendant process, then WAIT and RE-CHECK
    each targeted PID up to *wait_s* — never assumes success from the
    ``kill()`` call alone.

    :param pid: PID of the root process to kill (its children are found
        first, since the process may already be a zombie/exited by the
        time we get to killing it directly).
    :param wait_s: Seconds to wait for the kill to take effect before
        re-checking survivors.
    :return: A :class:`KillResult` recording every targeted PID, every
        PID that refused the kill (``AccessDenied``), and every PID
        confirmed still alive after the wait.
    """
    targeted: List[int] = []
    access_denied: List[int] = []
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return KillResult(targeted, access_denied, [])

    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []

    procs = children + [parent]
    for proc in procs:
        targeted.append(proc.pid)
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            access_denied.append(proc.pid)

    _gone, alive = psutil.wait_procs(procs, timeout=wait_s)
    # `alive` processes psutil itself confirms are still running after the
    # wait — re-verify with a direct is_running() pass too (belt-and-
    # braces: a process object can go stale between wait_procs' internal
    # check and here, but never the other way — pids_alive() below is the
    # single source of truth callers/tests should trust).
    survivors = pids_alive([p.pid for p in alive])
    return KillResult(targeted, access_denied, survivors)


def pids_alive(pids: Sequence[int]) -> List[int]:
    """Return the subset of *pids* that are still running (used by tests
    to prove a killed tree is actually gone, not merely requested to
    die)."""
    alive = []
    for pid in pids:
        try:
            if psutil.Process(pid).is_running():
                alive.append(pid)
        except psutil.NoSuchProcess:
            pass
    return alive


def run_bounded(
        args: Sequence[str],
        *,
        timeout: float,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
) -> BoundedResult:
    """
    Run *args* with a hard *timeout* and attempt to terminate the parent
    plus the descendant snapshot present when it fires (see the module
    docstring for the precise best-effort guarantee and race limitation).

    :param args: Argv to execute.
    :param timeout: Seconds before the process tree is killed.
    :param cwd: Working directory for the child.
    :param env: Environment for the child (``None`` inherits this
        process's environment, matching ``subprocess.run``'s default).
    :return: A :class:`BoundedResult` with ``returncode``/``stdout``/
        ``stderr``.
    :raises ProcTimeout: If *timeout* elapses. The exception message
        includes every PID that was targeted for the kill.
    """
    proc = subprocess.Popen(
        list(args), cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return BoundedResult(proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        result = _kill_tree(proc.pid)
        # Drain pipes so the now-dead process doesn't leave zombie file
        # descriptors; ignore whatever partial output exists.
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        if result.fully_terminated:
            status = f"pids targeted and CONFIRMED terminated: {result.targeted}"
        else:
            status = (
                f"pids targeted: {result.targeted}; "
                f"NOT confirmed terminated — access_denied: {result.access_denied}, "
                f"still alive after wait: {result.survivors}"
            )
        raise ProcTimeout(
            f"command timed out after {timeout}s; {status}: {list(args)!r}"
        )
