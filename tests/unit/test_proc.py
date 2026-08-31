#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_proc.py - Tests for the bounded subprocess helper
(``tests.cpp_parity_live._proc``), including the real process-tree kill on
timeout that ``subprocess.run(timeout=...)`` alone does not guarantee on
Windows.

No live C++ build needed — these tests only spawn plain Python child
processes.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import re
import sys
import time

import psutil
import pytest

from tests.cpp_parity_live._proc import (
    BoundedResult,
    KillResult,
    ProcTimeout,
    _kill_tree,
    pids_alive,
    run_bounded,
)

_GRANDCHILD_SCRIPT = (
    "import subprocess, sys, time; "
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
    "print(p.pid, flush=True); "
    "time.sleep(120)"
)


def _spawn_parent_and_grandchild():
    """Spawn (via run_bounded, with a tiny timeout so it's already killed
    by the time this returns) a child that itself spawns a grandchild.
    Returns the grandchild's PID, parsed from the child's one line of
    stdout before it was killed."""
    with pytest.raises(ProcTimeout) as exc_info:
        run_bounded(
            [sys.executable, "-c", _GRANDCHILD_SCRIPT],
            timeout=1.5,
        )
    return exc_info.value


def test_run_bounded_kills_the_full_process_tree_on_timeout():
    """The real proof: a child that spawns a grandchild, both still
    running when the timeout fires, both confirmed gone afterward — not
    merely "requested to die"."""
    proc_timeout = _spawn_parent_and_grandchild()
    message = str(proc_timeout)
    assert "CONFIRMED terminated" in message, (
        f"run_bounded must not claim confirmed termination unless every "
        f"targeted PID actually is confirmed gone: {message!r}"
    )
    # ProcTimeout's message embeds every PID run_bounded targeted for the
    # kill (parent + all children found via psutil at kill-time) as the
    # first bracketed list in the message.
    first_list = re.search(r"\[([0-9,\s]*)\]", message).group(1)
    pids = [int(p) for p in re.findall(r"\d+", first_list)]
    assert len(pids) >= 1  # at minimum the direct child

    # Give the OS a moment to finish tearing the processes down, then
    # confirm none of the targeted PIDs are still alive.
    deadline = time.time() + 5
    remaining = pids
    while time.time() < deadline:
        remaining = pids_alive(pids)
        if not remaining:
            break
        time.sleep(0.2)
    assert remaining == [], f"process(es) survived the tree kill: {remaining}"


def test_kill_tree_reports_survivor_when_a_process_refuses_to_die(monkeypatch):
    """A process that survives the kill+wait must be reported as a
    survivor in the KillResult, not silently treated as successfully
    torn down."""
    proc = psutil.Process()  # this very test process — never actually killed

    class _FakeParent:
        pid = proc.pid

        def children(self, recursive=True):
            return []

        def kill(self):
            pass  # pretend the kill was sent

        def is_running(self):
            return True  # this test's own process is genuinely still running

    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.Process",
        lambda pid: _FakeParent(),
    )
    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.wait_procs",
        lambda procs, timeout: ([], list(procs)),  # nothing "gone", all "alive"
    )

    result = _kill_tree(proc.pid, wait_s=0.1)
    assert not result.fully_terminated
    assert proc.pid in result.survivors
    assert result.access_denied == []


def test_kill_tree_reports_access_denied_when_kill_is_refused(monkeypatch):
    """A process whose kill() call itself raises AccessDenied must be
    recorded as access_denied, not silently ignored as if it were killed
    successfully."""
    class _FakeParent:
        pid = 999999

        def children(self, recursive=True):
            return []

        def kill(self):
            raise psutil.AccessDenied(pid=999999)

    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.Process",
        lambda pid: _FakeParent(),
    )
    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.wait_procs",
        lambda procs, timeout: (list(procs), []),
    )

    result = _kill_tree(999999, wait_s=0.1)
    assert not result.fully_terminated
    assert 999999 in result.access_denied


def test_kill_tree_fully_terminated_when_everything_confirmed_gone(monkeypatch):
    class _FakeParent:
        pid = 12345

        def children(self, recursive=True):
            return []

        def kill(self):
            pass

    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.Process",
        lambda pid: _FakeParent(),
    )
    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.psutil.wait_procs",
        lambda procs, timeout: (list(procs), []),
    )
    monkeypatch.setattr(
        "tests.cpp_parity_live._proc.pids_alive",
        lambda pids: [],
    )

    result = _kill_tree(12345, wait_s=0.1)
    assert result.fully_terminated
    assert result.survivors == []
    assert result.access_denied == []


def test_run_bounded_message_reports_survivors_when_not_fully_terminated(monkeypatch):
    """run_bounded's ProcTimeout message must explicitly surface survivors/
    access-denied PIDs when the kill was not fully confirmed — never
    silently claim success."""
    def _fake_kill_tree(pid, wait_s=5.0):
        return KillResult(targeted=[pid], access_denied=[], survivors=[pid])

    monkeypatch.setattr("tests.cpp_parity_live._proc._kill_tree", _fake_kill_tree)

    # A short-lived real process: run_bounded's timeout fires at 0.3s (the
    # fake _kill_tree above does NOT actually touch it), and it exits on
    # its own shortly after — so this test neither hangs nor leaves a real
    # orphan process behind despite the kill being faked.
    with pytest.raises(ProcTimeout) as exc_info:
        run_bounded([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.3)

    message = str(exc_info.value)
    assert "NOT confirmed terminated" in message
    assert "still alive after wait" in message


def test_run_bounded_returns_normally_on_success():
    result = run_bounded([sys.executable, "-c", "print('hello')"], timeout=10)
    assert isinstance(result, BoundedResult)
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_bounded_captures_nonzero_exit():
    result = run_bounded([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10)
    assert result.returncode == 3
