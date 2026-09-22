#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_xfail_audit_collector_plugin.py - a pytest plugin (not a test module
itself -- no ``test_*``/``Test*`` symbols) that records every xfail
outcome from a pytest run into a JSON file, read by
:mod:`test_xfail_disposition_audit`'s bounded subprocess. Loaded via
``-p _xfail_audit_collector_plugin`` with this file's own directory on
``PYTHONPATH``; writes to ``$XFAIL_COLLECTOR_OUT``.
"""
import json
import os

_OUT_PATH = os.environ.get("XFAIL_COLLECTOR_OUT")
_records = []


def pytest_runtest_logreport(report):
    """
    Record one xfail/xpass outcome, if this report carries one.

    :param report: The pytest ``TestReport`` for one test phase.
    :return: None.
    """
    if report.when != "call":
        return
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail is None:
        return
    _records.append({
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "reason": wasxfail,
    })


def pytest_sessionfinish(session, exitstatus):
    """
    Write every recorded xfail/xpass outcome to ``$XFAIL_COLLECTOR_OUT``.

    :param session: The pytest session (unused; required by the hook
        signature).
    :param exitstatus: The session's exit status (unused; required by
        the hook signature).
    :return: None.
    """
    if not _OUT_PATH:
        return
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(_records, f, indent=2)
