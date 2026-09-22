#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_xfail_disposition_audit.py - meta-test guaranteeing that the
comprehensive-suite xfail disposition audit
(``development/plans/gate0/08-xfail-disposition.csv``, produced by the
2026-09-18 xfail-disposition audit pass) exactly covers every currently
collected ``xfail`` node in the test files it audits.

This is NOT a snapshot count check: it re-runs the exact 8 test files the
CSV's own ``test_file`` column names, in a bounded, repository-local
subprocess, and asserts the resulting xfail node SET is identical (both
directions) to the CSV's ``node_id`` SET. A new xfail added to one of
these files without an audit row, or an audit row for a node that no
longer exists/no longer xfails, both fail this test loudly.

The CSV lives under ``development/plans/gate0/``, which is
LOCAL/GITIGNORED (matching ``07-branch-traceability.csv``'s own
established convention -- see ``docs/CODEBASE.md``/``AGENTS.md``): a
fresh clone or a packaging context without the accumulated local
development-plan directory will not have it. This test SKIPS (not
fails) when the file is absent, exactly like this repository's other
local-development-state-dependent checks.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.xfail_audit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _PROJECT_ROOT / "development" / "plans" / "gate0" / "08-xfail-disposition.csv"
_COLLECTOR_PLUGIN = Path(__file__).with_name("_xfail_audit_collector_plugin.py")

#: FIXED, independent of the CSV's own content -- the 2026-09-18 xfail-
#: disposition audit originally found these are the only files contributing
#: xfail nodes. The current accepted inventory is empty after the final
#: coefficient-precision corrections. The scan remains deliberately
#: hardcoded rather than derived from
#: the CSV's own ``test_file`` column: if a future edit deleted every
#: row for one of these files, deriving the scan scope FROM the CSV
#: would silently stop scanning that file too, making the audit
#: vacuously pass instead of catching the now-unaudited file. Keeping
#: this list independent means removing a whole file's rows is
#: detected as newly-missing nodes, not silently ignored. If a NEW
#: xfail is added to a file not in this list, this test will not see
#: it -- that gap is inherent to bounding the scan for speed; the
#: comprehensive-suite plan's own full-suite runs (plain/core/full)
#: remain the ultimate authority on the total xfail count.
_AUDITED_TEST_FILES = [
    "tests/cpp_parity_live/test_compare_cpp_python.py",
    "tests/unit/test_phase4_consumption_parity.py",
    "tests/unit/test_phase4_emissions_parity.py",
    "tests/unit/test_phase4_mortality_parity.py",
    "tests/unit/test_phase4_tree_structure_parity.py",
    "tests/unit/test_phase7_run_burnup_parity.py",
    "tests/unit/test_phase8_unit_system_contract.py",
]


def _load_csv_node_ids() -> set:
    """
    Read the audit CSV's ``node_id`` column.

    :returns: Set of every audited node id.
    """
    with open(_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        return {row["node_id"] for row in csv.DictReader(f)}


def test_audit_csv_exactly_covers_the_current_xfail_inventory(tmp_path):
    """
    Re-collect the real xfail outcomes from every test file the CSV
    audits, in a bounded subprocess, and assert the CSV's ``node_id``
    set is identical to the freshly-observed xfail node set -- neither
    missing a currently-xfailing node nor carrying a stale row for a
    node that no longer xfails.

    :param tmp_path: Pytest's own per-test temp directory (repository-
        external, matching this test's own bounded-subprocess needs).
    :return: None. Raises via ``assert`` on mismatch.
    """
    if not _CSV_PATH.exists():
        pytest.skip(
            f"{_CSV_PATH} is local/gitignored development-plan state "
            "(matches 07-branch-traceability.csv's own convention) and "
            "is not present in this checkout -- nothing to audit."
        )

    test_files = _AUDITED_TEST_FILES
    for rel in test_files:
        assert (_PROJECT_ROOT / rel).exists(), f"audited file missing: {rel}"

    with open(_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        csv_test_files = {row["test_file"] for row in csv.DictReader(f)}
    unaudited_files = csv_test_files - set(_AUDITED_TEST_FILES)
    assert not unaudited_files, (
        f"CSV references test file(s) outside _AUDITED_TEST_FILES's fixed "
        f"scan scope -- update that list too: {sorted(unaudited_files)}"
    )

    out_path = tmp_path / "xfail_report.json"
    env = {
        "XFAIL_COLLECTOR_OUT": str(out_path),
    }
    import os
    full_env = dict(os.environ)
    full_env.update(env)
    full_env["PYTHONPATH"] = str(_COLLECTOR_PLUGIN.parent) + os.pathsep + full_env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, "-m", "pytest", "-q",
        "-p", "_xfail_audit_collector_plugin",
        "--basetemp", str(tmp_path / "basetemp"),
    ] + test_files

    result = subprocess.run(
        cmd, cwd=str(_PROJECT_ROOT), env=full_env,
        capture_output=True, text=True, timeout=180,
    )
    assert out_path.exists(), (
        "collector plugin did not write a report -- subprocess output:\n"
        + result.stdout[-4000:] + "\n" + result.stderr[-2000:]
    )

    with open(out_path, "r", encoding="utf-8") as f:
        observed = json.load(f)
    observed_ids = {r["nodeid"] for r in observed}

    csv_ids = _load_csv_node_ids()

    missing_from_csv = observed_ids - csv_ids
    stale_in_csv = csv_ids - observed_ids

    assert not missing_from_csv, (
        f"{len(missing_from_csv)} currently-xfailing node(s) have no audit "
        f"row: {sorted(missing_from_csv)[:10]}"
    )
    assert not stale_in_csv, (
        f"{len(stale_in_csv)} audited node(s) no longer xfail (stale row, "
        f"needs re-audit): {sorted(stale_in_csv)[:10]}"
    )
