#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_xfail_disposition_audit.py - meta-test guaranteeing that the
comprehensive-suite xfail disposition audit
(``development/plans/gate0/08-xfail-disposition.csv``, produced by the
2026-09-18 xfail-disposition audit pass) exactly covers every currently
collected ``xfail`` node in the test files it audits.

This is NOT a snapshot count check: it re-runs the fixed inventory of 7
test files that contributed xfail nodes during the audit, in a bounded,
repository-local subprocess, and asserts the resulting xfail node SET is
identical (both directions) to the CSV's ``node_id`` SET. A new xfail added
to one of these files without an audit row, or an audit row for a node that
no longer exists/no longer xfails, both fail this test loudly.

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
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

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
    "tests/unit/cpp/test_consumption_parity.py",
    "tests/unit/cpp/test_emissions_parity.py",
    "tests/unit/cpp/test_mortality_parity.py",
    "tests/unit/cpp/test_tree_structure_parity.py",
    "tests/unit/cpp/test_burnup_extended_parity.py",
    "tests/unit/test_unit_system_contract.py",
]


def _collect_xfail_records(test_files: Sequence[str], run_root: Path) -> list[dict]:
    """
    Run the xfail collector against *test_files* and return its records.

    :param test_files: Test paths passed directly to the pytest subprocess.
    :param run_root: Existing or creatable directory for the report and
        subprocess basetemp.
    :returns: Parsed collector records.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    out_path = run_root / "xfail_report.json"
    full_env = dict(os.environ)
    full_env["XFAIL_COLLECTOR_OUT"] = str(out_path)
    full_env["PYTHONPATH"] = (
        str(_COLLECTOR_PLUGIN.parent)
        + os.pathsep
        + full_env.get("PYTHONPATH", "")
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "_xfail_audit_collector_plugin",
        "--basetemp",
        str(run_root / "basetemp"),
        *test_files,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "xfail collector subprocess failed; its report cannot be trusted:\n"
        + result.stdout[-4000:]
        + "\n"
        + result.stderr[-2000:]
    )
    assert out_path.exists(), (
        "collector plugin did not write a report -- subprocess output:\n"
        + result.stdout[-4000:]
        + "\n"
        + result.stderr[-2000:]
    )
    with open(out_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_node_ids() -> set:
    """
    Read the audit CSV's ``node_id`` column.

    :returns: Set of every audited node id.
    """
    with open(_CSV_PATH, "r", encoding="utf-8", newline="") as handle:
        return {row["node_id"] for row in csv.DictReader(handle)}


def _write_synthetic_test(tmp_path: Path, source: str) -> Path:
    """
    Write one temporary pytest module used to exercise collector failures.

    :param tmp_path: Directory in which to create the test module.
    :param source: Complete Python source for the synthetic test.
    :returns: Path to the created test module.
    """
    path = tmp_path / "test_synthetic_xfail_audit.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_audit_csv_exactly_covers_the_current_xfail_inventory(tmp_path):
    """
    Re-collect the real xfail outcomes and require exact CSV coverage.

    :param tmp_path: Pytest's own per-test temp directory (repository-
        external, matching this test's own bounded-subprocess needs).
    :returns: None. Raises via ``assert`` on mismatch.
    """
    if not _CSV_PATH.exists():
        pytest.skip(
            f"{_CSV_PATH} is local/gitignored development-plan state "
            "(matches 07-branch-traceability.csv's own convention) and "
            "is not present in this checkout -- nothing to audit."
        )

    for rel in _AUDITED_TEST_FILES:
        assert (_PROJECT_ROOT / rel).exists(), f"audited file missing: {rel}"

    with open(_CSV_PATH, "r", encoding="utf-8", newline="") as handle:
        csv_test_files = {row["test_file"] for row in csv.DictReader(handle)}
    unaudited_files = csv_test_files - set(_AUDITED_TEST_FILES)
    assert not unaudited_files, (
        f"CSV references test file(s) outside _AUDITED_TEST_FILES's fixed "
        f"scan scope -- update that list too: {sorted(unaudited_files)}"
    )

    observed = _collect_xfail_records(_AUDITED_TEST_FILES, tmp_path / "audit")
    observed_ids = {record["nodeid"] for record in observed}
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


def test_collector_subprocess_rejects_ordinary_failure(tmp_path):
    """
    Prove an ordinary child-test failure cannot masquerade as an empty audit.

    :param tmp_path: Pytest temporary directory for the synthetic module.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    path = _write_synthetic_test(
        tmp_path,
        "def test_failure():\n"
        "    assert False\n",
    )
    with pytest.raises(AssertionError, match="xfail collector subprocess failed"):
        _collect_xfail_records([str(path)], tmp_path / "ordinary-failure")


def test_collector_subprocess_rejects_strict_xpass(tmp_path):
    """
    Prove a strict XPASS cannot masquerade as an empty legitimate audit.

    :param tmp_path: Pytest temporary directory for the synthetic module.
    :returns: None. Raises via ``assert`` on mismatch.
    """
    path = _write_synthetic_test(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.xfail(strict=True, reason='synthetic strict XPASS')\n"
        "def test_strict_xpass():\n"
        "    assert True\n",
    )
    with pytest.raises(AssertionError, match="xfail collector subprocess failed"):
        _collect_xfail_records([str(path)], tmp_path / "strict-xpass")
