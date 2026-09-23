#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_serial_parallel_equivalence.py - Phase 8 item B: serial/
parallel equivalence for ``run_fofem_emissions(num_workers=...)``.

**Correction pass (2026-09-10, responding to independent review)**: this
module no longer calls ``run_fofem_emissions(num_workers>1)`` directly
from a collected pytest node. Doing so previously triggered a repeated
"Windows fatal exception: access violation" stderr dump from pytest's
own built-in ``faulthandler`` plugin (root-caused precisely: the
identical call via a bare ``python -c "..."`` subprocess, with no
pytest/faulthandler involved, produces ZERO fault-handler output and
exits 0 every time - a benign pytest-``faulthandler``-only dump, no
process crash or wrong result, see ``gate0/04-findings.md`` for the full
reconciliation). Every ``num_workers>1`` call in this module now goes
through ``tests.cpp_parity_live._driver_support.
run_phase8_batch()``/``invoke_phase8_driver()``, which spawn the real,
installed ``pyfofem`` in a PLAIN ``python`` subprocess
(``tests/cpp_parity_live/parallel_worker_driver.py`` - deliberately not
named ``test_*.py`` so it is never itself collected), bounded by
:func:`~tests.cpp_parity_live._proc.run_bounded` with an explicit
timeout and closed stdin, and fail closed on any native-crash indicator
in stderr, a malformed/missing result schema, or an unexpected return
code - see that module's own docstring for the full protocol.

**What this module proves, with direct executed evidence:**

- The COMPLETE per-cell scientific/status output set (all 63 fields
  ``run_fofem_emissions()`` returns at this call shape - see
  ``KNOWN_PERCELL_FIELDS``, not just a hand-picked subset) agrees
  field-by-field between ``num_workers=1`` (serial COMPUTATION - a
  single-process loop inside ``run_fofem_emissions()`` - run, like
  ``num_workers=2``, via the bounded driver subprocess, never in-process
  in the pytest process itself; see the item-7 correction note below)
  and ``num_workers=2`` (via the bounded driver, real
  ``ProcessPoolExecutor``) for an identical mixed valid/invalid batch,
  including non-consumption downstream fields (``PM10F``/``CO2F``/etc.)
  that are genuinely nonzero for valid cells - proving the comparison
  is discriminating, not vacuously passing on an all-zero field set.
- Output ordering is deterministic and follows the public contract:
  cell *i*'s result corresponds to cell *i*'s input regardless of
  ``num_workers``.
- Repeated ``num_workers=2`` driver invocations of the identical batch
  are deterministic.
- No child processes remain after a ``num_workers=2`` driver
  invocation completes, whether every cell succeeds or the batch mixes
  valid and model-error cells.
- The bounded driver's OWN safety plumbing is itself tested: a real
  timeout genuinely kills the full process tree with no survivors; a
  simulated malformed/missing result line and a simulated native-crash
  stderr indicator are both rejected by
  ``invoke_phase8_driver()``'s schema validation, using INJECTED
  ``BoundedResult`` stand-ins (never a real crash - see the module
  docstring's own "stop and report" instruction for what would happen
  if the driver genuinely crashed, which it does not, confirmed above).
- A hygiene check proves no COLLECTED Phase 8 test function calls
  ``run_fofem_emissions(...)`` with a non-1 ``num_workers`` directly -
  every such call is routed through the bounded driver.

**What this module does NOT attempt, and why (stop-and-report, not
silently skipped):** ``run_fofem_emissions()`` exposes no timeout
parameter for its internal ``ProcessPoolExecutor`` dispatch - there is
no production hook to exercise a genuine PRODUCTION-side pool timeout
without modifying production code, which this phase is not authorized
to do. The bounded driver's OWN timeout plumbing (a real safety
property of the TEST harness, not of production) is fully exercised
above via the ``PHASE8_DRIVER_SIMULATE_SLEEP_S`` test-only hook.

**Correction pass (2026-09-11, responding to independent review) - item
7 (wording).** Both ``num_workers=1`` and ``num_workers=2`` calls in
this module go through :func:`~tests.cpp_parity_live.
_driver_support.run_phase8_batch`, i.e. BOTH run inside the same
kind of bounded driver subprocess - the prior wording, "``num_workers=1``
(serial, in-process)", incorrectly implied the serial case ran directly
inside the pytest process. Corrected throughout this module: "serial" now
describes the COMPUTATION (a single-process loop inside
``run_fofem_emissions()`` itself, vs. a real multi-process
``ProcessPoolExecutor`` dispatch for ``num_workers>1``), never the
PROCESS PLACEMENT of the test call - both are placed identically, in a
driver subprocess, never in-process in pytest.

Also corrected this pass (items 2 and 3): the driver protocol is now an
exact, fully-enforced schema (see ``_driver_support.py``'s own
docstring) - malformed JSON, wrong-typed/extra/missing payload keys, a
non-boolean ``ok``, an extra stdout line, and any nonempty stderr are all
now rejected, not merely a subset; and the AST hygiene scan
(:func:`_direct_num_workers_gt1_calls`) now fails closed on ANY
non-literal-``1`` ``num_workers`` argument (a name, an expression, or a
literal other than ``1``), not only a literal constant other than ``1``.

**Correction pass (2026-09-11, third round) - item 1 (exact error exit
code).** ``invoke_phase8_driver()`` previously accepted ANY nonzero
return code for an ``ok: false`` payload (``result.returncode != 0``),
so a code ``2``, a negative/signal-style code, or a Windows crash-style
code would have been silently accepted as "consistent" even though the
documented protocol requires exactly ``1``. Now enforced exactly - see
``_driver_support.py``'s own docstring for the corrected
contract. 4 new regression tests added here:
``test_bounded_driver_accepts_ok_false_with_returncode_exactly_1``
(positive control - the correctly-formed error path must still be
accepted), ``test_bounded_driver_rejects_ok_false_with_returncode_0``,
``test_bounded_driver_rejects_ok_false_with_returncode_2``, and
``test_bounded_driver_rejects_ok_false_with_a_negative_or_windows_crash_style_returncode``
(uses ``-1073741819``, the signed 32-bit form of
``STATUS_ACCESS_VIOLATION``/``0xC0000005`` as Python's ``subprocess``
reports it on Windows). "``ok: true`` with any nonzero return code
remains rejected" is already proven by the pre-existing
``test_bounded_driver_rejects_a_returncode_ok_mismatch`` (unchanged;
re-verified against the edited validator).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import psutil
import pytest

from pyfofem.components.burnup import _FIRE_BOUNDS
from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._driver_support import (
    DRIVER_PATH,
    KNOWN_PERCELL_FIELDS,
    assert_known_percell_field_inventory,
    invoke_phase8_driver,
    percell_fields,
    run_phase8_batch,
)
from tests.cpp_parity_live._proc import BoundedResult, ProcTimeout, pids_alive

pytestmark = pytest.mark.multiprocessing

_INVALID_HFI = _FIRE_BOUNDS['fistart'][0] / 2.0
_VALID_HFI = 500.0

#: 8 cells: a genuine mix, invalid cells before/between/after valid ones,
#: distinct litter loads so a scrambled/contaminated result cannot pass by
#: coincidence.
_LITTER_8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_HFI_8 = [
    _INVALID_HFI, _VALID_HFI, _VALID_HFI, _INVALID_HFI,
    _VALID_HFI, _INVALID_HFI, _VALID_HFI, _VALID_HFI,
]
_INVALID_IDX_8 = {0, 3, 5}

#: The complete core-unit test-module surface (formerly grouped as
#: "Phase 8") this module's own hygiene check audits for direct,
#: unbounded ``num_workers>1`` calls.
_AUDITED_UNIT_TEST_MODULE_PATHS = (
    "tests/unit/test_array_isolation.py",
    "tests/unit/test_serial_parallel_equivalence.py",
    "tests/unit/test_mortality_facade.py",
    "tests/unit/test_moisture_regime_integration.py",
    "tests/unit/test_unit_system_contract.py",
    "tests/unit/test_runner_completeness.py",
    "tests/unit/test_operational_hardening.py",
)


def _batch_kwargs(litter, hfi, num_workers: int) -> dict:
    """
    Build the ``run_fofem_emissions()`` kwargs for the fixed 8-cell mixed
    valid/invalid batch (or a caller-supplied *litter*/*hfi* pair of the
    same length).

    :param litter: Per-cell pre-fire litter load (T/ac), list.
    :param hfi: Per-cell head fire intensity (kW/m), list.
    :param num_workers: Forwarded verbatim as ``num_workers``.
    :return: A JSON-serializable kwargs dict.
    """
    n = len(litter)
    zeros = [0.0] * n
    return dict(
        litter=litter, duff=zeros, duff_depth=zeros, herb=zeros, shrub=zeros,
        crown_foliage=zeros, crown_branch=zeros, pct_crown_burned=zeros,
        region=['InteriorWest'] * n,
        use_burnup=True, num_workers=num_workers, moisture_regime='Dry', units='Imperial',
        hfi=hfi,
    )


def _child_count() -> int:
    """
    Count live descendant processes of the current test process.

    :return: Number of descendant processes currently alive.
    """
    return len(psutil.Process().children(recursive=True))


def _direct_num_workers_gt1_calls(path: str) -> list:
    """
    AST-scan *path* for a direct ``run_fofem_emissions(...)`` call whose
    ``num_workers`` keyword argument is not PROVABLY the literal ``1`` -
    the bounded-driver architecture requires every such call to either
    omit ``num_workers`` entirely or pass exactly the literal ``1``.

    Fails closed on any other form: a literal constant other than ``1``
    (``num_workers=2``); a name reference (``num_workers=workers``); or
    any non-``Constant`` expression at all (``num_workers=1 + 1``,
    ``num_workers=int("2")``, ...) - since none of those can be proven
    equal to ``1`` by static inspection alone, each is treated as a
    potential unbounded direct call and rejected unless it is routed
    through the bounded driver instead (Phase 8 correction pass item 3).

    :param path: Repo-relative file path to scan.
    :return: List of human-readable finding strings (empty if clean).
    """
    full_path = os.path.join(PROJECT_ROOT, path)
    tree = ast.parse(open(full_path, encoding="utf-8").read(), filename=path)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "run_fofem_emissions":
            continue
        for kw in node.keywords:
            if kw.arg != "num_workers":
                continue
            is_literal_one = isinstance(kw.value, ast.Constant) and kw.value.value == 1
            if not is_literal_one:
                findings.append(
                    f"{path}:{node.lineno}: direct run_fofem_emissions(num_workers="
                    f"{ast.dump(kw.value)}) call is not provably literal 1 - outside "
                    "the bounded driver architecture"
                )
    return findings


def test_bounded_driver_accepts_ok_false_with_returncode_exactly_1(monkeypatch):
    """The documented, correctly-formed error path - ``ok: false`` paired
    with the exact documented exit code ``1`` - must be ACCEPTED, not
    merely tolerated as one of several nonzero codes."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", "error_message": "x"}'
        return BoundedResult(1, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    payload = invoke_phase8_driver({"litter": 1.0})
    assert payload == {"ok": False, "error_type": "ValueError", "error_message": "x"}


def test_bounded_driver_rejects_a_native_crash_indicator_in_stderr(monkeypatch):
    """``invoke_phase8_driver()`` must reject a run whose stderr contains
    a native-crash indicator, even if the process itself exited 0 -
    injected via a mocked :func:`run_bounded`, never a real crash."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(
            0, 'PHASE8_DRIVER_RESULT {"ok": true, "result": {}}',
            "Windows fatal exception: access violation\n",
        )

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="native-crash"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_a_returncode_ok_mismatch(monkeypatch):
    """A payload declaring ``ok: True`` while the process actually
    exited nonzero (or vice versa) must be rejected - a real, if
    unlikely, form of a malformed/inconsistent driver result."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(1, 'PHASE8_DRIVER_RESULT {"ok": true, "result": {}}', "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="ok=True"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_an_extra_non_marker_stdout_line(monkeypatch):
    """A second, non-marker line anywhere in stdout - before or after the
    real ``PHASE8_DRIVER_RESULT`` line - must be rejected: the documented
    protocol is exactly one stdout line, not merely 'at least one marker
    line, extra ignored'."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(
            0, 'unexpected stray line\nPHASE8_DRIVER_RESULT {"ok": true, "result": {}}', "",
        )

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly one"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_disallowed_nonempty_stderr(monkeypatch):
    """Nonempty stderr that contains no native-crash indicator must
    STILL be rejected - the documented protocol is stderr must be
    exactly empty, not merely 'free of known crash text'."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(
            0, 'PHASE8_DRIVER_RESULT {"ok": true, "result": {}}',
            "some/module.py:1: UserWarning: unrelated warning\n",
        )

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="stderr must be exactly empty"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_duplicate_result_lines(monkeypatch):
    """Two ``PHASE8_DRIVER_RESULT`` lines in one invocation must be
    rejected as malformed, never silently resolved by taking the first
    or last."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": true, "result": {}}'
        return BoundedResult(0, line + "\n" + line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly one"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_error_payload_with_extra_keys(monkeypatch):
    """An ``ok: false`` payload carrying an extra top-level key beyond
    the exact approved error set (``ok``/``error_type``/
    ``error_message``) must be rejected."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = ('PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", '
                '"error_message": "x", "extra": "unexpected"}')
        return BoundedResult(1, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly the keys"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_error_payload_with_non_string_error_fields(monkeypatch):
    """``error_type``/``error_message`` must each be a JSON string - a
    numeric or null value in either field must be rejected, not
    silently coerced."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", "error_message": null}'
        return BoundedResult(1, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="'error_message' must be a string"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_malformed_json_in_the_result_line(monkeypatch):
    """A ``PHASE8_DRIVER_RESULT`` line whose payload is not valid JSON
    must raise a clear ``AssertionError`` naming the malformed content,
    never an uncaught ``JSONDecodeError``."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(0, 'PHASE8_DRIVER_RESULT {not valid json', "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="malformed JSON"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_missing_result_line(monkeypatch):
    """A driver invocation that never printed a ``PHASE8_DRIVER_RESULT``
    line at all (the real symptom of an uncaught native crash) must be
    rejected, never silently treated as an empty success."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(1, "", "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly one"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_ok_as_a_non_boolean(monkeypatch):
    """``ok`` must be a genuine JSON boolean - a truthy non-boolean such
    as ``1`` must be rejected, not accepted merely because it is
    truthy."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(0, 'PHASE8_DRIVER_RESULT {"ok": 1, "result": {}}', "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="'ok' must be a JSON boolean"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_ok_false_with_a_negative_or_windows_crash_style_returncode(monkeypatch):
    """A representative Windows crash-style return code (the signed
    32-bit form of ``STATUS_ACCESS_VIOLATION``, ``0xC0000005``, as
    Python's ``subprocess`` reports it on Windows) paired with an
    ``ok: false`` payload must be rejected - a genuine native crash must
    never be masked as a clean, documented caught-exception exit."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", "error_message": "x"}'
        return BoundedResult(-1073741819, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="not the documented exact code 1"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_ok_false_with_returncode_0(monkeypatch):
    """An ``ok: false`` payload paired with returncode ``0`` must be
    rejected - a caught error can never legitimately co-occur with a
    clean-success exit code."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", "error_message": "x"}'
        return BoundedResult(0, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="not the documented exact code 1"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_ok_false_with_returncode_2(monkeypatch):
    """An ``ok: false`` payload paired with returncode ``2`` (a plausible
    but undocumented nonzero code) must be rejected - the protocol
    requires EXACTLY ``1``, not merely "nonzero"."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": false, "error_type": "ValueError", "error_message": "x"}'
        return BoundedResult(2, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="not the documented exact code 1"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_success_payload_with_extra_keys(monkeypatch):
    """An ``ok: true`` payload carrying an extra top-level key beyond
    the exact approved success set (``ok``/``result``) must be
    rejected."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        line = 'PHASE8_DRIVER_RESULT {"ok": true, "result": {}, "extra": "unexpected"}'
        return BoundedResult(0, line, "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly the keys"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_success_payload_with_missing_keys(monkeypatch):
    """An ``ok: true`` payload missing ``result`` entirely must be
    rejected, not treated as an implicit empty result."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(0, 'PHASE8_DRIVER_RESULT {"ok": true}', "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="exactly the keys"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_rejects_success_payload_with_non_dict_result(monkeypatch):
    """``result`` must be a JSON object - a list or scalar value must be
    rejected, not silently accepted as-is."""
    import tests.cpp_parity_live._driver_support as support

    def _fake_run_bounded(*args, **kwargs):
        return BoundedResult(0, 'PHASE8_DRIVER_RESULT {"ok": true, "result": [1, 2, 3]}', "")

    monkeypatch.setattr(support, "run_bounded", _fake_run_bounded)
    with pytest.raises(AssertionError, match="'result' must be a JSON object"):
        invoke_phase8_driver({"litter": 1.0})


def test_bounded_driver_timeout_kills_the_process_tree_with_no_survivors():
    """A driver invocation forced to sleep past its bound (via the
    test-only ``PHASE8_DRIVER_SIMULATE_SLEEP_S`` hook) must raise
    ``ProcTimeout`` reporting CONFIRMED termination, with zero PIDs
    still alive after a short grace wait - the real safety property the
    Phase 8 correction pass requires of the bounded parallel lane."""
    import re
    import time

    env = dict(os.environ)
    env["PHASE8_DRIVER_SIMULATE_SLEEP_S"] = "10"
    with pytest.raises(ProcTimeout) as exc_info:
        invoke_phase8_driver({"litter": 1.0}, timeout=1.5, env=env)
    message = str(exc_info.value)
    assert "CONFIRMED terminated" in message, message

    pid_list = re.search(r"\[([0-9,\s]*)\]", message).group(1)
    pids = [int(p) for p in re.findall(r"\d+", pid_list)]
    assert pids
    deadline = time.time() + 5
    remaining = pids
    while time.time() < deadline:
        remaining = pids_alive(pids)
        if not remaining:
            break
        time.sleep(0.2)
    assert remaining == [], f"process(es) survived the driver timeout kill: {remaining}"


def test_direct_num_workers_gt1_scan_accepts_literal_one_and_absent_rejects_names_and_expressions():
    """Meta-test for :func:`_direct_num_workers_gt1_calls` itself (Phase 8
    correction pass item 3): a literal ``num_workers=2`` and a NAME
    (``num_workers=workers``) and an EXPRESSION (``num_workers=1 + 1``)
    must all be rejected - none is provably literal ``1`` by static
    inspection - while an absent ``num_workers`` keyword and an explicit
    literal ``num_workers=1`` must both be accepted. Uses disposable
    repository-local scratch source files (never the system/user temp
    directory), never modifying any real collected Phase 8 module."""
    from tests.cpp_parity_live._scratch import scratch_tempdir

    snippets = {
        "literal_two.py": "run_fofem_emissions(litter=1.0, num_workers=2)\n",
        "a_name.py": "workers = 2\nrun_fofem_emissions(litter=1.0, num_workers=workers)\n",
        "an_expression.py": "run_fofem_emissions(litter=1.0, num_workers=1 + 1)\n",
        "a_call_expression.py": 'run_fofem_emissions(litter=1.0, num_workers=int("2"))\n',
        "literal_one.py": "run_fofem_emissions(litter=1.0, num_workers=1)\n",
        "absent.py": "run_fofem_emissions(litter=1.0)\n",
    }
    with scratch_tempdir("num_workers_ast_scan") as tmp_dir:
        for filename, source in snippets.items():
            with open(os.path.join(tmp_dir, filename), "w", encoding="utf-8") as f:
                f.write(source)

        rejected = {
            "literal_two.py", "a_name.py", "an_expression.py", "a_call_expression.py",
        }
        accepted = {"literal_one.py", "absent.py"}
        for filename in rejected:
            findings = _direct_num_workers_gt1_calls(os.path.relpath(os.path.join(tmp_dir, filename), PROJECT_ROOT))
            assert findings, f'{filename} should have been rejected but produced no findings'
        for filename in accepted:
            findings = _direct_num_workers_gt1_calls(os.path.relpath(os.path.join(tmp_dir, filename), PROJECT_ROOT))
            assert not findings, f'{filename} should have been accepted but was rejected: {findings}'


def test_known_percell_field_inventory_is_the_complete_set_for_this_call_shape():
    """The meta/contract check: for the standard 3-cell call shape this
    module's other tests use, the real per-cell field inventory must be
    EXACTLY :data:`KNOWN_PERCELL_FIELDS` - if ``run_fofem_emissions()``
    ever gains, loses, or renames a per-cell field, this test fails and
    forces a conscious update rather than a silent coverage gap."""
    result = run_phase8_batch(_batch_kwargs(_LITTER_8, _HFI_8, num_workers=1), timeout=60.0)
    assert_known_percell_field_inventory(result, len(_LITTER_8))
    assert len(KNOWN_PERCELL_FIELDS) == 63


def test_no_child_processes_remain_after_a_fully_valid_bounded_driver_run():
    """After a ``num_workers=2`` bounded-driver run where every cell
    succeeds, no descendant processes should remain alive."""
    before = _child_count()
    result = run_phase8_batch(
        _batch_kwargs([1.0, 2.0, 3.0, 4.0], [_VALID_HFI] * 4, num_workers=2), timeout=60.0,
    )
    assert np.all(result['BurnupError'] == 0)
    after = _child_count()
    assert after == before, f'expected no leftover descendant processes, before={before} after={after}'


def test_no_child_processes_remain_after_a_mixed_valid_invalid_bounded_driver_run():
    """After a ``num_workers=2`` bounded-driver run of the fixed mixed
    valid/invalid batch (some cells succeed, some carry an in-band
    ``BurnupError``), no descendant processes should remain alive - an
    isolated invalid cell must not leave the pool in a bad state."""
    before = _child_count()
    result = run_phase8_batch(_batch_kwargs(_LITTER_8, _HFI_8, num_workers=2), timeout=60.0)
    assert np.any(result['BurnupError'] != 0)
    assert np.any(result['BurnupError'] == 0)
    after = _child_count()
    assert after == before, f'expected no leftover descendant processes, before={before} after={after}'


def test_no_direct_num_workers_gt1_calls_in_any_collected_module():
    """Hygiene check: no collected Phase 8 test function may call
    ``run_fofem_emissions(...)`` with a literal ``num_workers`` other
    than ``1`` directly - every such call must be routed through the
    bounded driver architecture (this file's own ``_batch_kwargs()``
    helper only ever builds a kwargs DICT; the literal ``num_workers=2``
    it receives is data passed to the driver subprocess, not a direct
    in-process call - confirmed clean by this same scan, which greps
    for a ``run_fofem_emissions(`` CALL node specifically, not any
    occurrence of the substring ``num_workers``)."""
    all_findings = []
    for path in _AUDITED_UNIT_TEST_MODULE_PATHS:
        all_findings.extend(_direct_num_workers_gt1_calls(path))
    assert not all_findings, 'unbounded direct call(s) found:\n' + '\n'.join(all_findings)


def test_parallel_output_ordering_matches_input_order():
    """Cell *i*'s ``BurnupError`` under the ``num_workers=2`` bounded
    driver must match the OWN validity of cell *i*'s input (invalid at
    0/3/5, valid elsewhere) - proving the driver's underlying
    ``ProcessPoolExecutor.map()``'s documented order-preservation holds
    for this real call path."""
    result = run_phase8_batch(_batch_kwargs(_LITTER_8, _HFI_8, num_workers=2), timeout=60.0)
    for idx in range(len(_LITTER_8)):
        is_invalid = result['BurnupError'][idx] != 0
        assert is_invalid == (idx in _INVALID_IDX_8), (
            f'cell {idx} validity under num_workers=2 does not match its own input - '
            'output ordering may not follow input order'
        )


def test_repeated_bounded_driver_invocations_are_deterministic():
    """Running the identical mixed batch through the bounded driver
    twice with ``num_workers=2`` must produce bit-for-bit identical
    results both times, across the COMPLETE per-cell field set."""
    kwargs = _batch_kwargs(_LITTER_8, _HFI_8, num_workers=2)
    first = run_phase8_batch(kwargs, timeout=60.0)
    second = run_phase8_batch(kwargs, timeout=60.0)
    fields = percell_fields(first, len(_LITTER_8))
    assert fields == percell_fields(second, len(_LITTER_8))
    for key in fields:
        np.testing.assert_array_equal(
            first[key], second[key],
            err_msg=f'{key} differs between two identical num_workers=2 bounded-driver runs',
        )


def test_serial_and_bounded_parallel_batches_agree_on_every_percell_field():
    """The mixed valid/invalid 8-cell batch must produce field-by-field
    identical values across the COMPLETE per-cell output set (all 63
    ``KNOWN_PERCELL_FIELDS``, not a hand-picked subset) whether run with
    ``num_workers=1`` (serial COMPUTATION - a single-process loop inside
    ``run_fofem_emissions()`` - placed in the SAME kind of bounded driver
    subprocess as the ``num_workers=2`` case below, never in-process in
    pytest) or ``num_workers=2`` (via the bounded driver's real
    ``ProcessPoolExecutor``). Also proves the
    comparison is discriminating: at least one non-consumption
    downstream field (``PM10F``, an emissions output) is genuinely
    nonzero for a valid cell on both sides, so an all-zero field set
    could not produce a false pass."""
    serial = run_phase8_batch(_batch_kwargs(_LITTER_8, _HFI_8, num_workers=1), timeout=60.0)
    parallel = run_phase8_batch(_batch_kwargs(_LITTER_8, _HFI_8, num_workers=2), timeout=60.0)

    assert_known_percell_field_inventory(serial, len(_LITTER_8))
    assert_known_percell_field_inventory(parallel, len(_LITTER_8))

    valid_idx = sorted(set(range(len(_LITTER_8))) - _INVALID_IDX_8)[0]
    assert serial['PM10F'][valid_idx] > 0.0, 'PM10F should be nonzero for a valid cell (discriminating check)'

    for key in KNOWN_PERCELL_FIELDS:
        np.testing.assert_allclose(
            serial[key], parallel[key], atol=1e-9,
            err_msg=f'{key} differs between num_workers=1 and num_workers=2',
        )
