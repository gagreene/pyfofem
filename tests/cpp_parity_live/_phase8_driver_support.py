#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_phase8_driver_support.py - Invocation/validation helpers for
``phase8_parallel_driver.py`` (Phase 8 correction pass item 1).

Every Phase 8 test that needs to exercise ``run_fofem_emissions()``'s
``num_workers>1`` path goes through :func:`run_phase8_batch` (or, for
tests that deliberately expect a driver-side failure/malformed case,
:func:`invoke_phase8_driver` directly) - never a bare, in-process call
to ``run_fofem_emissions(num_workers>1)`` from a collected pytest node.
See ``phase8_parallel_driver.py``'s own module docstring for why.

**Correction pass (2026-09-11, responding to independent review) - the
driver protocol is now an EXACT, fully-enforced schema, not a
best-effort check.** :func:`invoke_phase8_driver` now requires: stdout
to contain EXACTLY one nonempty line (not merely "at least one marker
line" - an extra non-marker line, before or after, is now rejected);
stderr to be EXACTLY empty (documented choice, not a narrower allowlist
- confirmed by direct execution of every real scenario this suite
exercises, including a mixed valid/model-error 8-cell burnup batch under
``num_workers=2``, that the driver never writes to stderr in a normal
success or caught-exception path); ``payload["ok"]`` to be a genuine
JSON boolean (``isinstance(..., bool)``), never a truthy non-boolean
such as ``1``; a success payload's key set to be EXACTLY
:data:`_SUCCESS_KEYS` (``{"ok", "result"}``, no more, no fewer); an
error payload's key set to be EXACTLY :data:`_ERROR_KEYS`
(``{"ok", "error_type", "error_message"}``); and both
``error_type``/``error_message`` to be JSON strings. Malformed JSON now
raises a clear ``AssertionError`` (was: an uncaught ``JSONDecodeError``
with a less diagnostic message). This validation is deliberately
STRUCTURAL only - it does not require ``result`` to contain any
particular scientific field, since :func:`run_phase8_batch` supports
arbitrary ``run_fofem_emissions()`` call shapes (not just the default
63-field mode); callers that need the complete field inventory use
:func:`assert_known_percell_field_inventory` themselves, as
``test_phase8_serial_parallel_equivalence.py`` already does.

**Correction pass (2026-09-11, third round) - the error-path exit code is
now enforced EXACTLY, not merely "nonzero".** ``phase8_parallel_driver.py``
documents a precise contract: ``ok: true`` -> exit ``0``; ``ok: false``
(a caught Python/model error) -> exit exactly ``1``. The prior
``assert result.returncode != 0`` accepted ANY nonzero code alongside an
``ok: false`` payload - a code ``2``, a negative/signal-style code, or a
Windows crash-style code (e.g. ``0xC0000005``, cast to a large positive
or negative int depending on OS/Python return-code handling) would all
have been silently accepted as "consistent," even though none of those
match the documented protocol. :func:`invoke_phase8_driver` now asserts
``result.returncode == 1`` exactly for an ``ok: false`` payload.

Function order: private helpers first, then public functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._proc import BoundedResult, run_bounded

#: Absolute path to the non-collected driver script.
DRIVER_PATH = os.path.join(PROJECT_ROOT, "tests", "cpp_parity_live", "phase8_parallel_driver.py")

#: Default bound for a driver invocation - generous relative to the
#: ~0.04s measured cost of a small serial burnup call (see
#: ``test_phase8_operational_hardening.py``), so a real hang is still
#: caught quickly without flaking on normal load.
DEFAULT_TIMEOUT_S = 60.0

#: Stdout marker line prefix the driver emits exactly once per run.
_RESULT_MARKER = "PHASE8_DRIVER_RESULT "

#: The exact, complete key set for an ``ok: true`` payload - no more, no
#: fewer.
_SUCCESS_KEYS = frozenset({"ok", "result"})

#: The exact, complete key set for an ``ok: false`` payload - no more,
#: no fewer.
_ERROR_KEYS = frozenset({"ok", "error_type", "error_message"})

#: Substrings that indicate a native crash/fault-handler dump or an
#: uncaught Python traceback leaked to stderr - ANY of these anywhere in
#: a driver invocation's stderr is treated as a hard failure, never
#: dismissed merely because the process also happened to exit 0 (per
#: the Phase 8 correction pass's explicit instruction not to globally
#: disable pytest's faulthandler plugin or dismiss native exception
#: output).
NATIVE_CRASH_INDICATORS = (
    "Windows fatal exception",
    "access violation",
    "Segmentation fault",
    "AddressSanitizer",
    "Sanitizer:",
    "Traceback (most recent call last):",
)

#: The complete per-cell scientific/status output field set
#: ``run_fofem_emissions()`` returns for a default-mode
#: (``em_mode='default'``, ``soil_heating=False``) call - measured
#: directly by executing a real call and inspecting every key's shape
#: (Phase 8 correction pass item 7). Every one of these 63 keys is a
#: genuine per-cell array (shape ``(n,)``) with no aggregate/metadata
#: field mixed in at this configuration - confirmed by direct
#: inspection, not assumed. Used as a MAINTENANCE TRIPWIRE, not a
#: silent ceiling: :func:`assert_known_percell_field_inventory` asserts
#: EQUALITY (not merely "at least these"), so if ``run_fofem_emissions``
#: ever gains, renames, or loses a per-cell field, the assertion fails
#: and this constant must be consciously updated - a silent drift can
#: never pass unnoticed.
KNOWN_PERCELL_FIELDS = frozenset({
    "BraCon", "BraPos", "BraPre", "BurnupError", "BurnupLimitAdj",
    "CH4F", "CH4S", "CO2F", "CO2S", "COF", "COS",
    "DW100Con", "DW100Pos", "DW100Pre",
    "DW10Con", "DW10Pos", "DW10Pre",
    "DW1Con", "DW1Pos", "DW1Pre",
    "DW1kRotCon", "DW1kRotPos", "DW1kRotPre",
    "DW1kSndCon", "DW1kSndPos", "DW1kSndPre",
    "DufCon", "DufCon-Equ", "DufDepCon", "DufDepPos", "DufDepPre",
    "DufPos", "DufPre", "DufRed-Equ",
    "FlaCon", "FlaDur", "FolCon", "FolPos", "FolPre",
    "HerCon", "HerPos", "HerPre", "Herb-Equ",
    "Lit-Equ", "LitCon", "LitPos", "LitPre",
    "MSE", "MSE-Equ",
    "NOXF", "NOXS",
    "PM10F", "PM10S", "PM25F", "PM25S", "SO2F", "SO2S",
    "ShrCon", "ShrPos", "ShrPre", "Shrub-Equ",
    "SmoCon", "SmoDur",
})


def _assert_no_native_crash_indicators(stderr: str, *, context: str) -> None:
    """
    Fail loudly if *stderr* contains any :data:`NATIVE_CRASH_INDICATORS`
    substring.

    :param stderr: Captured subprocess stderr.
    :param context: Short label identifying the caller, used in the
        assertion message.
    :return: None.
    :raises AssertionError: If any indicator is present.
    """
    for indicator in NATIVE_CRASH_INDICATORS:
        assert indicator not in stderr, (
            f"{context}: stderr contains a native-crash/fault indicator "
            f"{indicator!r} - treated as a hard failure regardless of the "
            f"process's own return code.\nstderr:\n{stderr}"
        )


def assert_known_percell_field_inventory(result: Dict[str, np.ndarray], n: int) -> None:
    """
    Assert that :func:`percell_fields` for *result*/*n* is EXACTLY
    :data:`KNOWN_PERCELL_FIELDS` - the maintenance tripwire described on
    that constant's own docstring.

    :param result: A ``run_fofem_emissions()`` result dict (or the
        driver's deserialized equivalent).
    :param n: The batch size (number of cells).
    :return: None.
    :raises AssertionError: If the real per-cell field set has grown,
        shrunk, or been renamed relative to :data:`KNOWN_PERCELL_FIELDS`.
    """
    actual = set(percell_fields(result, n))
    missing = KNOWN_PERCELL_FIELDS - actual
    extra = actual - KNOWN_PERCELL_FIELDS
    assert not missing and not extra, (
        "run_fofem_emissions()'s per-cell field inventory has drifted from "
        f"KNOWN_PERCELL_FIELDS - missing: {sorted(missing)}, extra (new/renamed, "
        f"update KNOWN_PERCELL_FIELDS to include): {sorted(extra)}"
    )


def invoke_phase8_driver(
        raw_kwargs: dict,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Run :data:`DRIVER_PATH` once via :func:`~tests.cpp_parity_live._proc.
    run_bounded` (closed stdin, explicit timeout) and return its fully
    validated payload dict - which may be ``{"ok": True, "result": ...}``
    or ``{"ok": False, "error_type": ..., "error_message": ...}``; callers
    that require success should use :func:`run_phase8_batch` instead.

    Fails closed (raises ``AssertionError``) on: a native-crash/fault
    indicator anywhere in stderr; any nonempty stderr at all (the
    documented protocol - see the module docstring); stdout that is not
    EXACTLY one nonempty line; malformed JSON; a payload that is not a
    dict, is missing ``"ok"``, or has ``"ok"`` set to a non-boolean; a
    success (``ok: true``) payload whose key set is not EXACTLY
    ``{"ok", "result"}`` or whose ``"result"`` is not a JSON object; an
    error (``ok: false``) payload whose key set is not EXACTLY
    ``{"ok", "error_type", "error_message"}`` or whose
    ``error_type``/``error_message`` are not both strings; a return code
    other than exactly ``0`` for an ``ok: true`` payload; or a return
    code other than exactly ``1`` for an ``ok: false`` payload (the
    documented protocol - any other nonzero code, e.g. ``2``, a
    negative/signal-style code, or a Windows crash-style code, is never
    accepted as equivalent to ``1``).

    :param raw_kwargs: Forwarded verbatim as ``run_fofem_emissions()``'s
        keyword arguments inside the driver - list values become
        ``np.ndarray``.
    :param timeout: Seconds before the driver's process tree is killed.
    :param env: Optional environment override for the driver subprocess
        (``None`` inherits this process's environment).
    :return: The parsed, exact-schema-validated payload dict.
    :raises AssertionError: On any protocol/schema violation.
    :raises tests.cpp_parity_live._proc.ProcTimeout: If the driver does
        not finish within *timeout*.
    """
    config = json.dumps({"kwargs": raw_kwargs})
    result: BoundedResult = run_bounded(
        [sys.executable, DRIVER_PATH, config],
        timeout=timeout, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, env=env,
    )
    _assert_no_native_crash_indicators(result.stderr, context="phase8_parallel_driver.py")
    assert result.stderr == "", (
        "phase8_parallel_driver.py: stderr must be exactly empty (documented "
        "protocol - the driver never writes to stderr in a normal success or "
        f"caught-exception path); got {result.stderr!r}"
    )

    lines = result.stdout.splitlines()
    assert len(lines) == 1, (
        f"expected stdout to contain exactly one line, found {len(lines)} "
        f"(rc={result.returncode}); stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    line = lines[0]
    assert line.startswith(_RESULT_MARKER), (
        f"the single stdout line does not start with {_RESULT_MARKER!r}: {line!r}"
    )
    try:
        payload = json.loads(line[len(_RESULT_MARKER):])
    except json.JSONDecodeError as exc:
        raise AssertionError(f"malformed JSON in driver payload ({exc}): {line!r}") from exc

    assert isinstance(payload, dict), f"driver payload is not a JSON object: {payload!r}"
    assert "ok" in payload, f"driver payload missing 'ok': {payload!r}"
    assert isinstance(payload["ok"], bool), (
        f"'ok' must be a JSON boolean, got {payload['ok']!r} (type "
        f"{type(payload['ok']).__name__})"
    )

    if payload["ok"]:
        assert set(payload.keys()) == _SUCCESS_KEYS, (
            f"ok=True payload must have exactly the keys {sorted(_SUCCESS_KEYS)}, "
            f"got {sorted(payload.keys())}"
        )
        assert isinstance(payload["result"], dict), (
            f"'result' must be a JSON object, got {payload['result']!r}"
        )
        assert result.returncode == 0, (
            f"payload declares ok=True but the driver's own returncode was "
            f"{result.returncode}, not 0"
        )
    else:
        assert set(payload.keys()) == _ERROR_KEYS, (
            f"ok=False payload must have exactly the keys {sorted(_ERROR_KEYS)}, "
            f"got {sorted(payload.keys())}"
        )
        assert isinstance(payload["error_type"], str), (
            f"'error_type' must be a string, got {payload['error_type']!r}"
        )
        assert isinstance(payload["error_message"], str), (
            f"'error_message' must be a string, got {payload['error_message']!r}"
        )
        assert result.returncode == 1, (
            "payload declares ok=False but the driver's own returncode was "
            f"{result.returncode}, not the documented exact code 1 (a caught "
            "Python/model error must exit exactly 1 - a different nonzero code, "
            "e.g. 2, a negative/signal-style code, or a Windows crash-style code, "
            "is never accepted as equivalent)"
        )
    return payload


def percell_fields(result: Dict[str, np.ndarray], n: int) -> List[str]:
    """
    Return every key in *result* whose value is an array of shape
    ``(n,)`` - the complete per-cell scientific/status output inventory
    ``run_fofem_emissions()`` actually returned for this call, derived
    from the real result rather than a hand-maintained list, so a future
    field addition is picked up automatically (Phase 8 correction pass
    item 7's meta/contract requirement).

    :param result: A ``run_fofem_emissions()`` result dict (or the
        driver's deserialized equivalent), scalar values already
        converted to ``np.ndarray``.
    :param n: The batch size (number of cells).
    :return: Sorted list of per-cell field names.
    """
    return sorted(
        key for key, value in result.items()
        if np.asarray(value).shape == (n,)
    )


def run_phase8_batch(
        raw_kwargs: dict,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Run :data:`DRIVER_PATH` once, REQUIRE success (``ok: True``), and
    return the result dict with every value converted to ``np.ndarray``.

    :param raw_kwargs: Forwarded verbatim as ``run_fofem_emissions()``'s
        keyword arguments inside the driver.
    :param timeout: Seconds before the driver's process tree is killed.
    :param env: Optional environment override for the driver subprocess.
    :return: ``{field_name: np.ndarray, ...}`` for every field
        ``run_fofem_emissions()`` returned.
    :raises AssertionError: If the driver call did not succeed, or on
        any structural/schema violation (see
        :func:`invoke_phase8_driver`).
    """
    payload = invoke_phase8_driver(raw_kwargs, timeout=timeout, env=env)
    assert payload["ok"], (
        f"driver call failed: {payload.get('error_type')}: {payload.get('error_message')}"
    )
    return {key: np.asarray(value) for key, value in payload["result"].items()}
