#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase8_parallel_driver.py - Non-collected driver process that calls the
real, installed ``pyfofem.run_fofem_emissions()`` public API - including
its ``num_workers>1`` ``ProcessPoolExecutor`` path - from a PLAIN
``python`` subprocess, never from inside the pytest/``faulthandler``
process.

**Why this file exists (Phase 8 correction pass item 1).** Codex's
independent review found that collected pytest nodes calling
``run_fofem_emissions(num_workers=2)`` directly triggered a repeated
"Windows fatal exception: access violation" stderr dump. Root-caused
directly: the identical call made via a bare ``python -c "..."``
subprocess (no pytest involved at all) produces ZERO fault-handler
output and exits 0 every time; only invoking it INSIDE a process where
pytest's own built-in ``faulthandler`` plugin has already called
``faulthandler.enable()`` (installing a low-level Windows exception
handler) produces the dump. Running the real production call in a
plain ``python`` subprocess - this file - removes the trigger
structurally: there is no ``faulthandler`` handler installed in this
process to intercept and print whatever the underlying condition is.
See ``gate0/04-findings.md`` F-64 for the full reconciliation, including
what is and is not proven about the underlying condition itself.

This file is deliberately NOT named ``test_*.py`` so plain ``pytest``,
``--suite core``, and ``--suite full`` never collect or execute it
directly - it is only ever invoked as a subprocess, through
``tests.cpp_parity_live._phase8_driver_support.run_phase8_batch()``,
which wraps every invocation in
:func:`tests.cpp_parity_live._proc.run_bounded` with an explicit
timeout and a closed stdin.

**Protocol.** ``argv[1]`` is a single-line JSON object:
``{"kwargs": {...}}``, where any list-valued entry in ``kwargs`` is
converted to an ``np.ndarray`` before being forwarded to
``run_fofem_emissions()`` (scalar entries - ``bool``/``str``/``int``/
``float`` - are forwarded unchanged). On success, this driver prints
EXACTLY ONE line to stdout: ``PHASE8_DRIVER_RESULT <json>``, where
``<json>`` is ``{"ok": true, "result": {<field>: [<per-cell values>],
...}}`` (every ``ndarray`` result value converted via ``.tolist()``),
and exits ``0``. If ``run_fofem_emissions()`` raises an ordinary Python
exception (a deliberately-invalid config, not a crash), this driver
prints ``PHASE8_DRIVER_RESULT {"ok": false, "error_type": ...,
"error_message": ...}`` and exits ``1`` - still a clean, parseable
result, never a raw traceback. Nothing else is ever printed to stdout.

A genuine native crash (an access violation/segfault at the C level)
cannot be caught by the ``try``/``except`` below - that is precisely
the point: if one occurs, this process exits via an unhandled OS-level
fault (a nonzero/signal exit code with no
``PHASE8_DRIVER_RESULT`` line at all), and the caller
(``_phase8_driver_support.run_phase8_batch()``) treats a missing/
malformed result line as a hard failure - it must never be
interpreted as success or silently retried.

**Test-only hook.** If the environment variable
``PHASE8_DRIVER_SIMULATE_SLEEP_S`` is set, this driver sleeps that many
seconds BEFORE doing anything else - used by
``tests/unit/test_phase8_serial_parallel_equivalence.py`` and
``tests/unit/test_phase8_operational_hardening.py`` to exercise the
bounded-timeout path deterministically, without needing a real slow
production call. Never read outside those test modules.

**Protocol enforcement lives in the caller, not here** (Phase 8
correction pass item 2). This file's own contract is: exactly one
``PHASE8_DRIVER_RESULT <json>`` line on stdout and nothing else; stderr
left empty in every normal (success or caught-exception) path (no
``print``/``logging`` call in this file ever writes to stderr, and a
Python warning is not expected in any Phase 8 scenario - confirmed by
direct execution, not assumed); ``<json>`` is exactly
``{"ok": true, "result": {...}}`` or exactly
``{"ok": false, "error_type": "...", "error_message": "..."}``, never
extra/missing top-level keys. ``tests.cpp_parity_live.
_phase8_driver_support.invoke_phase8_driver()`` validates every one of
these properties and fails closed if any is violated - see that
module's own docstring for the exact enforced schema.

Function order: private helpers first, then ``main()``, per AGENTS.md
(this module has no public API beyond its CLI entry point).
"""
from __future__ import annotations

import json
import os
import sys


def _build_kwargs(raw_kwargs: dict) -> dict:
    """
    Convert every list-valued entry in *raw_kwargs* to an ``np.ndarray``,
    leaving scalar entries (``bool``/``str``/``int``/``float``/``None``)
    unchanged.

    :param raw_kwargs: The ``"kwargs"`` object decoded from the JSON
        config, as passed on the command line.
    :return: A dict suitable for ``run_fofem_emissions(**kwargs)``.
    """
    import numpy as np

    kwargs = {}
    for key, value in raw_kwargs.items():
        kwargs[key] = np.array(value) if isinstance(value, list) else value
    return kwargs


def _call_and_serialize(raw_kwargs: dict) -> dict:
    """
    Call the real ``run_fofem_emissions()`` and serialize its result (or
    any ordinary Python exception it raises) into a JSON-safe dict.

    :param raw_kwargs: The ``"kwargs"`` object decoded from the JSON
        config.
    :return: ``{"ok": True, "result": {...}}`` on success, or
        ``{"ok": False, "error_type": ..., "error_message": ...}`` if
        ``run_fofem_emissions()`` raised - deliberately broad
        ``except Exception``, since this driver's whole purpose is to
        report a caught model/config error back to the caller as clean
        data, never as a raw traceback (see the module docstring for why
        an actual native crash cannot reach this ``except`` clause at
        all).
    """
    import numpy as np
    from pyfofem import run_fofem_emissions

    try:
        kwargs = _build_kwargs(raw_kwargs)
        result = run_fofem_emissions(**kwargs)
        serialized = {key: np.asarray(value).tolist() for key, value in result.items()}
        return {"ok": True, "result": serialized}
    except Exception as exc:  # noqa: BLE001 - see docstring: reported as data, not raised
        return {"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}


def main(argv) -> int:
    """
    CLI entry point.

    :param argv: ``sys.argv`` (``argv[1]`` is the single-line JSON
        config).
    :return: ``0`` if ``run_fofem_emissions()`` succeeded, ``1`` if it
        raised an ordinary (caught) exception.
    """
    sleep_s = os.environ.get("PHASE8_DRIVER_SIMULATE_SLEEP_S")
    if sleep_s:
        import time
        time.sleep(float(sleep_s))

    config = json.loads(argv[1])
    payload = _call_and_serialize(config["kwargs"])
    print("PHASE8_DRIVER_RESULT " + json.dumps(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
