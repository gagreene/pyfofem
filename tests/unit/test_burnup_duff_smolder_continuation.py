#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_burnup_duff_smolder_continuation.py - F-62 partial-resolution
regression coverage (gate0/04-findings.md F-62; the ``hot-amb-duff``/
``SmoDur`` combination of
``tests/unit/test_phase7_run_burnup_parity.py::
test_flame_smolder_split_should_match_cpp_for_the_affected_scenarios``).

**Assertion class:** (b) source-relation regression. This is a
synthetic, self-contained scenario -- not tied to any committed golden
CSV -- proving the specific mechanism the fix addresses: pinned C++
``bur_brn.cpp``'s main time-stepping loop (``:377-380``) does not stop
once wood/litter/herb-shrub fire intensity ``fi`` drops to its minimum
``fimin`` if duff still has unconsumed mass (its own running
``d_Duf_Tot`` total, decremented every timestep by ``Duff_CPTS()``); it
only breaks once BOTH conditions hold. ``burnup()`` (``burnup.py``) now
implements the same gate via ``tis < tdf`` (this function's own
duff-burn-duration, matching C++'s ``d_tdf`` -- both reach the
"duff exhausted" instant identically, since duff burns at a constant
rate over ``[0, tdf)`` in both implementations).

Makes no claim about the OTHER 5 F-62 `(case_id, field)` combinations
(``FlaCon``/``SmoCon`` for both scenarios; ``SmoDur`` for
``long-igtime``), which remain genuinely divergent and unexplained --
see the finding's own 2026-09-21 entry for why this fix does not
address them.
"""
from __future__ import annotations

from pyfofem import run_burnup
from pyfofem.components.burnup import _duff_burn

pytestmark = []


def _run(duff_loading: float, duff_moisture: float) -> tuple:
    """
    Run a small, fast-fizzling litter-only fire, with or without duff.

    :param duff_loading: Duff dry-weight loading (kg/m^2); ``0.0`` for
        no duff.
    :param duff_moisture: Duff moisture content (fraction).
    :return: ``(results, summary, class_order)`` from :func:`run_burnup`.
    """
    return run_burnup(
        fuel_loadings={"litter": 0.05},
        fuel_moistures={"litter": 0.10},
        intensity=50.0,
        ig_time=60.0,
        windspeed=0.0,
        depth=0.3,
        ambient_temp=21.0,
        duff_loading=duff_loading,
        duff_moisture=duff_moisture,
        timestep=15.0,
        max_times=3000,
    )


def test_small_litter_fire_fizzles_quickly_with_no_duff():
    """Class (a) baseline: with NO duff present, a small litter-only
    fire's simulation ends shortly after the litter's own fire intensity
    drops to ``fimin`` -- establishes the "no duff" comparison point the
    discriminating test below needs.

    :return: None. Raises via ``assert`` on mismatch.
    """
    results, _summary, _class_order = _run(duff_loading=0.0, duff_moisture=0.30)
    final_time = results[-1].time
    # A small litter load with no duff should fizzle out well within a
    # few minutes -- a loose bound (not tuned to the exact value), just
    # establishing this is genuinely the "quick" baseline.
    assert final_time < 600.0, final_time


def test_duff_still_smoldering_extends_the_simulation_past_wood_litter_fizzle():
    """F-62 fix, discriminating proof: the SAME small litter fire, WITH
    a duff load whose own burn duration (:func:`_duff_burn`'s ``tdf``)
    is far longer than when litter alone fizzles out, must run until
    duff is (approximately) exhausted -- NOT stop at the same short time
    the no-duff case does. This is exactly what the pre-fix code got
    wrong: ``if fi_cur <= fimin or ncalls >= ntimes: break`` ignored
    duff entirely, so this scenario would have terminated at
    approximately the SAME short time as the no-duff case above,
    discarding duff's own smoldering tail.

    :return: None. Raises via ``assert`` on mismatch.
    """
    duff_loading = 6.0
    duff_moisture = 0.30

    no_duff_results, _s1, _c1 = _run(duff_loading=0.0, duff_moisture=duff_moisture)
    no_duff_final_time = no_duff_results[-1].time

    with_duff_results, _s2, _c2 = _run(
        duff_loading=duff_loading, duff_moisture=duff_moisture,
    )
    with_duff_final_time = with_duff_results[-1].time

    _dfi, tdf, _rate = _duff_burn(duff_loading, duff_moisture)

    # The discriminating assertion: presence of a long-burning duff load
    # must extend the simulation well past where the SAME fuel bed with
    # no duff fizzles out. Under the pre-fix termination logic, this
    # would fail (with_duff_final_time == no_duff_final_time, since duff
    # was never consulted).
    assert with_duff_final_time > no_duff_final_time + 60.0, (
        with_duff_final_time, no_duff_final_time,
    )

    # The simulation should end close to duff's own burn-out time (tdf),
    # not stop arbitrarily early or run away far past it -- within one
    # timestep's worth of slack either side (15 s, plus a `Duff_CPTS`-
    # style residual-amount tail is expected to land within one
    # additional timestep of tdf, matching the C++ mechanism this ports).
    assert abs(with_duff_final_time - tdf) < 30.0, (with_duff_final_time, tdf)
