#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase5_soil_campbell_characterization.py - Executed C++-vs-Python
comparisons for ``soil_heat_campbell`` (``src/pyfofem/components/soil_heating.py``).

**Assertion class: (c) cross-implementation CHARACTERIZATION, explicitly NOT
parity.** F-52 (``gate0/04-findings.md``) established that Python's
``soil_heat_campbell()`` and C++'s ``soiltemp_step`` (driven by
``SE_Mngr_Array``/``SD_Mngr``) are structurally different models, not one
model at two numerical precisions: C++ integrates coupled
temperature/water-pressure/humidity/vapor state plus an ambient
(Stefan-Boltzmann) radiative floor and recirculation parameters
(``r_xwo``/``r_cop``) that Python's Campbell function never represents at
all, and the two integrators do not share a stopping-time rule. Every test
below therefore compares real, executed Python output against the committed
Phase 5 golden and PINS the measured divergence as a regression value - it
never asserts scientific equality, never chooses a tolerance to make a
comparison pass, and never converts a divergence into an
``atol``/``rtol`` entry in ``tolerance_policy.json`` (every
``soil_campbell_p5`` route stays ``"unverified"``/``"contract_only"``, per
F-51/F-52). A future pass that changes either implementation's physics
should expect these pinned values to change and should re-measure and
re-pin them, not loosen the tolerance until green.

**Correction-pass part-2 item-3: characterization-regression precision is
centrally defined and is NOT a parity tolerance.**
``CHARACTERIZATION_REGRESSION_PRECISION_DEGC`` (``_phase5_contract.py``) is
the ``pytest.approx(..., abs=...)`` precision used below when re-asserting a
previously measured divergence as a pinned regression value - it exists
only to absorb ordinary floating-point reproducibility noise, never to
bound scientific agreement. It is not read by, or written back into,
``tolerance_policy.json``; every ``soil_campbell_p5`` route stays
``"unverified"``/``"contract_only"`` with null ``atol``/``rtol``. This
module's own ``test_phase5_contract_hygiene.py`` enforces that this module
references the named constant rather than hardcoding a raw tolerance
literal.

**Correction-pass part-3 item-2: the "physically sane envelope" backstop was
REMOVED, not independently justified.** Part 2 added a separate
``CHARACTERIZATION_SANITY_ENVELOPE_DEGC = 150.0`` bound and a dedicated test
asserting every measured divergence stayed under it. Independent review
correctly identified this as tuned-to-the-result, not independently
derived: ``SOI-DUF-06``'s real measured max divergence is EXACTLY 150.0 degC
(the envelope's own value), so the "loose, independently-justified backstop"
the comments claimed was in fact set by the very comparison it was meant to
police. No independently-derived physical/solver-domain bound for this
comparison exists in any bundled source, and inventing one would itself be
an unjustified tuned number. The envelope constant and its test
(``test_lane_outputs_remain_within_a_physically_sane_envelope``) were
therefore deleted outright, rather than loosened: every measurement below
already asserts ``np.isfinite(...)`` on its own lane output AND pins the
exact measured max/mean divergence via
``CHARACTERIZATION_REGRESSION_PRECISION_DEGC`` - a genuine blow-up (solver
divergence, a unit-conversion mistake in a future edit) would already fail
one of those two checks with no separate envelope needed.

Companion module ``test_phase5_soil_campbell_contract.py`` carries this
function's classes (a) Python contract tests and (b) source-relation checks;
this module exists only for class (c).

**Correction-pass item-1: characterization coverage is now complete across
all three BR-SOI-* scenario categories**, not just the 5 primary BR-SOI-NODUFF
families:

- BR-SOI-NODUFF (``SOI-NOD-01..05``, unchanged from before this pass) -
  ``test_lane_a_and_lane_b_max_temp_divergence_matches_measured_evidence``,
  ``test_coarse_silt_full_field_overlap_divergence_matches_measured_evidence``.
- BR-SOI-DUFF (``SOI-DUF-01..06``, added by the part-2 correction pass) -
  ``test_duff_route_max_temp_divergence_matches_measured_evidence``. Measuring
  it surfaced a genuine, previously unexecuted characterization fact,
  precisely stated (correcting a base-10 arithmetic error from the part-2
  pass, see F-53): Python receives every committed duff scenario's
  ``duff_moist_pct`` value (45-70%) UNCHANGED, exactly as documented (a
  percent). ``_duff_flux_and_duration()``'s
  ``i_d = max(7.5e-4 - 2.7e-4*duff_moisture, 0.0)`` reaches exactly zero
  once ``duff_moisture`` exceeds ``7.5e-4/2.7e-4 = 2.7778...`` -- **~2.78%,
  not ~27.8%** -- so on this documented percent scale, EVERY value in the
  45-70% range clamps ``i_d`` to zero: Python's current equation therefore
  computes exactly zero surface heat flux for the duff route at all 6
  committed scenarios, and its output is a flat ``start_temp`` line as a
  direct consequence. Lane A and lane B (family-property alignment) are
  therefore numerically identical for every duff scenario: the zero flux
  is 100% a consequence of Python's own current threshold arithmetic, not a
  family-constant-alignment question the way the non-duff lane comparison
  is. **This is characterization of CURRENT, CONFIRMED-DEFECTIVE
  behaviour (F-53), not a scientific-validation claim, and it is NOT
  described as "ignition" here** -- ``soil_heat_campbell()`` has no
  ignition/no-ignition decision of its own; "zero computed duff surface
  flux" is the directly proven fact, "never ignites" would imply a
  modelled ignition concept this function does not have. See
  ``test_duff_route_produces_zero_surface_flux_at_the_committed_scenario_moistures``
  for the isolated current-behaviour observation, and
  ``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``
  (a strict ``xfail``, correction-pass part 5) for the DESIRED-behaviour pin
  that genuinely fails against this defect today.
- BR-SOI-NOIG (``SOI-NOIG-01``, added by this pass) -
  ``test_no_ignition_scenario_has_no_python_counterpart``. This route is
  ``"contract_only"`` in ``tolerance_policy.json`` (not ``"unverified"``):
  ``soil_heat_campbell()`` structurally has no third value/route representing
  "burnup never ignited" - both ``model`` values always integrate a real ODE
  from ``t=0`` - so there is no divergence to measure, only a documented
  non-comparability to assert.

**Correction-pass part 5: F-53 is CONFIRMED, not unresolved, and is now
pinned as a strict xfail expressing DESIRED behaviour.** Part 3's write-up
left the duff-ignition-moisture unit convention unresolved for lack of a
bundled source stating it explicitly. Independent review supplied the
missing direct evidence: Frandsen (1991), "Burning Rate of Smoldering
Peat" (the exact primary source the FOFEM 6-7 Guide's p.51 formula cites as
"Frandsen 1991b"), defines its moisture ratio ``R_M`` as a mass ratio
(range 0.0-0.8), not a whole percent; and the pinned C++ independently
performs the identical conversion in the identical call path
(``fof_sd.cpp:100``: ``f_DuffMoist = a_SD->f_DufMoi / 100.0;``, feeding
``DuffBurn`` (``bur_brn.cpp:1950``), whose own header comment states
``dfm......Duff Moisture - decial percent, 0 -> 1.96`` -- a ratio, not a
percent). See F-53 (``gate0/04-findings.md``) for the complete evidence
chain. This is now a **confirmed Python defect** (a missing percent-to-
ratio conversion), not an open interpretation question, and is pinned
accordingly:
``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``
is a strict ``@pytest.mark.xfail`` asserting the DESIRED behaviour (a
realistic ``duff_moisture=45.0`` should heat the soil column above
``start_temp``) and genuinely fails against today's real output (a flat
``start_temp`` field) -- proven under ``--runxfail``, never an imperative
``pytest.xfail()`` call. No production code is changed by this pass; the
production fix itself remains a release-readiness decision requiring
separate user authorization, not a Phase 5 test-suite change. Every
existing characterization test in this section (measuring the CURRENT
defective flat-line output) is retained and relabeled explicitly as
current-defective-behaviour pinning, not scientific validation, and is
documented as needing re-measurement once the production fix lands. No
``atol``/``rtol`` is assigned for this defect and it is not described as
C++ parity; ``tolerance_policy.json`` records it as its own
``soil_campbell_p5.duff_moisture_unit`` contract-defect route
(``status: "known_divergent_strict_xfail"``, null ``atol``/``rtol``),
distinct from the ``duff``/``nonduff`` full-model-characterization routes,
which remain unchanged (``"unverified"``, F-51/F-52's model-structural
divergence is a separate, additional reason those stay uncompared).

**Scenario/parameter reconstruction, traced directly from the pinned C++
source (not guessed):** ``start_temp=21.0`` is C++'s ``e_StaSoiTem``
constant (``fof_sh2.h:17``), copied into every ``sr_SE``/``sr_SD`` row and
never overridden by the harness. ``depth_layers=[1..13]`` cm matches the
harness's own compile-time layer table exactly (``fof_sh.cpp:162-163``:
``rr_Lay = {0, 0, 10, 20, ..., 130}`` mm -> layer indices 1-13 are 1-13 cm
after the leading surface/dummy slots), confirmed independently by the
FOFEM 6-7 User Guide's example non-duff input file (p.55: ``layers 1 2 3 4
5 6 7 8 9 10 11 12 13``). The fire-intensity series and 15 s interval match
``generate_phase5_goldens.py``'s own deterministic ``wl=[50-3i]``/
``hs=[10-0.5i]`` construction and C++'s pinned ``i_frInc=15`` (``fof_sh.cpp:67``).
Efficiencies are left at Python's function defaults (0.15/0.10) because
every golden row exercised here carries the ``-1`` "use built-in default"
sentinel, which C++ resolves to the identical 0.15/0.10 (`fof_se.cpp:64-67`,
already pinned as a source relation in the companion contract module).

One acknowledged, documented restriction, not an assumption: C++'s per-row
total simulated duration is not reconstructed exactly (the outer loop's
``_Done()`` stopping rule is family-dependent and the time-stepped
nonlinear/Newton solve can halve its own step on non-convergence,
`fof_se.cpp:105-163`) - the full-field comparison below restricts the
comparison to the time window Python's own fixed-duration policy actually
covers. The per-row time axis itself is no longer an assumption: the
committed golden's ``field.csv`` carries a real ``time_s`` column (Phase 5
correction pass item-2, `time_index * SHA_GetInc()`, read directly from the
harness), read here in place of the earlier ``time_index * 10.0``
approximation this module used before that column existed.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import inspect
from typing import Dict, List

import numpy as np
import pytest

from pyfofem.components.soil_heating import soil_heat_campbell
from tests.cpp_parity_live._phase5_contract import (
    CHARACTERIZATION_REGRESSION_PRECISION_DEGC,
    golden_rows,
    require_golden_tree,
)

require_golden_tree()

pytestmark = pytest.mark.soil_solver

_START_TEMP = 21.0
_DEPTHS = list(range(1, 14))
_N_STEPS = 20
_TIMES_S = [i * 15.0 for i in range(_N_STEPS)]
_WL_SERIES = [max(0.0, 50.0 - 3.0 * i) for i in range(_N_STEPS)]
_HS_SERIES = [max(0.0, 10.0 - 0.5 * i) for i in range(_N_STEPS)]

#: C++ (soil_type, primary-input dict) for lane B, transcribed verbatim from
#: the pinned ``sr_SE``/``sr_SD`` tables (``fof_se2.h``/``fof_sd2.h`` -
#: identical between the two, both non-duff and duff share one physical
#: table). Only the 4 inputs ``soil_heat_campbell()`` actually consumes
#: (F-52) are listed - ``extrap_water``/``cop_power`` are deliberately
#: omitted, since aligning them would align nothing Python reads.
_CPP_PRIMARY_INPUTS: Dict[str, Dict[str, float]] = {
    "Fine-Silt": dict(bulk_density=1300.0, particle_density=2350.0,
                       k_mineral=2.31, vries_shape=0.071),
    "Loamy-Skeletal": dict(bulk_density=800.0, particle_density=2130.0,
                            k_mineral=1.03, vries_shape=0.13),
    "Fine": dict(bulk_density=1150.0, particle_density=2350.0,
                 k_mineral=2.21, vries_shape=0.084),
    "Coarse-Silt": dict(bulk_density=1230.0, particle_density=2350.0,
                         k_mineral=2.53, vries_shape=0.103),
    "Coarse-Loamy": dict(bulk_density=1300.0, particle_density=2350.0,
                          k_mineral=2.57, vries_shape=0.106),
}

#: (case_id, C++ soil_type, Python soil_family, soil_moist_pct) for the 5
#: primary BR-SOI-NODUFF scenarios (one per family; the 6th scenario is a
#: boundary variant of Coarse-Silt, not a 6th family, per
#: ``_phase5_contract.py``'s own scenario-matrix comment).
_NODUFF_FAMILY_CASES = (
    ("SOI-NOD-01", "Fine-Silt", "fine-silty", 10.0),
    ("SOI-NOD-02", "Loamy-Skeletal", "loamy-skeletal", 20.0),
    ("SOI-NOD-03", "Fine", "fine", 15.0),
    ("SOI-NOD-04", "Coarse-Silt", "coarse-silty", 5.0),
    ("SOI-NOD-05", "Coarse-Loamy", "coarse-loamy", 12.0),
)

#: Measured F-52 lane-A/B evidence (max|diff|, mean|diff|, degC), pinned as
#: a regression - see ``gate0/04-findings.md`` F-52 for the full table and
#: derivation. Re-measure and update these (do not loosen a tolerance)
#: if either implementation's physics changes.
_MEASURED_LANE_DIVERGENCE = {
    "SOI-NOD-01": dict(lane_a=(49.064, 9.493), lane_b=(31.301, 7.397)),
    "SOI-NOD-02": dict(lane_a=(27.595, 5.756), lane_b=(54.605, 8.878)),
    "SOI-NOD-03": dict(lane_a=(47.722, 8.483), lane_b=(33.602, 7.205)),
    "SOI-NOD-04": dict(lane_a=(16.119, 5.769), lane_b=(16.119, 5.769)),
    "SOI-NOD-05": dict(lane_a=(31.891, 6.749), lane_b=(26.029, 6.516)),
}

#: Measured F-52 full-field evidence for SOI-NOD-04 (Coarse-Silt), a
#: handful of representative layers - see F-52 for the complete 14-layer
#: table and the interpolation method. Comparison timing uses the golden's
#: real, harness-emitted ``time_s`` column (``time_index * SHA_GetInc()``,
#: Phase 5 correction pass item-2) - no constant C++ timestep is assumed.
_MEASURED_FIELD_DIVERGENCE = {
    0: dict(max=22.354, mean=4.029),
    4: dict(max=7.679, mean=4.415),
    8: dict(max=3.983, mean=3.009),
    13: dict(max=2.475, mean=1.124),
}

#: (case_id, C++ soil_type, Python soil_family, duff_dep_pre_in,
#: soil_moist_pct, duff_load_tac, duff_consumed_pct, duff_moist_pct) for all
#: 6 BR-SOI-DUFF scenarios, transcribed verbatim from
#: ``_phase5_contract._DUFF_SCENARIOS``. ``duff_dep_pos_in`` (post-fire
#: depth) has no Python counterpart -- ``soil_heat_campbell``'s duff route
#: takes a single ``duff_depth`` (used only to derive the surface-heat
#: proportion ``h``, ``_duff_flux_and_duration``), mapped here from the
#: PRE-fire depth (``duff_dep_pre_in``), the only one of the two C++ columns
#: with a like-for-like Python parameter.
_DUFF_SCENARIO_CASES = (
    ("SOI-DUF-01", "Fine-Silt", "fine-silty", 2.0, 10.0, 5.0, 50.0, 60.0),
    ("SOI-DUF-02", "Loamy-Skeletal", "loamy-skeletal", 3.0, 20.0, 8.0, 40.0, 70.0),
    ("SOI-DUF-03", "Fine", "fine", 1.5, 15.0, 3.0, 60.0, 55.0),
    ("SOI-DUF-04", "Coarse-Silt", "coarse-silty", 2.0, 5.0, 5.0, 50.0, 45.0),
    ("SOI-DUF-05", "Coarse-Loamy", "coarse-loamy", 2.5, 12.0, 6.0, 45.0, 50.0),
    ("SOI-DUF-06", "Coarse-Silt", "coarse-silty", 2.0, 5.0, 5.0, 97.5, 45.0),
)

#: CURRENT, CONFIRMED-DEFECTIVE Python behaviour (F-53), NOT a scientific
#: characterization of two working implementations the way the non-duff
#: lane divergence above is. Measured max|diff|/mean|diff| (degC, over all
#: 14 layers) between the golden's real C++ duff heating and Python's
#: current flat-``start_temp`` output, which F-53 confirms is a
#: factor-of-100 unit-conversion defect (Frandsen 1991 + pinned
#: ``fof_sd.cpp:100`` both confirm the equation requires ``duff_moisture``
#: as a ratio, e.g. ``0.45``, not the documented whole percent ``45.0``
#: Python actually receives and applies unconverted). These pinned values
#: MUST be re-measured (not merely re-tolerated) once F-53's production fix
#: lands -- a fix changes Python's underlying duff-route output from a flat
#: line to real computed values, so today's diffs become meaningless. See
#: also ``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``,
#: the strict-xfail pin of the DESIRED (currently failing) behaviour.
_MEASURED_DUFF_DIVERGENCE = {
    "SOI-DUF-01": dict(max=5.0, mean=0.571),
    "SOI-DUF-02": dict(max=3.0, mean=0.571),
    "SOI-DUF-03": dict(max=14.0, mean=1.714),
    "SOI-DUF-04": dict(max=13.0, mean=1.214),
    "SOI-DUF-05": dict(max=1.0, mean=0.5),
    "SOI-DUF-06": dict(max=150.0, mean=19.143),
}


def _golden_max_temps(case_id: str) -> List[float]:
    """
    Return the golden's 14 ``lay0X_max_temp_c`` values for *case_id*, in
    layer order (0 = Surface).

    :param case_id: Scenario identifier.
    :returns: 14 max-temperature values (degC).
    """
    row = _golden_summary_row(case_id)
    return [float(row[f"lay{i:02d}_max_temp_c"]) for i in range(14)]


def _golden_summary_row(case_id: str) -> Dict[str, str]:
    """
    Return the committed Phase 5 ``soil_campbell`` summary row for *case_id*.

    :param case_id: Scenario identifier (e.g. ``"SOI-NOD-04"``).
    :returns: The row as ``{column: raw string value}``.
    :raises StopIteration: If no row matches *case_id*.
    """
    return next(r for r in golden_rows("soil_campbell", "_summary")
                if r["case_id"] == case_id)


def _python_duff_max_temps(family: str, soil_moist_pct: float,
                            duff_dep_pre_in: float, duff_load_tac: float,
                            duff_consumed_pct: float, duff_moist_pct: float,
                            overrides: Dict[str, float] = None) -> List[float]:
    """
    Run ``soil_heat_campbell('duff', ...)`` with the reconstructed BR-SOI-DUFF
    scenario inputs and return the per-column maximum over the whole run.

    :param family: One of ``_SOIL_FAMILY_DEFAULTS``'s keys.
    :param soil_moist_pct: Soil moisture, percent (golden's ``soil_moist_pct``).
    :param duff_dep_pre_in: Pre-fire duff depth (in), mapped to Python's
        ``duff_depth``.
    :param duff_load_tac: Duff load (T/ac), mapped to Python's ``duff_load``
        (accepted but not consumed by ``_duff_flux_and_duration``).
    :param duff_consumed_pct: Duff consumed (percent), mapped to Python's
        ``pct_consumed``.
    :param duff_moist_pct: Duff moisture (percent), mapped to Python's
        ``duff_moisture``.
    :param overrides: Optional soil-property overrides (lane B alignment).
    :returns: 14 max-temperature values (degC), Surface then 13 depths.
    """
    duff_params = dict(duff_load=duff_load_tac, duff_depth=duff_dep_pre_in,
                        duff_moisture=duff_moist_pct, pct_consumed=duff_consumed_pct)
    soil_params = dict(soil_family=family, start_water=soil_moist_pct / 100.0,
                        start_temp=_START_TEMP)
    if overrides:
        soil_params.update(overrides)
    df = soil_heat_campbell("duff", duff_params, soil_params, _DEPTHS)
    return df.max().to_numpy().tolist()


def _python_max_temps(family: str, soil_moist_pct: float,
                       overrides: Dict[str, float] = None) -> List[float]:
    """
    Run ``soil_heat_campbell('non_duff', ...)`` with the reconstructed
    scenario inputs and return the per-column maximum over the whole run.

    :param family: One of ``_SOIL_FAMILY_DEFAULTS``'s keys.
    :param soil_moist_pct: Soil moisture, percent (golden's ``soil_moist_pct``).
    :param overrides: Optional soil-property overrides (lane B alignment).
    :returns: 14 max-temperature values (degC), Surface then 13 depths.
    """
    soil_params = dict(soil_family=family, start_water=soil_moist_pct / 100.0,
                        start_temp=_START_TEMP)
    if overrides:
        soil_params.update(overrides)
    df = soil_heat_campbell(
        "non_duff", {}, soil_params, _DEPTHS,
        burnup_intensity=_WL_SERIES, burnup_intensity_hs=_HS_SERIES,
        burnup_times=_TIMES_S,
    )
    return df.max().to_numpy().tolist()


def test_coarse_silt_full_field_overlap_divergence_matches_measured_evidence():
    """Class (c). Full time x depth comparison for SOI-NOD-04 (Coarse-Silt),
    over the window Python's fixed-duration policy actually covers, pins
    F-52's measured per-layer max/mean |diff| (degC) via linear
    interpolation of Python's 30 s grid onto the golden's own real
    ``time_s`` column (Phase 5 correction pass item-2 - executable
    evidence, not an assumed step). This is a regression pin of a
    documented divergence, not a parity assertion."""
    case_id = "SOI-NOD-04"
    field_rows = [r for r in golden_rows("soil_campbell", "_field")
                  if r["case_id"] == case_id]
    cpp_t = np.array([float(r["time_s"]) for r in field_rows])
    cpp_lay = np.array([int(r["layer_index"]) for r in field_rows])
    cpp_temp = np.array([float(r["temp_c"]) for r in field_rows])

    soil_params = dict(soil_family="coarse-silty", start_water=0.05,
                        start_temp=_START_TEMP)
    df = soil_heat_campbell(
        "non_duff", {}, soil_params, _DEPTHS,
        burnup_intensity=_WL_SERIES, burnup_intensity_hs=_HS_SERIES,
        burnup_times=_TIMES_S,
    )
    py_t = df.index.to_numpy() * 60.0
    overlap_end = py_t.max()

    for layer, expected in _MEASURED_FIELD_DIVERGENCE.items():
        mask = cpp_lay == layer
        t_l, temp_l = cpp_t[mask], cpp_temp[mask]
        keep = t_l <= overlap_end
        t_l, temp_l = t_l[keep], temp_l[keep]
        assert t_l.size > 0
        py_interp = np.interp(t_l, py_t, df.iloc[:, layer].to_numpy())
        diff = np.abs(py_interp - temp_l)
        assert np.isfinite(diff).all()
        assert float(diff.max()) == pytest.approx(expected["max"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
        assert float(diff.mean()) == pytest.approx(expected["mean"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)


@pytest.mark.parametrize(
    "case_id,soil_type,family,dep_pre,soil_moist_pct,load,consumed,moist",
    _DUFF_SCENARIO_CASES,
)
def test_duff_route_max_temp_divergence_matches_measured_evidence(
        case_id, soil_type, family, dep_pre, soil_moist_pct, load, consumed, moist):
    """Class (c). For each of the 6 BR-SOI-DUFF scenarios, compares Python's
    per-layer maximum temperature (lane A: Python defaults; lane B: Python
    with bulk_density/particle_density/k_mineral/vries_shape aligned to the
    pinned C++ table) against the golden's own ``lay0X_max_temp_c`` columns,
    and pins the measured max/mean |diff| (see this module's docstring and
    ``_MEASURED_DUFF_DIVERGENCE``). Lane A and lane B are identical for
    every scenario here - not a Coarse-Silt-only coincidence the way it is
    for the non-duff comparison - because Python's current duff-flux
    equation already computes exactly zero surface heat flux at every one
    of these scenarios' moisture values (see
    ``test_duff_route_produces_zero_surface_flux_at_the_committed_scenario_moistures``
    and F-53), so the family-property override changes nothing Python's
    output actually depends on for this route."""
    golden_max = np.array(_golden_max_temps(case_id))
    expected = _MEASURED_DUFF_DIVERGENCE[case_id]

    py_a = np.array(_python_duff_max_temps(family, soil_moist_pct, dep_pre,
                                            load, consumed, moist))
    diff_a = np.abs(py_a - golden_max)
    assert np.isfinite(diff_a).all()
    assert float(diff_a.max()) == pytest.approx(expected["max"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert float(diff_a.mean()) == pytest.approx(expected["mean"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)

    py_b = np.array(_python_duff_max_temps(family, soil_moist_pct, dep_pre,
                                            load, consumed, moist,
                                            overrides=_CPP_PRIMARY_INPUTS[soil_type]))
    diff_b = np.abs(py_b - golden_max)
    assert np.isfinite(diff_b).all()
    assert float(diff_b.max()) == pytest.approx(expected["max"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert float(diff_b.mean()) == pytest.approx(expected["mean"], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)


def test_duff_route_produces_zero_surface_flux_at_the_committed_scenario_moistures():
    """Class (c), oracle-independent structural observation, not a parity
    claim, and NOT a claim about "ignition" (``soil_heat_campbell()`` has no
    ignition/no-ignition decision of its own - only C++'s ``brn_ignited``
    input models that). Python receives every committed BR-SOI-DUFF
    scenario's ``duff_moist_pct`` value (45-70%) UNCHANGED, exactly as
    documented. Its current equation
    (``_duff_flux_and_duration``: ``i_d = max(7.5e-4 - 2.7e-4*duff_moisture,
    0.0)``) reaches exactly 0 once ``duff_moisture`` exceeds
    ``7.5e-4/2.7e-4 = 2.7778...`` -- **~2.78%, not ~27.8%** (F-53 corrects a
    base-10 arithmetic error made when this test was first written) -- so
    EVERY value in 45-70% already computes zero. ``soil_heat_campbell('duff',
    ...)`` therefore produces a flat ``start_temp`` output for every one of
    these 6 scenarios, confirmed by direct execution, because its computed
    surface flux is zero, not because of any modelled ignition concept.
    This pins real, current Python behaviour - a CONFIRMED defect (F-53:
    Frandsen 1991 and the pinned C++'s own ``fof_sd.cpp:100`` conversion
    both independently confirm the equation requires a ratio, not the
    documented whole percent Python actually receives and applies
    unconverted) - not merely a claim about which implementation's
    threshold is "correct" or scientifically intended. See
    ``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``
    for the strict-xfail pin of the desired behaviour this defect
    contradicts."""
    for case_id, _soil_type, family, dep_pre, soil_moist_pct, load, consumed, moist in _DUFF_SCENARIO_CASES:
        py_max = np.array(_python_duff_max_temps(family, soil_moist_pct, dep_pre,
                                                   load, consumed, moist))
        assert py_max == pytest.approx(_START_TEMP), case_id


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-53 (gate0/04-findings.md): _duff_flux_and_duration() never "
        "converts the documented whole-percent duff_moisture input to the "
        "ratio Frandsen (1991)/pinned C++ (fof_sd.cpp:100, "
        "f_DufMoi/100.0) require, so i_d clamps to zero above ~2.78% and "
        "the duff route computes zero surface flux for every realistic "
        "input. No production fix has been made; this pin fails until "
        "one lands."
    ),
)
def test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture():
    """DESIRED-behaviour pin (F-53), explicitly NOT class (c) current-
    behaviour characterization -- the only test in this module asserting
    what SHOULD happen rather than what currently does. A realistic,
    ordinary ``duff_moisture=45.0`` (a whole percent, exactly as
    ``soil_heat_campbell``'s own docstring documents the parameter) should
    make the duff route deliver positive surface heat flux and raise the
    soil column above ``start_temp`` -- both Frandsen (1991), "Burning Rate
    of Smoldering Peat" (the actual primary source the FOFEM 6-7 Guide's
    p.51 formula cites, moisture ratio range 0.0-0.8), and the pinned C++
    (``fof_sd.cpp:100``: ``f_DuffMoist = a_SD->f_DufMoi / 100.0;``,
    independently confirmed by ``DuffBurn``'s own header comment,
    ``bur_brn.cpp:1950``: ``dfm......Duff Moisture - decial percent, 0 ->
    1.96``) agree the underlying equation is defined on a ratio, not a
    whole percent. This assertion genuinely fails against real, current
    Python output (a flat ``start_temp`` field, confirmed directly by
    execution) -- do not weaken it to make it pass; re-run under
    ``--runxfail`` to confirm it still fails for real. No production code
    is changed by this pass -- the fix (dividing ``duff_moisture`` by 100
    before this equation, or an equivalent API-level decision) is a
    release-readiness change requiring separate user authorization."""
    df = soil_heat_campbell(
        "duff",
        dict(duff_load=5.0, duff_depth=2.0, duff_moisture=45.0, pct_consumed=50.0),
        dict(soil_family="coarse-silty", start_water=0.10, start_temp=_START_TEMP),
        _DEPTHS,
    )
    assert df.to_numpy().max() > _START_TEMP + 1e-6


def test_golden_oracle_never_drops_below_start_temp():
    """Class (c), oracle-independent physical invariant. The C++ golden's
    own reported per-layer maxima must be >= start_temp (21.0 degC) for
    every primary non-duff scenario - a sanity check on the oracle itself,
    true from energy conservation alone (this comparison's forcing terms
    are all non-negative), independent of any Python behaviour."""
    for case_id, _soil_type, _family, _pct in _NODUFF_FAMILY_CASES:
        maxima = _golden_max_temps(case_id)
        assert all(m >= _START_TEMP - 1e-6 for m in maxima)


@pytest.mark.parametrize("case_id,soil_type,family,soil_moist_pct", _NODUFF_FAMILY_CASES)
def test_lane_a_and_lane_b_max_temp_divergence_matches_measured_evidence(
        case_id, soil_type, family, soil_moist_pct):
    """Class (c). For each of the 5 primary non-duff families, compares
    Python's per-layer maximum temperature (lane A: Python defaults; lane
    B: Python with bulk_density/particle_density/k_mineral/vries_shape
    aligned to the pinned C++ table) against the golden's own
    ``lay0X_max_temp_c`` columns, and pins the measured max/mean |diff| from
    F-52. Lane A == lane B for Coarse-Silt (its inputs are already
    aligned); lane B does NOT uniformly reduce divergence for the other
    families (Loamy-Skeletal's roughly doubles) - both facts are asserted
    below exactly as measured, not smoothed over."""
    golden_max = np.array(_golden_max_temps(case_id))
    expected = _MEASURED_LANE_DIVERGENCE[case_id]

    py_a = np.array(_python_max_temps(family, soil_moist_pct))
    diff_a = np.abs(py_a - golden_max)
    assert np.isfinite(diff_a).all()
    assert float(diff_a.max()) == pytest.approx(expected["lane_a"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert float(diff_a.mean()) == pytest.approx(expected["lane_a"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)

    py_b = np.array(_python_max_temps(family, soil_moist_pct,
                                       overrides=_CPP_PRIMARY_INPUTS[soil_type]))
    diff_b = np.abs(py_b - golden_max)
    assert np.isfinite(diff_b).all()
    assert float(diff_b.max()) == pytest.approx(expected["lane_b"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert float(diff_b.mean()) == pytest.approx(expected["lane_b"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)


def test_no_ignition_scenario_has_no_python_counterpart():
    """Class (c), documented non-comparability, not a divergence
    measurement. BR-SOI-NOIG's committed golden (real C++ harness outcome,
    cross-checked against ``test_soil_no_ignition`` in
    ``test_cpp_harness_contract.py``, which drives the harness directly)
    reports outcome=ok, model="" (SH_Mngr never sets cr_Model on this path),
    and every summary maximum at exactly 0.0 - the genuine unmodified C++
    state for a never-ignited row. ``soil_heat_campbell()`` has no matching
    third route: ``model`` accepts only ``'duff'``/``'non_duff'`` and both
    always integrate a real ODE from t=0 - there is no way to construct a
    Python call representing "never ignited", so there is no divergence to
    measure here, only this documented absence (matching
    ``tolerance_policy.json``'s ``soil_campbell_p5.noig`` "contract_only"
    status, distinct from ``duff``/``nonduff``'s "unverified")."""
    row = _golden_summary_row("SOI-NOIG-01")
    assert row["outcome"] == "ok"
    assert row["model"] == ""
    for i in range(14):
        assert float(row[f"lay{i:02d}_max_temp_c"]) == 0.0
        assert float(row[f"lay{i:02d}_max_time_s"]) == 0.0

    field_rows = [r for r in golden_rows("soil_campbell", "_field")
                  if r["case_id"] == "SOI-NOIG-01"]
    assert len(field_rows) == 14 * int(row["n_time_indices"])
    assert all(float(r["temp_c"]) == pytest.approx(_START_TEMP) for r in field_rows)

    sig = inspect.signature(soil_heat_campbell)
    assert "model" in sig.parameters
    with pytest.raises(ValueError, match="model must be"):
        soil_heat_campbell("no_ignition", {}, dict(soil_family="coarse-silty",
                                                     start_water=0.10, start_temp=_START_TEMP),
                            _DEPTHS)


def test_python_depth_grid_matches_cpp_layer_table_source_relation():
    """Class (b)/(c) grid-alignment check. This module's ``_DEPTHS``
    ([1..13] cm) must match C++'s own compile-time layer table exactly
    (``fof_sh.cpp:162-163``: ``rr_Lay`` mm values 10,20,...,130 at layer
    indices 1-13), confirmed independently by the FOFEM 6-7 User Guide's
    example non-duff input file (p.55: ``layers 1 2 3 4 5 6 7 8 9 10 11 12
    13``). This means the depth axis of every comparison above needs no
    interpolation - only the time axis does."""
    cpp_layer_mm = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    cpp_layer_cm = [mm / 10.0 for mm in cpp_layer_mm]
    assert cpp_layer_cm == [0.0] + [float(d) for d in _DEPTHS]
