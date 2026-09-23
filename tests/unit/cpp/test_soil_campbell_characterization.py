#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_soil_campbell_characterization.py - Executed C++-vs-Python
comparisons for ``soil_heat_campbell`` (``src/pyfofem/components/soil_heating.py``).

**SUPERSEDED 2026-09-18 (F-70 narrow numerical-compatibility pass, third
round).** Every "F-52's conclusion is UNCHANGED" / "class (c)
characterization, never parity" / "structurally different models"
statement below (this note's own paragraph, the "Assertion class"
paragraph, and every per-scenario dict's own historical note) is now
HISTORICAL. F-70 found and fixed a real, isolated, directly-evidenced
transcription error: Python's ``_SOIL_FAMILY_DEFAULTS`` divided
``bulk_density``/``particle_density`` by 1000 ("g/m^3 -> kg/m^3, for
this module's own SI convention"), which is harmless everywhere ``bd``/
``pd`` are used as the ratio ``xs = bd/pd``, but silently corrupted the
ONE place they are used as an ABSOLUTE value: ``_soiltemp_step``'s own
``cp[i] = v[i]*(0.87*bd + 4.18e6*wn[i])/dt`` heat-capacity term -- see
``gate0/04-findings.md`` F-70 for the live-C++-diagnostic evidence
(measured pinned ``cp[1]`` for the dry non-duff scenario's first Newton
sub-iteration: 639.550, vs the pre-fix-derived 105.035). With that fixed,
Python's coupled solver (the F-70 first-round port of C++'s real
``soiltemp_step``, NOT the old heat-only ``_campbell_rhs``/``solve_ivp``
model F-52 originally compared) reproduces the pinned C++ execution to
well under 0.001 degC max |diff|, pooled over all 14 layers and every
recorded timestep, for every one of the 11 committed BR-SOI-DUFF/
BR-SOI-NODUFF scenarios (all 5 soil families) -- see
``_MEASURED_LANE_DIVERGENCE``/``_MEASURED_DUFF_DIVERGENCE``/
``_MEASURED_FIELD_DIVERGENCE``'s own re-measured values. Every
comparison in this module is therefore now genuine class (c) PARITY
evidence, not characterization of an accepted structural difference --
``soil_campbell.duff``/``nonduff`` in ``tolerance_policy.json`` were
updated from ``"unverified"`` to ``"verified"`` accordingly (see that
file's own dated note). The values pinned below were re-measured
against the golden's own PRECISE ``_field.csv`` data, not the
``_summary``'s integer-truncated ``lay0X_max_temp_c`` columns (C++'s own
``d_SO.ir_Temp[i]`` is declared ``int`` -- comparing floats against that
column produces a spurious +-1 degC "residual" that is pure rounding
noise, not model disagreement, once the real divergence shrinks below
it).

HISTORICAL (2026-09-16 Campbell duff-forcing correction pass, F-53 / new
finding F-69). Everything below this note that describes F-53 as an
unresolved, currently-failing defect ("current, confirmed-defective
behaviour", "strict xfail", "flat start_temp output") is now HISTORICAL:
``_duff_flux_and_duration()`` was replaced by ``_duff_burn_profile()``
(ported directly from C++'s ``DuffBurn()``/``SD_HeatAdj()``,
``bur_brn.cpp:1950-1986``/``fof_sd.cpp:98-129,294-313``), which converts
``duff_moisture`` from the documented whole-percent convention to C++'s
ratio convention exactly once, uses ``duff_load`` as a real required
input, and evaluates the duff-to-soil heat-transmission fraction at a
TIME-VARYING remaining duff depth instead of a static pre-fire value
(finding F-69: the duff route's own missing load-dependence and static-
depth defects, distinct from F-53's unit-conversion defect, found and
fixed in the same pass). The duff route no longer produces a flat
``start_temp`` line for any of the 6 committed BR-SOI-DUFF scenarios. The
former strict-xfail pin
(``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``)
is now a real passing test
(``test_duff_route_produces_positive_surface_forcing_at_realistic_moisture``),
and ``_MEASURED_DUFF_DIVERGENCE`` was re-measured against the real,
now-nonzero Python output (see that dict's own note for the new values).
**F-52's conclusion is UNCHANGED**: this pass corrects only the duff
SURFACE FORCING, not the underlying Campbell heat-conduction solver, so
the two implementations remain structurally different models and this
module's comparisons remain class (c) characterization, never parity.

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
``soil_campbell`` route stays ``"unverified"``/``"contract_only"``, per
F-51/F-52). A future pass that changes either implementation's physics
should expect these pinned values to change and should re-measure and
re-pin them, not loosen the tolerance until green.

**Correction-pass part-2 item-3: characterization-regression precision is
centrally defined and is NOT a parity tolerance.**
``CHARACTERIZATION_REGRESSION_PRECISION_DEGC`` (``_soil_campbell_contract.py``) is
the ``pytest.approx(..., abs=...)`` precision used below when re-asserting a
previously measured divergence as a pinned regression value - it exists
only to absorb ordinary floating-point reproducibility noise, never to
bound scientific agreement. It is not read by, or written back into,
``tolerance_policy.json``; every ``soil_campbell`` route stays
``"unverified"``/``"contract_only"`` with null ``atol``/``rtol``. This
module's own ``test_soil_campbell_tolerance_hygiene.py`` enforces that this module
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

Companion module ``test_soil_campbell_contract.py`` carries this
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
``soil_campbell.duff_moisture_unit`` contract-defect route
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
``generate_soil_campbell_goldens.py``'s own deterministic ``wl=[50-3i]``/
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
from tests.cpp_parity_live._soil_campbell_contract import (
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
    "Fine-Silt": dict(bulk_density=1.3e6, particle_density=2.35e6,
                       k_mineral=2.31, vries_shape=0.071),
    "Loamy-Skeletal": dict(bulk_density=0.8e6, particle_density=2.13e6,
                            k_mineral=1.03, vries_shape=0.13),
    "Fine": dict(bulk_density=1.15e6, particle_density=2.35e6,
                 k_mineral=2.21, vries_shape=0.084),
    "Coarse-Silt": dict(bulk_density=1.23e6, particle_density=2.35e6,
                         k_mineral=2.53, vries_shape=0.103),
    "Coarse-Loamy": dict(bulk_density=1.3e6, particle_density=2.35e6,
                          k_mineral=2.57, vries_shape=0.106),
}

#: (case_id, C++ soil_type, Python soil_family, soil_moist_pct) for the 5
#: primary BR-SOI-NODUFF scenarios (one per family; the 6th scenario is a
#: boundary variant of Coarse-Silt, not a 6th family, per
#: ``_soil_campbell_contract.py``'s own scenario-matrix comment).
_NODUFF_FAMILY_CASES = (
    ("SOI-NOD-01", "Fine-Silt", "fine-silty", 10.0),
    ("SOI-NOD-02", "Loamy-Skeletal", "loamy-skeletal", 20.0),
    ("SOI-NOD-03", "Fine", "fine", 15.0),
    ("SOI-NOD-04", "Coarse-Silt", "coarse-silty", 5.0),
    ("SOI-NOD-05", "Coarse-Loamy", "coarse-loamy", 12.0),
)

#: RE-MEASURED 2026-09-18 (F-70 narrow numerical-compatibility pass, third
#: round). F-52's "materially different implementations" conclusion is
#: SUPERSEDED (see the module docstring's RESOLVED note and
#: ``gate0/04-findings.md`` F-52/F-70): a real, isolated
#: ``bulk_density``/``particle_density`` unit-conversion transcription
#: error was found and fixed (``_soiltemp_step``'s ``cp[i]`` heat-capacity
#: term used ``bd`` as an absolute value, not a ratio, so the prior
#: dict's /1000 "SI conversion" silently corrupted every node's heat
#: capacity). These values are now genuine PARITY evidence, not
#: characterization of a structural difference: measured against the
#: golden's own precise ``_field.csv`` per-timestep/per-layer data (the
#: same interpolation methodology
#: ``test_coarse_silt_full_field_overlap_divergence_matches_measured_evidence``
#: already used), not the ``_summary``'s integer-truncated
#: ``lay0X_max_temp_c`` columns (C++'s own ``d_SO.ir_Temp[i]`` is
#: declared ``int`` -- comparing against it caused a spurious ~0.5-1.3
#: degC "residual" that was pure integer-rounding noise, not a remaining
#: model difference; confirmed directly by recomputing against the
#: precise field data instead). Lane A (Python defaults) and lane B
#: (explicit override to the same, now-identical pinned C++ literals)
#: are bit-for-bit identical for every family, confirmed by direct
#: measurement -- overriding with an already-matching value is a no-op.
#: Re-measure and update these (do not loosen the precision constant) if
#: either implementation's physics changes.
_MEASURED_LANE_DIVERGENCE = {
    "SOI-NOD-01": dict(lane_a=(0.000160, 0.000007), lane_b=(0.000160, 0.000007)),
    "SOI-NOD-02": dict(lane_a=(0.000019, 0.000003), lane_b=(0.000019, 0.000003)),
    "SOI-NOD-03": dict(lane_a=(0.000024, 0.000002), lane_b=(0.000024, 0.000002)),
    "SOI-NOD-04": dict(lane_a=(0.000375, 0.000028), lane_b=(0.000375, 0.000028)),
    "SOI-NOD-05": dict(lane_a=(0.000011, 0.000001), lane_b=(0.000011, 0.000001)),
}

#: RE-MEASURED 2026-09-18 (F-70 third round, same fix as above). Full-field
#: evidence for SOI-NOD-04 (Coarse-Silt), a handful of representative
#: layers -- see F-70 for the complete write-up. Comparison timing uses
#: the golden's real, harness-emitted ``time_s`` column exactly as
#: before (Phase 5 correction pass item-2) -- only the compared VALUES
#: changed, not the methodology.
_MEASURED_FIELD_DIVERGENCE = {
    0: dict(max=0.000234, mean=0.000050),
    4: dict(max=0.000194, mean=0.000038),
    8: dict(max=0.000043, mean=0.000007),
    13: dict(max=0.000000, mean=0.000000),
}

#: (case_id, C++ soil_type, Python soil_family, duff_dep_pre_in,
#: soil_moist_pct, duff_load_tac, duff_consumed_pct, duff_moist_pct) for all
#: 6 BR-SOI-DUFF scenarios, transcribed verbatim from
#: ``_soil_campbell_contract._DUFF_SCENARIOS``. ``duff_dep_pos_in`` (post-fire
#: depth) has no direct Python INPUT counterpart -- ``soil_heat_campbell``'s
#: duff route takes a single ``duff_depth`` (the PRE-fire depth), mapped
#: here from ``duff_dep_pre_in``, the only one of the two C++ columns with
#: a like-for-like Python parameter. As of the Campbell duff-forcing
#: correction pass, Python now DERIVES its own post-fire depth internally
#: (``_duff_burn_profile``'s ``post_depth_cm``, from ``pct_consumed``) and
#: evaluates the surface-heat transmission fraction at the time-varying
#: remaining depth between the two, rather than at a single static value.
_DUFF_SCENARIO_CASES = (
    ("SOI-DUF-01", "Fine-Silt", "fine-silty", 2.0, 10.0, 5.0, 50.0, 60.0),
    ("SOI-DUF-02", "Loamy-Skeletal", "loamy-skeletal", 3.0, 20.0, 8.0, 40.0, 70.0),
    ("SOI-DUF-03", "Fine", "fine", 1.5, 15.0, 3.0, 60.0, 55.0),
    ("SOI-DUF-04", "Coarse-Silt", "coarse-silty", 2.0, 5.0, 5.0, 50.0, 45.0),
    ("SOI-DUF-05", "Coarse-Loamy", "coarse-loamy", 2.5, 12.0, 6.0, 45.0, 50.0),
    ("SOI-DUF-06", "Coarse-Silt", "coarse-silty", 2.0, 5.0, 5.0, 97.5, 45.0),
)

#: RE-MEASURED 2026-09-18 (F-70 narrow numerical-compatibility pass, third
#: round). F-52's structural-model-difference conclusion is SUPERSEDED
#: (see the module docstring's RESOLVED note and ``gate0/04-findings.md``
#: F-52/F-70): the SAME isolated ``bulk_density``/``particle_density``
#: unit-conversion fix described above for the non-duff route applies
#: identically to the duff route (both call the same
#: ``_soiltemp_step``). Measured against the golden's own precise
#: ``_field.csv`` data (not ``_summary``'s integer-truncated columns --
#: see ``_MEASURED_LANE_DIVERGENCE``'s own note for why that matters).
#: Lane A and lane B are bit-for-bit identical for every family (an
#: override to an already-matching value is a no-op). These values MUST
#: be re-measured again (not merely re-tolerated) if either
#: implementation's physics changes further.
_MEASURED_DUFF_DIVERGENCE = {
    "SOI-DUF-01": dict(lane_a=(0.000006, 0.000001), lane_b=(0.000006, 0.000001)),
    "SOI-DUF-02": dict(lane_a=(0.000009, 0.000001), lane_b=(0.000009, 0.000001)),
    "SOI-DUF-03": dict(lane_a=(0.000010, 0.000001), lane_b=(0.000010, 0.000001)),
    "SOI-DUF-04": dict(lane_a=(0.000009, 0.000002), lane_b=(0.000009, 0.000002)),
    "SOI-DUF-05": dict(lane_a=(0.000009, 0.000001), lane_b=(0.000009, 0.000001)),
    "SOI-DUF-06": dict(lane_a=(0.000129, 0.000019), lane_b=(0.000129, 0.000019)),
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


def _python_duff_df(family: str, soil_moist_pct: float,
                     duff_dep_pre_in: float, duff_load_tac: float,
                     duff_consumed_pct: float, duff_moist_pct: float,
                     overrides: Dict[str, float] = None):
    """
    Run ``soil_heat_campbell('duff', ...)`` with the reconstructed BR-SOI-DUFF
    scenario inputs and return the raw output DataFrame.

    :param family: One of ``_SOIL_FAMILY_DEFAULTS``'s keys.
    :param soil_moist_pct: Soil moisture, percent (golden's ``soil_moist_pct``).
    :param duff_dep_pre_in: Pre-fire duff depth (in), mapped to Python's
        ``duff_depth``.
    :param duff_load_tac: Duff load (T/ac), mapped to Python's ``duff_load``
        (a required, genuinely consumed input as of the Campbell
        duff-forcing correction pass -- see ``_duff_burn_profile``).
    :param duff_consumed_pct: Duff consumed (percent), mapped to Python's
        ``pct_consumed``.
    :param duff_moist_pct: Duff moisture (percent), mapped to Python's
        ``duff_moisture``.
    :param overrides: Optional soil-property overrides (lane B alignment).
    :returns: The ``soil_heat_campbell`` output DataFrame (index: minutes,
        14 depth columns).
    """
    duff_params = dict(duff_load=duff_load_tac, duff_depth=duff_dep_pre_in,
                        duff_moisture=duff_moist_pct, pct_consumed=duff_consumed_pct)
    soil_params = dict(soil_family=family, start_water=soil_moist_pct / 100.0,
                        start_temp=_START_TEMP)
    if overrides:
        soil_params.update(overrides)
    return soil_heat_campbell("duff", duff_params, soil_params, _DEPTHS)


def _python_duff_max_temps(family: str, soil_moist_pct: float,
                            duff_dep_pre_in: float, duff_load_tac: float,
                            duff_consumed_pct: float, duff_moist_pct: float,
                            overrides: Dict[str, float] = None) -> List[float]:
    """
    Same scenario as :func:`_python_duff_df`, reduced to the per-column
    maximum over the whole run.

    :returns: 14 max-temperature values (degC), Surface then 13 depths.
    """
    df = _python_duff_df(family, soil_moist_pct, duff_dep_pre_in, duff_load_tac,
                          duff_consumed_pct, duff_moist_pct, overrides)
    return df.max().to_numpy().tolist()


def _python_nonduff_df(family: str, soil_moist_pct: float,
                        overrides: Dict[str, float] = None):
    """
    Run ``soil_heat_campbell('non_duff', ...)`` with the reconstructed
    scenario inputs and return the raw output DataFrame.

    :param family: One of ``_SOIL_FAMILY_DEFAULTS``'s keys.
    :param soil_moist_pct: Soil moisture, percent (golden's ``soil_moist_pct``).
    :param overrides: Optional soil-property overrides (lane B alignment).
    :returns: The ``soil_heat_campbell`` output DataFrame (index: minutes,
        14 depth columns).
    """
    soil_params = dict(soil_family=family, start_water=soil_moist_pct / 100.0,
                        start_temp=_START_TEMP)
    if overrides:
        soil_params.update(overrides)
    return soil_heat_campbell(
        "non_duff", {}, soil_params, _DEPTHS,
        burnup_intensity=_WL_SERIES, burnup_intensity_hs=_HS_SERIES,
        burnup_times=_TIMES_S,
    )


def _python_max_temps(family: str, soil_moist_pct: float,
                       overrides: Dict[str, float] = None) -> List[float]:
    """
    Same scenario as :func:`_python_nonduff_df`, reduced to the per-column
    maximum over the whole run.

    :returns: 14 max-temperature values (degC), Surface then 13 depths.
    """
    return _python_nonduff_df(family, soil_moist_pct, overrides).max().to_numpy().tolist()


def _field_max_mean_diff(case_id: str, df) -> tuple:
    """
    Compare *df* (a ``soil_heat_campbell`` output DataFrame) against the
    golden's own precise ``_field.csv`` per-timestep/per-layer data for
    *case_id*, via linear interpolation of Python's own time grid onto
    the golden's real, harness-emitted ``time_s`` column (Phase 5
    correction pass item-2 -- no constant C++ timestep is assumed),
    restricted to the time window Python's own fixed-duration policy
    actually covers. This is the SAME precise-data methodology
    :func:`test_coarse_silt_full_field_overlap_divergence_matches_measured_evidence`
    uses, generalized to any scenario/family -- deliberately NOT the
    ``_summary``'s ``lay0X_max_temp_c`` columns, which are declared
    ``int`` in the pinned C++ ``d_SO`` struct and therefore carry an
    inherent +-1 degC rounding floor unrelated to model agreement.

    :param case_id: Golden scenario identifier (e.g. ``"SOI-NOD-04"``).
    :param df: Python's output DataFrame for the same scenario.
    :returns: ``(max_abs_diff, mean_abs_diff)`` (degC), pooled over all
        14 layers within the overlapping time window.
    """
    field_rows = [r for r in golden_rows("soil_campbell", "_field")
                  if r["case_id"] == case_id]
    cpp_t = np.array([float(r["time_s"]) for r in field_rows])
    cpp_lay = np.array([int(r["layer_index"]) for r in field_rows])
    cpp_temp = np.array([float(r["temp_c"]) for r in field_rows])

    py_t = df.index.to_numpy() * 60.0
    overlap_end = py_t.max()

    max_diffs = []
    mean_diffs = []
    for layer in range(14):
        mask = cpp_lay == layer
        t_l, temp_l = cpp_t[mask], cpp_temp[mask]
        keep = t_l <= overlap_end
        t_l, temp_l = t_l[keep], temp_l[keep]
        if t_l.size == 0:
            continue
        py_interp = np.interp(t_l, py_t, df.iloc[:, layer].to_numpy())
        diff = np.abs(py_interp - temp_l)
        max_diffs.append(float(diff.max()))
        mean_diffs.append(float(diff.mean()))
    return max(max_diffs), float(np.mean(mean_diffs))


def test_coarse_silt_full_field_overlap_divergence_matches_measured_evidence():
    """Class (c), now genuine PARITY evidence (F-70 third round -- see
    ``_MEASURED_FIELD_DIVERGENCE``'s own note). Full time x depth
    comparison for SOI-NOD-04 (Coarse-Silt), over the window Python's
    own fixed-duration policy actually covers, pins the measured
    per-layer max/mean |diff| (degC) via linear interpolation of
    Python's own time grid onto the golden's real ``time_s`` column
    (Phase 5 correction pass item-2 - executable evidence, not an
    assumed step)."""
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
    """Class (c), now genuine PARITY evidence (F-70 third round -- see
    ``_MEASURED_DUFF_DIVERGENCE``'s own note). For each of the 6
    BR-SOI-DUFF scenarios, compares Python's per-timestep/per-layer
    temperature (lane A: Python defaults; lane B: Python with
    bulk_density/particle_density/k_mineral/vries_shape explicitly
    overridden to the same, now-identical pinned C++ literals) against
    the golden's own precise ``_field.csv`` data (:func:`_field_max_mean_diff`
    -- NOT the ``_summary``'s integer-truncated ``lay0X_max_temp_c``
    columns), and pins the measured max/mean |diff|. Lane A and lane B
    are bit-for-bit identical for every family now (confirmed directly:
    overriding with an already-matching value is a no-op)."""
    expected = _MEASURED_DUFF_DIVERGENCE[case_id]

    df_a = _python_duff_df(family, soil_moist_pct, dep_pre, load, consumed, moist)
    max_a, mean_a = _field_max_mean_diff(case_id, df_a)
    assert np.isfinite([max_a, mean_a]).all()
    assert max_a == pytest.approx(expected["lane_a"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert mean_a == pytest.approx(expected["lane_a"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)

    df_b = _python_duff_df(family, soil_moist_pct, dep_pre, load, consumed, moist,
                            overrides=_CPP_PRIMARY_INPUTS[soil_type])
    max_b, mean_b = _field_max_mean_diff(case_id, df_b)
    assert np.isfinite([max_b, mean_b]).all()
    assert max_b == pytest.approx(expected["lane_b"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert mean_b == pytest.approx(expected["lane_b"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)


def test_duff_route_produces_nonzero_surface_flux_at_the_committed_scenario_moistures():
    """RESOLVED 2026-09-16 (F-53 / F-69, Campbell duff-forcing correction
    pass). Class (c), oracle-independent structural observation, not a
    parity claim, and NOT a claim about "ignition"
    (``soil_heat_campbell()`` has no ignition/no-ignition decision of its
    own - only C++'s ``brn_ignited`` input models that). Python receives
    every committed BR-SOI-DUFF scenario's ``duff_moist_pct`` value
    (45-70%) UNCHANGED, exactly as documented, and ``_duff_burn_profile()``
    now converts it to C++'s ratio convention before evaluating C++'s own
    intensity formula, so every one of these 6 scenarios now computes
    genuinely positive, time-varying surface flux and a non-flat output -
    confirmed directly by execution (every scenario's max exceeds
    ``start_temp`` by at least ~2.7 degC; SOI-DUF-03 exceeds it by ~45.5
    degC).

    HISTORICAL: this test was previously named
    ``test_duff_route_produces_zero_surface_flux_at_the_committed_scenario_moistures``
    and asserted the OPPOSITE - a flat ``start_temp`` output for all 6
    scenarios - as CURRENT, CONFIRMED-DEFECTIVE behaviour (F-53:
    ``_duff_flux_and_duration()``'s ``i_d = max(7.5e-4 -
    2.7e-4*duff_moisture, 0.0)`` reached exactly 0 once ``duff_moisture``
    exceeded ``7.5e-4/2.7e-4 = 2.7778...`` -- ~2.78% on the documented
    percent scale, so every value in the committed 45-70% range clamped to
    zero). That defect is now fixed; see
    ``test_duff_route_produces_positive_surface_forcing_at_realistic_moisture``
    (formerly a strict ``xfail``) for the single-scenario desired-behaviour
    pin this test's own fix made pass."""
    for case_id, _soil_type, family, dep_pre, soil_moist_pct, load, consumed, moist in _DUFF_SCENARIO_CASES:
        py_max = np.array(_python_duff_max_temps(family, soil_moist_pct, dep_pre,
                                                   load, consumed, moist))
        assert np.isfinite(py_max).all(), case_id
        # At least one layer's max must genuinely DEVIATE from
        # start_temp (either direction) -- proving nonzero forcing
        # actually reaches the profile. Requiring every layer to
        # individually EXCEED start_temp is no longer valid post-F-70-fix:
        # some scenarios (weaker forcing / wetter soil, e.g. SOI-DUF-02)
        # now correctly show the surface layer's reported (extrapolated)
        # temperature dip transiently BELOW start_temp before recovering
        # -- confirmed to match the live pinned C++ diagnostic exactly
        # for this same scenario (see gate0/04-findings.md F-70's
        # "stable-duff" trace), not a residual Python defect.
        assert np.abs(py_max - _START_TEMP).max() > 0.05, case_id


def test_duff_route_produces_positive_surface_forcing_at_realistic_moisture():
    """RESOLVED 2026-09-16 (F-53 / F-69, Campbell duff-forcing correction
    pass): this was a strict ``xfail`` pinning the DESIRED behaviour;
    ``_duff_burn_profile()`` (the ``_duff_flux_and_duration()``
    replacement) now converts ``duff_moisture`` from the documented
    whole-percent convention to C++'s internal ratio convention exactly
    once, at the forcing boundary (matching ``fof_sd.cpp:100``:
    ``f_DuffMoist = a_SD->f_DufMoi / 100.0;``, and ``DuffBurn``'s own
    header comment, ``bur_brn.cpp:1950``: ``dfm......Duff Moisture -
    decial percent, 0 -> 1.96``, a ratio, not a percent). A realistic,
    ordinary ``duff_moisture=45.0`` (a whole percent, exactly as
    ``soil_heat_campbell``'s own docstring documents the parameter) now
    genuinely delivers positive surface heat flux and raises the soil
    column above ``start_temp`` -- confirmed by direct execution, not
    merely by the assertion below passing. Verified under ``--runxfail``
    equivalent (this is no longer marked xfail at all) that this is a
    real pass, not a vacuous one.

    HISTORICAL: this test was previously
    ``test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``,
    a strict ``xfail`` (F-53) asserting exactly this same desired
    behaviour, confirmed genuinely failing against the pre-fix flat-
    ``start_temp`` output. The assertion and inputs are unchanged; only
    the ``xfail`` marker and name (dropping "should_produce" for
    "produces") were removed/updated to reflect that this is now real,
    current behaviour, not a desired-but-unmet pin."""
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
    """Class (c), now genuine PARITY evidence (F-70 third round -- see
    ``_MEASURED_LANE_DIVERGENCE``'s own note). For each of the 5 primary
    non-duff families, compares Python's per-timestep/per-layer
    temperature (lane A: Python defaults; lane B: Python with
    bulk_density/particle_density/k_mineral/vries_shape explicitly
    overridden to the same, now-identical pinned C++ literals) against
    the golden's own precise ``_field.csv`` data
    (:func:`_field_max_mean_diff` -- NOT the ``_summary``'s
    integer-truncated ``lay0X_max_temp_c`` columns), and pins the
    measured max/mean |diff|. Lane A and lane B are bit-for-bit
    identical for every family now (confirmed directly: overriding with
    an already-matching value is a no-op)."""
    expected = _MEASURED_LANE_DIVERGENCE[case_id]

    df_a = _python_nonduff_df(family, soil_moist_pct)
    max_a, mean_a = _field_max_mean_diff(case_id, df_a)
    assert np.isfinite([max_a, mean_a]).all()
    assert max_a == pytest.approx(expected["lane_a"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert mean_a == pytest.approx(expected["lane_a"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)

    df_b = _python_nonduff_df(family, soil_moist_pct, overrides=_CPP_PRIMARY_INPUTS[soil_type])
    max_b, mean_b = _field_max_mean_diff(case_id, df_b)
    assert np.isfinite([max_b, mean_b]).all()
    assert max_b == pytest.approx(expected["lane_b"][0], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)
    assert mean_b == pytest.approx(expected["lane_b"][1], abs=CHARACTERIZATION_REGRESSION_PRECISION_DEGC)


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
    ``tolerance_policy.json``'s ``soil_campbell.noig`` "contract_only"
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
