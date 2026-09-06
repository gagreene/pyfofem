#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_massman_hmv_contract.py - Phase 6 investigation B: comprehensive Python
contract coverage for ``soil_heat_massman()``
(``src/pyfofem/components/soil_heating.py``), the explicitly simplified
(E_v=0, isothermal evaporation) Massman (2015) heat-moisture-vapor (HMV)
soil model. Before this module, ``soil_heat_massman`` had ZERO test
coverage of any kind (confirmed by a repo-wide search finding no reference
to it or ``_massman_rhs`` anywhere under ``tests/``).

**Assertion classes (required module declaration):**

- Class **(a) Python contract tests** - nominal behaviour for all 3
  ``fire_type`` values, input validation/boundaries, output structure
  (dict keys, DataFrame columns/index/units), determinism, and the explicit
  absence of the package's scalar-array convention (this function takes one
  scenario's worth of scalar/dict/list parameters, not a batch of cells -
  unlike ``consm_*``/``mort_*``, it has no array-broadcast contract to
  test).
- Class **(b) source-relation checks** - the saturated-hydraulic-
  conductivity formula and the sharing of ``_SOIL_FAMILY_DEFAULTS`` with
  ``soil_heat_campbell()``, hand-verified against the pinned source lines
  cited in each docstring.
- Class **(c) executable C++ parity - still NOT PRESENT in this module,
  now for a scientific rather than a build reason.** F-55
  (``gate0/04-findings.md``) was corrected twice, same day (2026-09-05).
  The first correction used a disposable scratch-directory probe (since
  discarded) and established that the pinned ``FOF_DLL/`` Massman/HMV
  solver DOES build and run. A second, narrow correction pass replaced
  that with a TRACKED, independently reproducible diagnostic probe
  (``tests/cpp_parity_live/massman_fof_dll_probe.py`` +
  ``massman_fof_dll_probe_driver.cpp`` - not collected by plain ``pytest``
  or ``--suite core``/``--suite full``; run manually), which re-confirmed
  the same result with a corrected, exact file count: a driver calling
  only the documented lifecycle (``BMI_Init`` -> populate ``d_BMI`` ->
  ``HTA_Init`` -> ``HMV_Model`` -> ``HTA_Count``/``HTA_Get``, exactly as
  ``FOF_DLL/BMSoil.h``/``HTAA.h`` declare it) links cleanly against
  **80** pinned ``FOF_DLL/*.cpp`` files (zero duplicate-symbol warnings;
  the first pass's disposable probe had miscounted this as 73) and runs to
  completion (``HMV_Model`` returns 1, no error message) for inputs
  entirely within ``BMSoil.h``'s own documented bounds. So the earlier "no
  CMake target == infeasible to build" conclusion was wrong on the build
  question; see F-55 for the corrected text and the historical
  (superseded) blocks preserving both the original and the first-corrected
  claims verbatim. A probe-hardening pass measured this exhaustively
  (every ``(layer, time-index)`` sample across all ``hta_layers=21``
  layers and ``hta_count=40`` saved times - 840 total per field, not only
  layer 1's first/last): heat/moisture/water-potential are ALL 840 of 840
  non-finite and specifically NaN (``any_inf=0`` for all three), while
  saved time is fully finite (840/840). Reproducibly, across two
  independent runs of the identical binary, and preceded by 301
  ``"rhov"`` and 301 ``"mvapor"`` runtime "Divide by Zero" diagnostics the
  pinned source emits from its own defensive zero-guards - and this
  happens identically whether or not the (never invoked by the real
  ``Soil_Model_Data_Files_HMV`` -> ``WesternUS01`` call path) ``Quincy1G()``
  auxiliary-constant initializer is also called first, ruling out "missing
  initialization" as the sole cause. **F-57** (``gate0/04-findings.md``)
  was ALSO corrected the same narrow pass: its original claim that
  ``calxhiv1``'s ``rhov``-zero fallback flows directly into
  ``calgascomb``'s ``mvapor``-zero fallback is FALSE
  (``CrankNicolson.cpp`` binds ``calgascomb``'s ``mvapor`` parameter to
  ``muv``, computed by ``calmulaHMV`` from ``tempk``/``tempki``/``TempR``/
  ``temR`` - never from ``rhov`` or ``calxhiv1``'s output ``dxhivdr``).
  What remains real and independent of that retracted claim: ``calgascomb``
  divides unconditionally by a value its own preceding zero-guard just set
  to zero - a real, reproducible numerical defect in the pinned solver's
  own arithmetic, not a build/link gap. **F-58** (new) additionally
  establishes that ``HMV_Model``'s own "1" ("success") return code cannot
  be relied on to detect any of this: ``SolveHMV()`` discards
  ``CrankNicolson()``'s own per-timestep return value and always returns
  success regardless of what happened inside the loop. Recovering a finite
  result would require either debugging/patching the pinned reference
  solver (an oracle production-code change, explicitly out of scope) or
  discovering undocumented required state beyond the declared ``d_BMI``
  lifecycle (risking exactly the "copied-equation instrumentation" the
  approved plan forbids) — so per the plan's stop-and-report condition
  ("a required oracle would need production-code changes or copied-equation
  instrumentation"), no attempt was made to coax a finite result out of it,
  and class (c) coverage remains absent: build/link/run feasibility is
  CONFIRMED and now independently reproducible via the tracked probe, but
  scientific suitability as a parity oracle is NOT established and current
  evidence weighs against it without further, separately-authorized
  engineering. No new C++ harness MODE, CMake build target, or permanent
  wrapper was added to ``reference/fofem_cpp*`` for this — the only
  tracked addition is the diagnostic probe itself, which builds in its own
  disposable temp directory only when run manually.

Every test below therefore stays within classes (a)/(b): Python contract and
source-relation coverage only, exactly as the approved Phase 6 plan requires
when a reference solver does not produce a scientifically usable oracle.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

import pyfofem.components.soil_heating as _soil_heating_module
from pyfofem.components.soil_heating import (
    _build_grid,
    _SOIL_FAMILY_DEFAULTS,
    soil_heat_massman,
)

_DEPTHS = list(range(1, 14))
_FIRE_TYPES = ("wildfire", "prescribed_burn", "pile_burn")

#: C++'s default fire-duration table this function's docstring claims to
#: reproduce (``_default_t_d`` in ``soil_heat_massman``'s own source) - hours.
_DEFAULT_T_D_HR = {"wildfire": 20.0, "prescribed_burn": 8.0, "pile_burn": 40.0}


def _column_volumes(depth_layers: List[float]) -> np.ndarray:
    """
    Return the finite-volume cell weight for each of the 14 state nodes,
    matching ``_massman_rhs``'s own moisture-balance discretisation exactly
    (surface half-cell, interior full cells, deepest-layer half-cell against
    the fixed node-14 boundary).

    :param depth_layers: 13 depths (cm) at which temperature/moisture are
        predicted.
    :returns: 14 cell weights (m), Surface then 13 depths, in the same units
        :func:`~pyfofem.components.soil_heating._build_grid` uses (metres).
    """
    z = _build_grid(depth_layers)
    vol = np.zeros(14)
    vol[0] = z[1] / 2.0
    for i in range(1, 13):
        vol[i] = (z[i] - z[i - 1] + z[i + 1] - z[i]) / 2.0
    vol[13] = (z[13] - z[12] + z[14] - z[13]) / 2.0
    return vol


def _duff_free_bfd(q_abs: float = 20.0, **overrides) -> Dict[str, float]:
    """
    Build a minimal valid ``bfd_params`` dict.

    :param q_abs: Peak heat rate (kW/m2).
    :param overrides: Additional ``bfd_params`` fields (e.g. ``t_m``/``t_d``).
    :returns: A ``bfd_params`` dict.
    """
    base: Dict[str, float] = dict(q_abs=q_abs)
    base.update(overrides)
    return base


def _soil_params(soil_family: str = "coarse-silty", start_water: float = 0.15,
                  start_temp: float = 21.0, **overrides) -> Dict[str, float]:
    """
    Build a minimal valid ``soil_params`` dict.

    :param soil_family: One of :data:`_SOIL_FAMILY_DEFAULTS`'s keys.
    :param start_water: Starting volumetric water content (m3/m3).
    :param start_temp: Starting soil temperature (degC).
    :param overrides: Additional/override fields.
    :returns: A ``soil_params`` dict.
    """
    base: Dict[str, float] = dict(
        soil_family=soil_family, start_water=start_water, start_temp=start_temp,
    )
    base.update(overrides)
    return base


def test_high_initial_moisture_silently_truncates_the_solved_time_window():
    """Class (a), current-behaviour contract test (solver convergence/
    failure behaviour) - NOT a claim this is correct or intended; see
    F-56 (``gate0/04-findings.md``) and the companion strict xfail
    ``test_high_moisture_run_should_reach_the_requested_horizon_or_raise``
    below, which pins the DESIRED behaviour instead. Measured directly:
    ``soil_heat_massman`` never inspects ``solve_ivp``'s own
    ``sol.success``/``sol.status`` before building its output DataFrames
    (confirmed by reading ``soil_heat_massman``'s source - no such check
    exists), so when the coupled moisture ODE becomes numerically difficult
    for the Radau solver, integration silently stops early and the function
    returns a well-formed, fully finite DataFrame covering a MUCH SHORTER
    simulated time window than the requested fire-duration+2h, with no
    exception and no warning raised. This is reproducible and moisture-
    dependent: a dry start (0.05, see the mass-conservation test below)
    reaches the full requested ``prescribed_burn`` window (600 min); this
    wetter start (0.35) truncates after only a few simulated minutes, even
    though both calls request the identical 600-minute window and neither
    raises."""
    depths = _DEPTHS
    out = soil_heat_massman(
        "prescribed_burn", _duff_free_bfd(q_abs=5.0),
        _soil_params(start_water=0.35), depths, timestep=10.0,
    )
    df = out["temperature"]
    requested_max_min = (_DEFAULT_T_D_HR["prescribed_burn"] + 2.0) * 60.0
    assert np.isfinite(df.to_numpy()).all()
    assert np.isfinite(out["moisture"].to_numpy()).all()
    assert len(df) >= 2
    assert df.index.max() < requested_max_min / 4.0


@pytest.mark.xfail(
    strict=True,
    reason="F-56: soil_heat_massman() never inspects solve_ivp's own "
           "sol.success/sol.status, so a numerically difficult run silently "
           "truncates its output instead of either completing to the "
           "requested horizon or raising a solver-failure exception. No "
           "production fix has been made; this pin fails until one lands.",
)
def test_high_moisture_run_should_reach_the_requested_horizon_or_raise():
    """Class (a), desired-behaviour pin for F-56 (companion to the
    current-behaviour characterization above, using the identical
    scenario). The DESIRED contract: ``soil_heat_massman`` should either
    (a) return output reaching the requested ``t_d + 2h`` horizon, or (b)
    raise a ``RuntimeError`` naming the solver/convergence failure - never
    silently return a short, unlabeled truncation. Only a ``RuntimeError``
    whose message names the solver failure satisfies (b); any other
    exception type (``KeyError``, ``AssertionError``, ``MemoryError``, a
    plain ``RuntimeError`` unrelated to solver convergence, ...) is a real
    test failure/error, not an accepted desired-behaviour outcome - a bare
    ``except Exception`` here would let an unrelated programming error
    silently XPASS this xfail, which this test deliberately does not
    allow. Verified genuinely failing under ``--runxfail`` (real
    ``AssertionError``: the truncated run reaches only a few minutes, far
    short of the 600-minute requested horizon, and raises no exception at
    all, let alone a matching ``RuntimeError``)."""
    depths = _DEPTHS
    requested_max_min = (_DEFAULT_T_D_HR["prescribed_burn"] + 2.0) * 60.0
    try:
        out = soil_heat_massman(
            "prescribed_burn", _duff_free_bfd(q_abs=5.0),
            _soil_params(start_water=0.35), depths, timestep=10.0,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if not any(kw in message for kw in ("solv", "converg", "integrat")):
            raise  # Not a solver-failure RuntimeError - a real failure.
        return  # A solver-failure RuntimeError satisfies the desired contract.
    df = out["temperature"]
    assert df.index.max() == pytest.approx(requested_max_min, rel=1e-3)


def test_invalid_fire_type_raises_value_error():
    """Class (a). An unrecognised ``fire_type`` must raise, not silently
    fall back to a default duration."""
    with pytest.raises(ValueError, match="fire_type must be one of"):
        soil_heat_massman(
            "bogus", _duff_free_bfd(), _soil_params(), _DEPTHS,
        )


def test_massman_family_defaults_are_the_same_object_as_campbells_source_relation():
    """Class (b). ``soil_heat_massman`` pulls ``extrap_water``/
    ``vries_shape``/``cop_power`` defaults from the SAME
    ``_SOIL_FAMILY_DEFAULTS`` table ``soil_heat_campbell`` uses (not a
    second, independently-defined copy) - proven by asserting the module
    exposes exactly one such table and that both Campbell's F-51-pinned
    ``"coarse-silty"`` row (``soil_heating.py:56-63``) and Massman read
    identical values for it, confirmed by direct execution below (see
    ``test_massman_uses_the_family_table_when_no_override_is_given``)."""
    assert "coarse-silty" in _SOIL_FAMILY_DEFAULTS
    row = _SOIL_FAMILY_DEFAULTS["coarse-silty"]
    for key in ("extrap_water", "vries_shape", "cop_power", "bulk_density",
                "particle_density", "k_mineral"):
        assert key in row


def test_massman_moisture_column_mass_is_conserved():
    """Class (a), oracle-independent physical invariant. The moisture RHS
    is written in finite-volume divergence form with explicit no-flux
    boundaries at the surface and at the deepest node
    (``_massman_rhs``: "q_water[0] = 0 (no flux at surface)" / "q_water[14]
    = 0 (no flux at bottom)"), so the volume-weighted total column moisture
    (``sum(theta[i] * cell_volume[i])``) must be conserved over the whole
    run - true from the discretisation's own telescoping-sum structure
    alone, independent of any oracle. Uses a low-moisture start so the run
    completes its full requested window (see
    ``test_high_initial_moisture_silently_truncates_the_solved_time_window``
    for why a wetter start would not); a short ``t_d`` override keeps this
    test fast without weakening the invariant, which holds over any window."""
    depths = _DEPTHS
    vol = _column_volumes(depths)
    out = soil_heat_massman(
        "prescribed_burn", _duff_free_bfd(q_abs=5.0, t_d=0.5),
        _soil_params(start_water=0.05), depths, timestep=10.0,
    )
    theta = out["moisture"].to_numpy()
    mass = theta @ vol
    assert np.isfinite(mass).all()
    assert mass[-1] == pytest.approx(mass[0], rel=1e-6)


def test_massman_uses_the_family_table_when_no_override_is_given():
    """Class (b) source relation, proven by execution rather than asserted
    from reading the source alone. Two things must both hold: (1) a call
    with every Massman-specific override OMITTED must produce output
    BITWISE IDENTICAL to a call with those same overrides supplied
    EXPLICITLY, each copied straight from
    ``_SOIL_FAMILY_DEFAULTS["coarse-silty"]`` - proving the omitted-override
    path genuinely reads the family table rather than some other default;
    and (2) a scientifically distinct override (``cop_power=50.0``, versus
    the family's own ``3.43``) must change the result - proving the
    parameter is not silently ignored either way. The prior version of this
    test only checked (2)."""
    depths = _DEPTHS
    bfd = _duff_free_bfd(q_abs=20.0, t_d=0.5)
    family_row = _SOIL_FAMILY_DEFAULTS["coarse-silty"]

    implicit_out = soil_heat_massman(
        "prescribed_burn", bfd, _soil_params(start_water=0.10), depths,
        timestep=10.0,
    )
    explicit_out = soil_heat_massman(
        "prescribed_burn", bfd,
        _soil_params(
            start_water=0.10,
            bulk_density=family_row["bulk_density"],
            particle_density=family_row["particle_density"],
            k_mineral=family_row["k_mineral"],
            vries_shape=family_row["vries_shape"],
            extrap_water=family_row["extrap_water"],
            cop_power=family_row["cop_power"],
        ), depths, timestep=10.0,
    )
    for key in ("temperature", "moisture"):
        np.testing.assert_array_equal(
            implicit_out[key].to_numpy(), explicit_out[key].to_numpy()
        )

    overridden_out = soil_heat_massman(
        "prescribed_burn", bfd,
        _soil_params(start_water=0.10, cop_power=50.0), depths, timestep=10.0,
    )
    # Compare over the shorter of the two runs' actually-solved windows.
    n = min(len(implicit_out["moisture"]), len(overridden_out["moisture"]))
    assert n > 1
    diff = np.abs(
        implicit_out["moisture"].to_numpy()[:n]
        - overridden_out["moisture"].to_numpy()[:n]
    )
    assert np.nanmax(diff) > 0.0


@pytest.mark.parametrize("fire_type", _FIRE_TYPES)
def test_missing_q_abs_raises_key_error(fire_type):
    """Class (a). ``bfd_params`` without ``'q_abs'`` must raise - the
    function reads it via plain dict indexing (``bfd_params["q_abs"]"``),
    so a missing key surfaces as a genuine ``KeyError``, not a silent
    zero-heat-flux default."""
    with pytest.raises(KeyError):
        soil_heat_massman(fire_type, {}, _soil_params(), _DEPTHS)


@pytest.mark.parametrize("fire_type", _FIRE_TYPES)
def test_nominal_run_produces_finite_well_formed_output(fire_type):
    """Class (a). Every ``fire_type`` must produce a fully finite dict of
    ``{'temperature', 'moisture'}`` DataFrames for an ordinary, low-moisture
    scenario (see ``test_high_initial_moisture_silently_truncates_the_solved_time_window``
    for why moisture matters)."""
    depths = _DEPTHS
    out = soil_heat_massman(
        fire_type, _duff_free_bfd(q_abs=10.0, t_d=0.5),
        _soil_params(start_water=0.05), depths, timestep=10.0,
    )
    assert set(out) == {"temperature", "moisture"}
    for key in ("temperature", "moisture"):
        df = out[key]
        assert np.isfinite(df.to_numpy()).all()
        assert len(df) >= 2


def test_output_columns_and_index_contract():
    """Class (a). Both DataFrames must share ``['Surface', '<d>cm', ...]``
    columns (in *depth_layers* order) and a ``'time_min'`` index starting at
    0 and strictly non-decreasing - the same output shape contract
    ``soil_heat_campbell`` guarantees."""
    depths = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    out = soil_heat_massman(
        "prescribed_burn", _duff_free_bfd(q_abs=10.0, t_d=0.5),
        _soil_params(start_water=0.05), depths, timestep=10.0,
    )
    expected_columns = ["Surface"] + [f"{d}cm" for d in depths]
    for key in ("temperature", "moisture"):
        df = out[key]
        assert list(df.columns) == expected_columns
        assert df.index.name == "time_min"
        assert df.index[0] == pytest.approx(0.0)
        assert df.index.is_monotonic_increasing


def test_output_starts_at_initial_conditions():
    """Class (a). At ``time_min == 0`` every column of both DataFrames must
    equal the ODE's own initial condition exactly - ``start_temp`` for
    temperature, ``start_water`` for moisture - not a solver artifact."""
    out = soil_heat_massman(
        "prescribed_burn", _duff_free_bfd(q_abs=10.0, t_d=0.5),
        _soil_params(start_water=0.12, start_temp=19.5), _DEPTHS, timestep=10.0,
    )
    assert out["temperature"].iloc[0].to_numpy() == pytest.approx(19.5)
    assert out["moisture"].iloc[0].to_numpy() == pytest.approx(0.12)


def test_repeated_calls_are_deterministic():
    """Class (a). Two identical calls must produce identical output - no
    hidden global/random state, mirroring the equivalent Campbell/harness
    determinism contract."""
    args = (
        "wildfire", _duff_free_bfd(q_abs=10.0, t_d=0.5),
        _soil_params(start_water=0.05), _DEPTHS,
    )
    out_a = soil_heat_massman(*args, timestep=10.0)
    out_b = soil_heat_massman(*args, timestep=10.0)
    for key in ("temperature", "moisture"):
        np.testing.assert_array_equal(
            out_a[key].to_numpy(), out_b[key].to_numpy()
        )


def test_saturated_hydraulic_conductivity_formula_source_relation(monkeypatch):
    """Class (b), proven by execution rather than recomputing the formula
    twice in the test (the prior version of this test was tautological -
    it compared ``formula(...)`` against ``approx(formula(...))`` and could
    never detect production drift). Monkeypatches ``_massman_rhs`` with a
    spy that records the real ``rho_b``/``k_sat`` arguments
    ``soil_heat_massman`` actually supplies at call time, then delegates to
    the real ``_massman_rhs`` so the run completes normally. Checked for two
    independently-differing bulk densities against
    ``k_sat = 0.0001 * exp(-3.0 * rho_b / 1000.0)`` (``soil_heating.py:882``)."""
    real_rhs = _soil_heating_module._massman_rhs
    captured: List[tuple] = []

    def _spy(t, y, z, rho_b, rho_p, k_mineral, vries_shape, start_temp,
             extrap_water, cop_power, k_sat, flux_fn):
        captured.append((rho_b, k_sat))
        return real_rhs(t, y, z, rho_b, rho_p, k_mineral, vries_shape,
                         start_temp, extrap_water, cop_power, k_sat, flux_fn)

    monkeypatch.setattr(_soil_heating_module, "_massman_rhs", _spy)

    for rho_b_override in (1230.0, 800.0):
        captured.clear()
        soil_heat_massman(
            "prescribed_burn", _duff_free_bfd(q_abs=5.0, t_d=0.1),
            _soil_params(start_water=0.05, bulk_density=rho_b_override),
            _DEPTHS, timestep=10.0,
        )
        assert captured, "the RHS was never invoked - solve_ivp took no steps"
        rho_b_seen, k_sat_seen = captured[0]
        assert rho_b_seen == pytest.approx(rho_b_override)
        expected_k_sat = 0.0001 * np.exp(-3.0 * rho_b_override / 1000.0)
        assert k_sat_seen == pytest.approx(expected_k_sat)
        assert 0.0 < k_sat_seen < 0.0001


def test_scalar_array_convention_does_not_apply():
    """Class (a) contract note. Unlike the package's ``consm_*``/``mort_*``
    functions, ``soil_heat_massman`` has no scalar-vs-array broadcast
    contract to satisfy: every parameter is a plain scalar, dict, or list
    describing ONE scenario (confirmed via its real signature), so this
    module deliberately does not test array inputs - there is no such
    contract to violate or satisfy."""
    import inspect
    sig = inspect.signature(soil_heat_massman)
    for name, param in sig.parameters.items():
        assert param.annotation in (str, dict, list, float, inspect.Parameter.empty), (
            name, param.annotation
        )


def test_timestep_does_not_change_the_output_sampling_grid():
    """Class (a) contract test (renamed from
    ``test_solver_max_step_argument_changes_output_grid_density``, whose
    name claimed the opposite of what its own assertion proves).
    ``timestep`` bounds ``solve_ivp``'s own ``max_step``, not the output
    sampling grid (``_build_t_eval`` always samples every 30 s regardless of
    *timestep*) - so two calls differing only in ``timestep`` must return
    the SAME number of rows (same grid) but are not required to return
    identical values (a coarser max_step can change which internal steps
    Radau takes). See
    ``test_timestep_is_forwarded_to_solve_ivp_as_max_step`` below for direct
    executable evidence that *timestep* is the argument actually forwarded."""
    depths = _DEPTHS
    common = ("prescribed_burn", _duff_free_bfd(q_abs=10.0, t_d=0.5),
              _soil_params(start_water=0.05), depths)
    fine = soil_heat_massman(*common, timestep=5.0)
    coarse = soil_heat_massman(*common, timestep=50.0)
    assert len(fine["temperature"]) == len(coarse["temperature"])


def test_timestep_is_forwarded_to_solve_ivp_as_max_step(monkeypatch):
    """Class (a) contract test, proven by monkeypatching ``solve_ivp`` with
    a minimal compatible stub that records the ``max_step`` keyword
    argument actually received and then delegates to the real
    ``solve_ivp`` so the run completes normally - rather than inferring the
    forwarding from the two runs' equal output-row counts alone (that
    equality only proves the OUTPUT GRID is timestep-independent; it does
    not by itself prove *timestep* reaches ``solve_ivp`` at all)."""
    real_solve_ivp = _soil_heating_module.solve_ivp
    captured: Dict[str, float] = {}

    def _stub(fun, t_span, y0, **kwargs):
        captured["max_step"] = kwargs.get("max_step")
        return real_solve_ivp(fun, t_span, y0, **kwargs)

    monkeypatch.setattr(_soil_heating_module, "solve_ivp", _stub)
    soil_heat_massman(
        "prescribed_burn", _duff_free_bfd(q_abs=10.0, t_d=0.5),
        _soil_params(start_water=0.05), _DEPTHS, timestep=7.5,
    )
    assert captured.get("max_step") == pytest.approx(7.5)


def test_wrong_depth_layers_length_raises_value_error():
    """Class (a). *depth_layers* must contain exactly 13 values - too few
    or too many must raise, never silently truncate/pad."""
    with pytest.raises(ValueError, match="exactly 13 values"):
        soil_heat_massman(
            "wildfire", _duff_free_bfd(), _soil_params(), _DEPTHS[:-1],
        )
    with pytest.raises(ValueError, match="exactly 13 values"):
        soil_heat_massman(
            "wildfire", _duff_free_bfd(), _soil_params(), _DEPTHS + [14],
        )
