#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_soil_solver_diagnostic_comparison.py - F-70 diagnostic pass:
Python-side comparison against the live C++ soil-solver diagnostic
output (``tests/cpp_parity_live/test_cpp_harness_contract.py``'s
``_soidiag.csv``, gated by ``FOFEM_TEST_SOIL_DIAG``).

**This module makes NO parity claim.** It is explicitly a
characterization/investigation module: it locates and PINS the current
first observed divergence between Python's coupled soil solver
(``pyfofem.components.soil_heating``) and the real, live-compiled C++
``SH_Mngr`` execution path, for three required scenario categories (a
previously near-matching duff case, a materially-divergent dry duff case,
and a materially-divergent dry non-duff case). A future correction pass
that narrows or eliminates this divergence should update the pinned
values here alongside the production fix -- these tests failing (because
the divergence moved) is the CORRECT signal that something changed, not
evidence of a broken test.

**Class (b)/(c) coverage:**

- ``test_*_forcing_matches_cpp_exactly``: class (b), REAL bit-close
  agreement between Python's own ``forcing_fn`` output and the C++
  diagnostic's real per-tick ``surface_flux_w`` (read from the pinned
  execution path's own ``SHA_TP`` table, not reimplemented) -- this DOES
  hold, ruling out forcing-value/integer-time-index mismatch as the
  divergence source for these scenarios.
- ``test_*_first_temperature_divergence_is_pinned``: class (c)
  characterization, pinning exactly where (time_index, node) Python's
  own per-timestep temperature trajectory first differs from the live
  C++ diagnostic by more than a small, documented threshold, and by how
  much. NOT a tolerance/acceptance test.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import pytest

from pyfofem.components.soil_heating import (
    _campbell_ambient_radiation,
    _campbell_soil_depths_mm,
    _duff_burn_profile,
    _make_duff_forcing_fn,
    _make_nonduff_forcing_fn,
    _run_coupled_soil_sim,
    _soi_done_duff,
    _soi_done_nonduff,
    _soiltemp_initconsts,
    _soiltemp_initprofile,
    _soiltemp_step,
    _SOIL_FAMILY_DEFAULTS,
    _CAMPBELL_DUFF_TIMESTEP,
    _CAMPBELL_NONDUFF_TIMESTEP,
)
from tests.cpp_parity_live._harness_support import (
    HARNESS_EXE_OVERRIDE_ENV_VAR,
    HARNESS_SOIDIAG_EXE,
    ensure_soidiag_built,
    run_harness,
)
from tests.cpp_parity_live.test_cpp_harness_contract import (
    SOIL_CAMPBELL_DIAG_SUFFIXES,
    SOIL_CAMPBELL_HEADER,
    SOIL_CAMPBELL_SUFFIXES,
    SOIL_DIAG_NODES,
    _run_soil,
    _soil_diag_env,
    _soil_duff_row_with_burn_inputs,
    _soil_state_diag_env,
    _soil_zduff_row,
    _write_soil_side_files,
    soidiag_exe,
    toolchain_status,
)

pytestmark = pytest.mark.soil_solver

#: Node (fof_soi.cpp 1-based index) -> Python DataFrame/records column
#: index (0-based: Surface=0, 1cm=1, ..., 13cm=13). node n -> column n-1.
_NODE_TO_COLUMN = {n: n - 1 for n in SOIL_DIAG_NODES}

_FAMILY_MAP = {
    "Fine-Silt": "fine-silty", "Loamy-Skeletal": "loamy-skeletal",
    "Fine": "fine", "Coarse-Silt": "coarse-silty", "Coarse-Loamy": "coarse-loamy",
}

#: RE-PURPOSED 2026-09-18 (F-70 narrow numerical-compatibility pass,
#: third round). Originally a "skip float-formatting noise only"
#: threshold for a CHARACTERIZATION test that expected to find (and
#: pin) a real divergence "orders of magnitude larger" than this value.
#: F-70's third round found and fixed the real, isolated root cause (a
#: bulk_density/particle_density unit-conversion transcription error --
#: see gate0/04-findings.md F-70), so the tests below now use this SAME
#: constant as a genuine PARITY tolerance: no (time_index, node) pair
#: anywhere in the recorded trajectory may differ by more than this
#: amount. This is a real, evidence-based bound (measured max|diff| for
#: these scenarios is under 1e-4 degC -- see the module's own
#: measurement note below), not an invented placeholder, and it is NOT
#: loosened from its original value.
_FIRST_DIVERGENCE_THRESHOLD_C = 0.01


def _duff_forcing_and_records(soil_type, dep_pre_in, soil_moist_pct, load,
                               consumed, moist):
    """
    Run Python's OWN duff-route coupled solver directly (bypassing the
    public ``soil_heat_campbell()`` DataFrame wrapper) and return both the
    per-timestep ``(time_s, temps)`` records and the forcing function, so
    a test can compare per-node/per-timestep temperature AND per-timestep
    surface flux against the live C++ diagnostic.

    :param soil_type: C++ soil-family name (mapped to Python's own key).
    :param dep_pre_in: Pre-fire duff depth (in).
    :param soil_moist_pct: Starting soil moisture, percent.
    :param load: Duff load (T/ac).
    :param consumed: Duff consumed (%).
    :param moist: Duff moisture (%).
    :return: ``(records, forcing_fn)`` -- *records* as returned by
        :func:`pyfofem.components.soil_heating._run_coupled_soil_sim`.
    """
    props = dict(_SOIL_FAMILY_DEFAULTS[_FAMILY_MAP[soil_type]])
    start_temp = 21.0
    start_water = soil_moist_pct / 100.0
    duff_params = dict(duff_load=load, duff_depth=dep_pre_in,
                        duff_moisture=moist, pct_consumed=consumed)
    ambient = _campbell_ambient_radiation(start_temp)
    duff_profile = _duff_burn_profile(duff_params)
    forcing_fn = _make_duff_forcing_fn(duff_profile, ambient)
    z_mm = _campbell_soil_depths_mm(list(range(1, 14)))
    state = _soiltemp_initconsts(
        props["bulk_density"], props["particle_density"], props["k_mineral"],
        props["vries_shape"], props["recirc_water"], props["cop_power"],
        props["extrap_water"], z_mm,
    )
    _soiltemp_initprofile(state, start_water, start_temp)
    duration_s = duff_profile["duration_s"]

    def done_fn(temps, st_temp, clock_sec, still_burning):
        """Duff termination check bound to this scenario's burn duration."""
        return _soi_done_duff(temps, st_temp, clock_sec, duration_s)

    records = _run_coupled_soil_sim(state, _CAMPBELL_DUFF_TIMESTEP, forcing_fn,
                                     done_fn, start_temp)
    return records, forcing_fn


def _find_first_temp_divergence(cpp_rows, records, threshold_c):
    """
    Compare Python's per-(time_index, node) ``temp_tn`` against the live
    C++ diagnostic's "timestep" rows and return the first mismatch found,
    scanning in (time_index, node) order.

    :param cpp_rows: Parsed ``_soidiag.csv`` rows (only "timestep" ones
        are used).
    :param records: Python ``(time_s, temps)`` records from
        :func:`pyfofem.components.soil_heating._run_coupled_soil_sim`.
    :param threshold_c: Minimum absolute difference (degC) to count as a
        divergence (filters float-formatting noise only).
    :return: ``(time_index, node, py_temp, cpp_temp, diff)`` for the
        first mismatch, or ``None`` if none exceeds *threshold_c* within
        the overlapping time range.
    """
    ts_rows = [r for r in cpp_rows if r["record_kind"] == "timestep"]
    by_key = {(int(r["time_index"]), int(r["node_index"])): float(r["temp_tn_c"])
              for r in ts_rows}
    n_common = min(len(records), max(int(r["time_index"]) for r in ts_rows) + 1)
    for time_index in range(n_common):
        _, temps = records[time_index]
        for node in SOIL_DIAG_NODES:
            key = (time_index, node)
            if key not in by_key:
                continue
            py_val = float(temps[_NODE_TO_COLUMN[node]])
            cpp_val = by_key[key]
            diff = abs(py_val - cpp_val)
            if diff > threshold_c:
                return time_index, node, py_val, cpp_val, diff
    return None


def _run_duff_scenario_diag(tmp_path, case_id, soil_type, dep_pre, dep_pos,
                             soil_moist_pct, load, consumed, moist):
    """
    Drive the live C++ harness for one duff scenario with diagnostics
    enabled, and return its parsed ``_soidiag.csv`` rows.

    :param tmp_path: Scratch directory (pytest ``tmp_path``).
    :param case_id: Scenario identifier.
    :param soil_type: C++ soil-family name (e.g. ``"Coarse-Silt"``).
    :param dep_pre: Pre-fire duff depth (in), as a string.
    :param dep_pos: Post-fire duff depth (in), as a string (unused by the
        Python side, which derives its own post-depth from *consumed*).
    :param soil_moist_pct: Starting soil moisture, percent, as a string.
    :param load: Duff load (T/ac), as a string.
    :param consumed: Duff consumed (%), as a string.
    :param moist: Duff moisture (%), as a string.
    :return: List of parsed diagnostic row dicts.
    """
    row = _soil_duff_row_with_burn_inputs(case_id, load=load, consumed=consumed,
                                           moist=moist, dep_pre=dep_pre)
    row[3] = soil_type
    row[6] = dep_pos
    row[7] = soil_moist_pct
    env = _soil_diag_env(case_id)
    res = _run_soil([row], tmp_path, name=case_id, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows, "expected diagnostic rows"
    return rows


def _run_nonduff_scenario_diag(tmp_path, case_id, soil_type, soil_moist_pct):
    """
    Drive the live C++ harness for one non-duff (zero-duff) scenario with
    diagnostics enabled, and return its parsed ``_soidiag.csv`` rows.

    :param tmp_path: Scratch directory (pytest ``tmp_path``).
    :param case_id: Scenario identifier.
    :param soil_type: C++ soil-family name.
    :param soil_moist_pct: Starting soil moisture, percent, as a string.
    :return: List of parsed diagnostic row dicts.
    """
    row = _soil_zduff_row(case_id=case_id, soil_type=soil_type,
                          soil_moist_pct=soil_moist_pct)
    env = _soil_diag_env(case_id)
    res = _run_soil([row], tmp_path, name=case_id, env=env,
                     output_suffixes=SOIL_CAMPBELL_DIAG_SUFFIXES)
    assert res.returncode == 0, res.stderr
    rows = res.rows("_soidiag")
    assert rows, "expected diagnostic rows"
    return rows


def test_divergent_dry_duff_case_matches_cpp_within_tolerance(tmp_path):
    """Class (c), now genuine PARITY evidence (F-70 third round). SOI-DUF-04
    (Coarse-Silt, 5% soil moisture) -- the task's named
    previously-materially-divergent duff case. HISTORICAL: this test
    used to be named ``..._first_temperature_divergence_is_pinned`` and
    asserted that a real divergence existed and pinned its exact
    magnitude (~2.82 degC on the FIRST recorded timestep) -- the root
    cause (a ``bulk_density``/``particle_density`` unit-conversion
    transcription error in ``_soiltemp_step``'s own ``cp[i]`` term, see
    ``gate0/04-findings.md`` F-70) is now fixed, and NO (time_index,
    node) pair anywhere in the recorded trajectory differs by more than
    :data:`_FIRST_DIVERGENCE_THRESHOLD_C`. Measured directly: the
    largest actual |diff| for this scenario is ~8e-6 degC."""
    cpp_rows = _run_duff_scenario_diag(
        tmp_path, "divergent-duff-div", "Coarse-Silt", "2", "1", "5",
        load="5", consumed="50", moist="45",
    )
    records, _ = _duff_forcing_and_records(
        "Coarse-Silt", 2.0, 5.0, 5.0, 50.0, 45.0,
    )
    first = _find_first_temp_divergence(cpp_rows, records, _FIRST_DIVERGENCE_THRESHOLD_C)
    assert first is None, (
        f"expected genuine parity (no |diff| > {_FIRST_DIVERGENCE_THRESHOLD_C} "
        f"degC) for this scenario after the F-70 bulk_density/particle_density "
        f"fix; found a real divergence instead -- re-investigate before "
        f"re-pinning: {first}"
    )


def test_divergent_dry_duff_forcing_matches_cpp_exactly(tmp_path):
    """Class (b). SOI-DUF-04 (Coarse-Silt, 5% soil moisture): Python's
    own duff ``forcing_fn`` reproduces the live C++ per-tick
    ``surface_flux_w`` to within float-formatting precision, at EVERY
    recorded timestep -- rules out forcing-value or integer-time-index
    mismatch as the source of this scenario's large final divergence.
    The divergence must therefore originate inside the Newton solve
    itself (initialization, water-content/matric-potential relation,
    humidity/vapor-pressure relation, vapor-enhanced conductivity, or the
    nonlinear residual/derivative update) -- see the companion
    first-temperature-divergence test."""
    cpp_rows = _run_duff_scenario_diag(
        tmp_path, "divergent-duff-force", "Coarse-Silt", "2", "1", "5",
        load="5", consumed="50", moist="45",
    )
    _, forcing_fn = _duff_forcing_and_records(
        "Coarse-Silt", 2.0, 5.0, 5.0, 50.0, 45.0,
    )
    ts_rows = [r for r in cpp_rows if r["record_kind"] == "timestep"]
    checked = 0
    for r in ts_rows:
        if int(r["node_index"]) != SOIL_DIAG_NODES[0]:
            continue
        clock_sec = float(r["time_s"])
        cpp_flux = float(r["surface_flux_w"])
        py_flux, _ = forcing_fn(clock_sec)
        assert py_flux == pytest.approx(cpp_flux, abs=1e-2), (clock_sec, py_flux, cpp_flux)
        checked += 1
    assert checked > 0


def test_dry_nonduff_matches_cpp_within_tolerance(tmp_path):
    """Class (c), now genuine PARITY evidence (F-70 third round).
    HISTORICAL: this test used to be named
    ``..._first_temperature_divergence_is_pinned`` and asserted a real,
    ~46.96 degC first-timestep divergence for the SOI-NOD-04-like dry
    non-duff scenario -- the largest of the three required scenario
    categories, and the one whose per-sub-iteration trace isolated the
    root cause (see ``gate0/04-findings.md`` F-70). With that fixed, NO
    (time_index, node) pair anywhere in the recorded trajectory (1629
    timesteps) differs by more than :data:`_FIRST_DIVERGENCE_THRESHOLD_C`.
    Measured directly: the largest actual |diff| for this scenario is
    ~3.75e-4 degC."""
    _write_soil_side_files(tmp_path)
    n = 20
    wl_series = [max(0.0, 50.0 - i * 3.0) for i in range(n)]
    hs_series = [max(0.0, 10.0 - i * 0.5) for i in range(n)]
    cpp_rows = _run_nonduff_scenario_diag(
        tmp_path, "dry-nonduff-div", "Coarse-Silt", "5",
    )
    props = dict(_SOIL_FAMILY_DEFAULTS["coarse-silty"])
    start_temp = 21.0
    start_water = 0.05
    ambient = _campbell_ambient_radiation(start_temp)
    forcing_fn = _make_nonduff_forcing_fn(wl_series, hs_series, 0.15, 0.10, ambient)
    z_mm = _campbell_soil_depths_mm(list(range(1, 14)))
    state = _soiltemp_initconsts(
        props["bulk_density"], props["particle_density"], props["k_mineral"],
        props["vries_shape"], props["recirc_water"], props["cop_power"],
        props["extrap_water"], z_mm,
    )
    _soiltemp_initprofile(state, start_water, start_temp)
    records = _run_coupled_soil_sim(state, _CAMPBELL_NONDUFF_TIMESTEP, forcing_fn,
                                    _soi_done_nonduff, start_temp)

    first = _find_first_temp_divergence(cpp_rows, records, _FIRST_DIVERGENCE_THRESHOLD_C)
    assert first is None, (
        f"expected genuine parity (no |diff| > {_FIRST_DIVERGENCE_THRESHOLD_C} "
        f"degC) for this scenario after the F-70 bulk_density/particle_density "
        f"fix; found a real divergence instead -- re-investigate before "
        f"re-pinning: {first}"
    )


def test_dry_nonduff_forcing_matches_cpp_exactly(tmp_path):
    """Class (b). SOI-NOD-04-like (Coarse-Silt, 5% soil moisture, zero
    duff): Python's own non-duff ``forcing_fn`` reproduces the live C++
    per-tick ``surface_flux_w`` to within float-formatting precision --
    rules out forcing-value mismatch as this scenario's divergence
    source. Uses the harness's own default fire-intensity series
    (``_write_soil_side_files``'s formula) directly, matching what the
    C++ run actually consumed."""
    _write_soil_side_files(tmp_path)
    n = 20
    wl_series = [max(0.0, 50.0 - i * 3.0) for i in range(n)]
    hs_series = [max(0.0, 10.0 - i * 0.5) for i in range(n)]
    cpp_rows = _run_nonduff_scenario_diag(
        tmp_path, "dry-nonduff-force", "Coarse-Silt", "5",
    )
    start_temp = 21.0
    ambient = _campbell_ambient_radiation(start_temp)
    forcing_fn = _make_nonduff_forcing_fn(wl_series, hs_series, 0.15, 0.10, ambient)

    ts_rows = [r for r in cpp_rows if r["record_kind"] == "timestep"]
    checked = 0
    for r in ts_rows:
        if int(r["node_index"]) != SOIL_DIAG_NODES[0]:
            continue
        clock_sec = float(r["time_s"])
        cpp_flux = float(r["surface_flux_w"])
        py_flux, _ = forcing_fn(clock_sec)
        assert py_flux == pytest.approx(cpp_flux, abs=1e-2), (clock_sec, py_flux, cpp_flux)
        checked += 1
    assert checked > 0


def test_stable_duff_case_matches_cpp_within_tolerance(tmp_path):
    """Class (c), now genuine PARITY evidence (F-70 third round).
    SOI-DUF-02-like (Loamy-Skeletal, wetter soil). HISTORICAL: this test
    used to be named ``..._first_temperature_divergence_is_pinned`` and
    asserted a real, ~1.08 degC first-timestep divergence (previously
    the SMALLEST measured divergence of the six duff scenarios) -- with
    the F-70 root-cause fix, NO (time_index, node) pair anywhere in the
    recorded trajectory differs by more than
    :data:`_FIRST_DIVERGENCE_THRESHOLD_C`. Measured directly: the
    largest actual |diff| for this scenario is ~9e-6 degC."""
    cpp_rows = _run_duff_scenario_diag(
        tmp_path, "stable-duff-div", "Loamy-Skeletal", "3", "1.5", "20",
        load="8", consumed="40", moist="70",
    )
    records, _ = _duff_forcing_and_records(
        "Loamy-Skeletal", 3.0, 20.0, 8.0, 40.0, 70.0,
    )
    first = _find_first_temp_divergence(cpp_rows, records, _FIRST_DIVERGENCE_THRESHOLD_C)
    assert first is None, (
        f"expected genuine parity (no |diff| > {_FIRST_DIVERGENCE_THRESHOLD_C} "
        f"degC) for this scenario after the F-70 bulk_density/particle_density "
        f"fix; found a real divergence instead -- re-investigate before "
        f"re-pinning: {first}"
    )


def test_stable_duff_forcing_matches_cpp_exactly(tmp_path):
    """Class (b). SOI-DUF-02-like (Loamy-Skeletal): Python's own duff
    ``forcing_fn`` reproduces the live C++ per-tick ``surface_flux_w`` to
    within float-formatting precision -- rules out forcing-value
    mismatch for the "stable" scenario category too."""
    cpp_rows = _run_duff_scenario_diag(
        tmp_path, "stable-duff-force", "Loamy-Skeletal", "3", "1.5", "20",
        load="8", consumed="40", moist="70",
    )
    _, forcing_fn = _duff_forcing_and_records(
        "Loamy-Skeletal", 3.0, 20.0, 8.0, 40.0, 70.0,
    )
    ts_rows = [r for r in cpp_rows if r["record_kind"] == "timestep"]
    checked = 0
    for r in ts_rows:
        if int(r["node_index"]) != SOIL_DIAG_NODES[0]:
            continue
        clock_sec = float(r["time_s"])
        cpp_flux = float(r["surface_flux_w"])
        py_flux, _ = forcing_fn(clock_sec)
        assert py_flux == pytest.approx(cpp_flux, abs=1e-2), (clock_sec, py_flux, cpp_flux)
        checked += 1
    assert checked > 0


# ===========================================================================
# F-70 second diagnostic pass: per-Newton-sub-iteration comparison (the
# deeper facility this module needed once the per-TIMESTEP comparison
# above proved insufficient to localise the non-duff route's large
# first-timestep divergence -- see the class (c) test below for the
# full write-up of what was found and, just as importantly, what was
# RULED OUT). Requires the fofem_test_soidiag binary (soidiag_exe
# fixture, imported from test_cpp_harness_contract).
# ===========================================================================

def _python_first_subiter_trace(soil_type, soil_moist_pct):
    """
    Run Python's own ``_soiltemp_step`` directly for the FIRST timestep
    of the SOI-NOD-04-like dry non-duff scenario, using the SAME
    ``on_subiter`` diagnostic hook mechanism added to
    :func:`pyfofem.components.soil_heating._soiltemp_step` in this pass
    (mirroring the C++ overlay's ``SoiDiagRecordSubIteration`` exactly:
    same 5 fields, same call point in the loop).

    :param soil_type: C++ soil-family name (mapped to Python's own key).
    :param soil_moist_pct: Starting soil moisture, percent.
    :return: List of ``(n_subiter, tn1, p1, r_sev, r_seh)`` tuples, one
        per Newton sub-iteration of the first timestep.
    """
    props = dict(_SOIL_FAMILY_DEFAULTS[_FAMILY_MAP[soil_type]])
    start_temp = 21.0
    start_water = soil_moist_pct / 100.0
    n = 20
    wl_series = [max(0.0, 50.0 - i * 3.0) for i in range(n)]
    hs_series = [max(0.0, 10.0 - i * 0.5) for i in range(n)]
    ambient = _campbell_ambient_radiation(start_temp)
    forcing_fn = _make_nonduff_forcing_fn(wl_series, hs_series, 0.15, 0.10, ambient)
    z_mm = _campbell_soil_depths_mm(list(range(1, 14)))
    state = _soiltemp_initconsts(
        props["bulk_density"], props["particle_density"], props["k_mineral"],
        props["vries_shape"], props["recirc_water"], props["cop_power"],
        props["extrap_water"], z_mm,
    )
    _soiltemp_initprofile(state, start_water, start_temp)
    r_rabs, _ = forcing_fn(0.0)

    trace = []
    _soiltemp_step(state, r_rabs, _CAMPBELL_NONDUFF_TIMESTEP,
                    on_subiter=lambda *args: trace.append(args))
    return trace


def _python_first_surface_update_trace(soil_type, soil_moist_pct):
    """
    Run Python's own ``_soiltemp_step`` directly for the FIRST timestep
    of the SOI-NOD-04-like dry non-duff scenario, using the richer
    ``on_surface_update`` diagnostic hook (F-70 third round) -- mirrors
    the C++ overlay's ``SoiDiagRecordSurfaceUpdate`` exactly: same 48
    named fields, same call point in the loop.

    :param soil_type: C++ soil-family name (mapped to Python's own key).
    :param soil_moist_pct: Starting soil moisture, percent.
    :return: List of ``(n_subiter, fields_dict)`` tuples, one per Newton
        sub-iteration of the first timestep.
    """
    props = dict(_SOIL_FAMILY_DEFAULTS[_FAMILY_MAP[soil_type]])
    start_temp = 21.0
    start_water = soil_moist_pct / 100.0
    n = 20
    wl_series = [max(0.0, 50.0 - i * 3.0) for i in range(n)]
    hs_series = [max(0.0, 10.0 - i * 0.5) for i in range(n)]
    ambient = _campbell_ambient_radiation(start_temp)
    forcing_fn = _make_nonduff_forcing_fn(wl_series, hs_series, 0.15, 0.10, ambient)
    z_mm = _campbell_soil_depths_mm(list(range(1, 14)))
    state = _soiltemp_initconsts(
        props["bulk_density"], props["particle_density"], props["k_mineral"],
        props["vries_shape"], props["recirc_water"], props["cop_power"],
        props["extrap_water"], z_mm,
    )
    _soiltemp_initprofile(state, start_water, start_temp)
    r_rabs, _ = forcing_fn(0.0)

    trace = []
    _soiltemp_step(state, r_rabs, _CAMPBELL_NONDUFF_TIMESTEP,
                    on_surface_update=lambda n, d: trace.append((n, d)))
    return trace


def _run_nonduff_first_surfup_diag(tmp_path, soidiag_exe, monkeypatch,
                                    case_id, soil_type, soil_moist_pct):
    """
    Drive the live C++ diagnostic-observer binary for one non-duff
    scenario with ``FOFEM_TEST_SOIL_STATE_DIAG`` enabled, and return the
    first timestep's (``step_index == 0``) parsed ``_soisurfup.csv``
    rows, sorted by ``n_subiter``.
    """
    monkeypatch.setenv(HARNESS_EXE_OVERRIDE_ENV_VAR, soidiag_exe)
    _write_soil_side_files(tmp_path)
    row = _soil_zduff_row(case_id=case_id, soil_type=soil_type,
                           soil_moist_pct=soil_moist_pct)
    env = _soil_state_diag_env(case_id)
    env[HARNESS_EXE_OVERRIDE_ENV_VAR] = soidiag_exe
    res = run_harness(
        "soil_campbell", SOIL_CAMPBELL_HEADER, [row],
        str(tmp_path / case_id), env=env,
        output_suffixes=SOIL_CAMPBELL_SUFFIXES + ("_soistate", "_soisubiter", "_soisurfup"),
    )
    assert res.returncode == 0, res.stderr
    rows = [r for r in res.rows("_soisurfup") if r["step_index"] == "0"]
    assert rows, "expected first-timestep surface-update rows"
    return sorted(rows, key=lambda r: int(r["n_subiter"]))


#: Field-name mapping: C++ _soisurfup.csv column -> Python
#: on_surface_update dict key. Identical for every field except case
#: (the C++ CSV column names are already snake_case matching the
#: SoiSurfaceUpdateDiag/dict field names exactly) -- this dict exists so
#: a future rename on either side fails loudly (KeyError) rather than
#: silently comparing the wrong pair of columns.
_SURFUP_FIELD_MAP = {
    "old_tn1": "old_tn1", "new_tn1": "new_tn1", "old_p1": "old_p1", "new_p1": "new_p1",
    "old_wn1": "old_wn1", "new_wn1": "new_wn1", "old_h1": "old_h1", "new_h1": "new_h1",
    "psat0": "psat0", "h0": "h0", "psat1": "psat1", "psat2": "psat2", "h2": "h2",
    "s1": "s1", "hvap1": "hvap1", "kh1": "kh1", "enh1": "enh1", "kv1": "kv1",
    "kh2": "kh2", "kv2": "kv2", "ke0": "ke0", "ke1": "ke1", "kev0": "kev0", "kev1": "kev1",
    "conv1": "conv1", "vcon1": "vcon1", "cp1": "cp1",
    "d_jv": "d_jv", "d_jvdt": "d_jvdt", "d_jvdp": "d_jvdp",
    "dC_before_boundary": "dC_before_boundary", "dC_after_boundary": "dC_after_boundary",
    "dv": "dv", "dCdp": "dCdp", "dvdp": "dvdp",
    "dCdt_before_boundary": "dCdt_before_boundary", "dCdt_after_boundary": "dCdt_after_boundary",
    "dvdt": "dvdt", "r_rabs_in": "r_rabs_in", "stefan_term": "stefan_term", "tk_old": "tk_old",
    "dtn_temperature_raw": "dtn_temperature_raw",
    "dtn_temperature_clamped": "dtn_temperature_clamped",
    "dtn_matric_raw": "dtn_matric_raw",
    "p1_before_range_clamp": "p1_before_range_clamp", "p1_clamp_branch": "p1_clamp_branch",
    "r_sev_running": "r_sev_running", "r_seh_running": "r_seh_running",
}

#: Per-field absolute tolerance (degC-equivalent scale varies per field;
#: chosen generously relative to the measured ~1e-6-1e-4 agreement, never
#: tuned down to the smallest value that happens to pass). cp1/ke1/etc.
#: are O(1-1000), kv1/kev1/d_jvdp are O(1e-8-1e-11) -- a single global
#: absolute tolerance would either be meaninglessly loose for the large
#: fields or fail on legitimate float noise for the tiny ones, so this
#: uses `pytest.approx`'s combined `rel`+`abs` form per field instead of
#: a bespoke per-field table.
_SURFUP_REL_TOL = 1e-3
#: dC_before_boundary/dv are the PRE-Newton-update heat/water residuals
#: at the surface node before any correction is applied -- for a
#: uniform initial profile these are analytically near-zero (heavy
#: floating-point cancellation of terms that are each individually
#: O(1)-O(1e4)), so BOTH engines correctly compute a value near zero but
#: with real cross-language cancellation noise at up to ~0.005 in
#: absolute terms (measured directly: Python 8.5e-12, C++ -0.004567 for
#: this scenario's first sub-iteration) -- still 6+ orders of magnitude
#: below any of the genuinely large (O(1)-O(1e4)) fields this same
#: tolerance is also applied to, so this is not loosened to mask a real
#: bug.
_SURFUP_ABS_TOL = 0.01


def test_dry_nonduff_first_timestep_surface_update_matches_cpp_field_by_field(
        tmp_path, soidiag_exe, monkeypatch):
    """Class (c), now genuine PARITY evidence (F-70 third round) -- the
    deepest crosswalk this module performs. For the SOI-NOD-04-like dry
    non-duff scenario's FIRST timestep, compares EVERY ONE of the 48
    named quantities the surface-node Newton update reads or writes
    (old/new temperature and matric potential, water content, humidity,
    vapor pressure, every conductivity term, the boundary
    Stefan-Boltzmann correction before/after, both residuals, the
    Jacobian terms, both Newton increments, and any clamp branch taken)
    between Python's own ``on_surface_update`` hook and the live C++
    diagnostic's ``SoiDiagRecordSurfaceUpdate`` hook, for every Newton
    sub-iteration of the first timestep.

    HISTORICAL: this test used to be named
    ``test_dry_nonduff_first_subiteration_already_diverges_with_matched_inputs``
    and used only the 5-field ``on_subiter``/``_soisubiter.csv``
    facility to PIN a real ~34 degC first-sub-iteration divergence,
    concluding (at the time, correctly, given the evidence then
    available) that a near-singular Newton-update denominator made the
    result "extremely sensitive to sub-ULP differences ... too small to
    isolate as a single transcription error with the diagnostic tooling
    built in this pass". The richer 48-field ``on_surface_update``/
    ``SoiDiagRecordSurfaceUpdate`` facility THIS test now uses was built
    specifically to go one level deeper, and isolated the real, exact
    root cause: ``cp1`` (the ``cp[i]`` heat-capacity term) differed by a
    factor of ~6.09x between the two engines at sub-iteration 1 (Python
    105.035 vs C++ 639.550) -- traced to ``_SOIL_FAMILY_DEFAULTS``
    dividing ``bulk_density``/``particle_density`` by 1000 for a
    "module SI convention" that is harmless for the ratio ``xs=bd/pd``
    but corrupts ``cp[i]``'s own use of ``bd`` as an absolute value. See
    ``gate0/04-findings.md`` F-70 for the complete before/after field
    trace. With that fixed, every one of the 48 fields below matches to
    well within float32-vs-float64 precision noise, for every
    sub-iteration of the first timestep, not merely the final converged
    temperature."""
    cpp_rows = _run_nonduff_first_surfup_diag(
        tmp_path, soidiag_exe, monkeypatch, "surfup-parity", "Coarse-Silt", "5",
    )
    py_trace = _python_first_surface_update_trace("Coarse-Silt", 5.0)

    assert len(cpp_rows) == len(py_trace), (
        "sub-iteration COUNT itself differs -- a real regression, not a "
        "field-precision question", len(cpp_rows), len(py_trace),
    )

    fields_checked = set()
    for cpp_row, (py_n_subiter, py_fields) in zip(cpp_rows, py_trace):
        assert int(cpp_row["n_subiter"]) == py_n_subiter
        for cpp_field, py_field in _SURFUP_FIELD_MAP.items():
            cpp_val = float(cpp_row[cpp_field])
            py_val = float(py_fields[py_field])
            assert py_val == pytest.approx(cpp_val, rel=_SURFUP_REL_TOL, abs=_SURFUP_ABS_TOL), (
                py_n_subiter, cpp_field, py_val, cpp_val,
            )
            fields_checked.add(cpp_field)
    assert fields_checked == set(_SURFUP_FIELD_MAP), (
        "not every declared field was actually compared", fields_checked,
    )
