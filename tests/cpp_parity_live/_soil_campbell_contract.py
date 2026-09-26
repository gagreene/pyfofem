#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_soil_campbell_contract.py - "Soil Campbell" golden-dataset contract for the
``soil_campbell`` harness mode (scenario matrix, tolerance-policy routes,
and expected-divergence routes).

Phase 5 is the one dataset that adds its OWN new harness mode
(``soil_campbell``, Part 1) rather than reusing the six Phase 2 modes the
way Phase 4 does. Its scenario matrix is therefore built directly against
``SOIL_CAMPBELL_HEADER``/``SOIL_CAMPBELL_SUFFIXES``
(``test_cpp_harness_contract.py``), not against ``MODES[mode]`` — Part 1's
harness-contract self-tests deliberately keep ``soil_campbell`` OUT of the
shared ``MODES`` dict (see that module's "Mode: soil_campbell" section
header comment) because ``MODES`` is iterated at generator import time by
code (``generate_canonical_goldens.GOLDEN_TOLERANCE_KEYS``) scoped to the six
Phase 2 modes.

Why this lives in its own module rather than in ``_output_contract.py`` or
``_expanded_matrix_contract.py``: same reasoning as ``_expanded_matrix_contract.py``'s own
docstring — ``_output_contract.py`` is hashed into every Phase 2 manifest's
``generator_source_sha256``, and ``_expanded_matrix_contract.py`` is hashed into
every Phase 4 manifest's; a new dataset gets a new module so neither
existing dataset's generator-source digest moves.

Scenario-matrix design rule (same as Phase 4's, from the approved plan):
equivalence partitions and meaningful boundaries, never an indiscriminate
Cartesian product. Every scenario below names the
``gate0/07-branch-traceability.csv`` branch ID it exercises.

**CURRENT STATUS (2026-09-16/17, F-70) — supersedes the "Item-7" paragraph
immediately below as HISTORICAL.** By explicit user decision, F-52's own
"characterization is the permanent target" conclusion is no longer
acceptable: `soil_heat_campbell()` must eventually reproduce the pinned
C++ soil-temperature outputs. The SciPy-``solve_ivp``-driven heat-only
model the paragraph below describes was entirely REMOVED and replaced
with a coupled ``soiltemp_step`` Newton solver validated against C++'s reference behavior
(``pyfofem/components/soil_heating.py``); all 5 soil families now match
the pinned C++ table bit-for-bit (F-51 fully resolved, not just
"Coarse-Silt"). A 2026-09-17 diagnostic pass
(``test_soil_solver_diagnostic_comparison.py``, and ``test_soil_diag_*``
in ``test_cpp_harness_contract.py``) built real C++ intermediate-state
observability and located — but has not yet fixed — a real divergence
present from the very first Newton-converged timestep; forcing-value
mismatch was ruled out as the cause. Every ``duff``/``nonduff``
tolerance-policy route below STILL correctly reads ``"status":
"unverified"`` — this is now accurate because full parity genuinely has
not yet been demonstrated, not because it is being treated as a
permanently-acceptable target. See F-70 in ``gate0/04-findings.md`` for
the complete crosswalk and evidence.

**HISTORICAL — Item-7 scientific-risk note (F-17 / F-51 / F-52), describes
the REMOVED heat-only implementation, not the current one.** Python
integrated the
Campbell heat-conduction model with SciPy (``solve_ivp(method="Radau")``,
``soil_heating.py``); C++ performs a time-stepped nonlinear solve using
residual/derivative updates (with step reduction on non-convergence)
(``soiltemp_step``, ``fof_soi.cpp``) — a genuine scheme difference (F-17).
Separately, Python's
``_SOIL_FAMILY_DEFAULTS`` soil-property table (``soil_heating.py:31-72``)
does not match the pinned C++ ``sr_SE``/``sr_SD`` table
(``fof_se2.h``/``fof_sd2.h``) for 4 of the 5 named soil families — only
``Coarse-Silt``/``"coarse-silty"`` carries matching physical constants
(F-51). **F-52 (2026-09-03) corrects the scope of both**: C++'s
``soiltemp_step`` integrates coupled temperature, water pressure, humidity
and vapor state plus an ambient radiative floor and recirculation
parameters (``r_xwo``/``r_cop``) that Python's ``soil_heat_campbell()`` never
represents at all — measured directly at 16.1 degC max / 5.8 degC mean
divergence for Coarse-Silt even with every Python-consumed input
(``bulk_density``/``particle_density``/``k_mineral``/``vries_shape``) aligned
to the pinned C++ table. There is therefore no soil family, including
Coarse-Silt, for which a "scheme-only" comparison is currently well-posed.
Every ``duff``/``nonduff`` tolerance-policy route below is recorded
``"status": "unverified"`` (real null ``atol``/``rtol``) citing all three
findings — this dataset's C++-vs-Python comparisons are cross-implementation
characterization, not a parity claim, until a production decision is made
either to extend Python's solver or to formally accept the two as
intentionally different approximations (F-52's scope recommendation).

**F-53 (2026-09-03, CONFIRMED 2026-09-04, RESOLVED 2026-09-16): a separate,
distinct percent-to-ratio unit defect on the duff route, independent of
F-17/F-51/F-52's full-model divergence.** The pre-correction
``_duff_flux_and_duration()`` never converted the documented whole-percent
``duff_moisture`` input to the ratio scale its equation required
(Frandsen 1991's ``R_M`` and the pinned C++'s own ``fof_sd.cpp:100``
conversion both confirm a ratio, e.g. ``0.45``, not ``45.0``), so every
realistic ``duff_moisture`` value computed exactly zero surface flux. This
was resolved by the Campbell duff-forcing correction pass (see **F-69** in
``gate0/04-findings.md``, which bundles this fix together with the
SOI-01/SOI-02 forcing-shape defects): the new ``_duff_burn_profile()``
converts ``duff_moisture`` from percent to ratio exactly once, at its own
boundary. The desired-behaviour assertion that was previously a strict
xfail is now a real passing test,
``tests/unit/cpp/test_soil_campbell_characterization.py::test_duff_route_produces_positive_surface_forcing_at_realistic_moisture``.
This remains tracked as a DISTINCT route,
``soil_campbell.duff_moisture_unit`` (``status: "contract_only"``, null
``atol``/``rtol`` — never scored as C++ parity, since this route has no
golden-comparison scenario of its own), not folded into the ``duff``
route's full-model-comparison status above, which stays exactly as F-52
left it (F-51/F-52's structural divergence is a SEPARATE, still-open
divergence source, unaffected by F-53/F-69's fix — fixing the unit
conversion did not, and was never expected to, resolve the ``duff``/
``nonduff`` routes' own ``EXPECT-INVESTIGATE``/``"unverified"`` status).

Function order: module constants first, then top-level functions alphabetized
private-then-public, per AGENTS.md.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tests._support import PROJECT_ROOT, TEST_GOLDEN_DIR
from tests.cpp_parity_live._dataset_contract_base import make_dataset_helpers
from tests.cpp_parity_live._golden_manifest import (
    load_tolerance_policy,
    MODE_OUTPUT_SUFFIXES as _MODE_OUTPUT_SUFFIXES,
)
from tests.cpp_parity_live.test_cpp_harness_contract import (
    SOIL_CAMPBELL_FI_HS_NAME,
    SOIL_CAMPBELL_FI_WL_NAME,
    SOIL_CAMPBELL_HEADER,
    SOIL_CAMPBELL_N_STEPS,
    SOIL_CAMPBELL_SUFFIXES,
)

#: Root of the Phase 5 golden tree. Sibling of ``phase2/``/``phase4/``;
#: never merged with either, never regenerated by their generators.
GOLDEN_ROOT: str = os.path.join(TEST_GOLDEN_DIR, "soil_campbell")

#: Dataset identifier stamped into every Phase 5 manifest's ``dataset``
#: field.
DATASET_NAME: str = "soil_campbell"

#: The one Phase 5 harness mode.
SOIL_CAMPBELL_MODES: Tuple[str, ...] = ("soil_campbell",)

#: Output-file suffixes ``soil_campbell`` writes, imported through
#: ``_golden_manifest.MODE_OUTPUT_SUFFIXES`` rather than a second hardcoded
#: tuple, so this can never drift from what the generator actually
#: produces.
MODE_OUTPUT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    mode: tuple(_MODE_OUTPUT_SUFFIXES[mode]) for mode in SOIL_CAMPBELL_MODES
}
assert MODE_OUTPUT_SUFFIXES["soil_campbell"] == SOIL_CAMPBELL_SUFFIXES

#: Shared golden-file I/O helpers (see ``_dataset_contract_base.py``):
#: ``_required_golden_files``/``golden_dir``/``golden_manifest``/
#: ``golden_rows``/``missing_golden_files``/``require_golden_tree`` below
#: are thin aliases onto this, not independent implementations.
_HELPERS = make_dataset_helpers(
    golden_root=GOLDEN_ROOT,
    dataset_label="Soil Campbell",
    modes=SOIL_CAMPBELL_MODES,
    mode_output_suffixes=MODE_OUTPUT_SUFFIXES,
    generator_hint="tests/cpp_parity_live/generate_soil_campbell_goldens.py",
)

#: This dataset's own generation-time dependencies, hashed into every
#: manifest whenever the repo is dirty. Includes ``test_cpp_harness_contract.py``
#: (the source of ``SOIL_CAMPBELL_HEADER`` and the scenario rows'
#: side-file helpers) and ``generate_canonical_goldens.py`` (the shared
#: promotion/qualification/verification machinery this dataset's own
#: generator reuses verbatim rather than re-implementing).
GENERATOR_SOURCE_FILES_RELATIVE: Tuple[str, ...] = (
    "tests/cpp_parity_live/_golden_manifest.py",
    "tests/cpp_parity_live/_harness_support.py",
    "tests/cpp_parity_live/_soil_campbell_contract.py",
    "tests/cpp_parity_live/_proc.py",
    "tests/cpp_parity_live/generate_canonical_goldens.py",
    "tests/cpp_parity_live/generate_soil_campbell_goldens.py",
    "tests/cpp_parity_live/test_cpp_harness_contract.py",
    "tests/cpp_parity_live/tolerance_policy.json",
)

# ===========================================================================
# soil_campbell scenarios
#
# Driven directly through SH_Mngr (fof_sh.cpp:42). Column order is
# SOIL_CAMPBELL_HEADER (16 columns; Part 1's item-1 audit found the
# harness-contract-approved 13-column schema left the Duff route's
# f_DufLoaPre/f_DufConPer/f_DufMoi uninitialised — see that module's own
# header comment and MODE_SCHEMA_VERSIONS's soil_campbell docstring).
#
# Side files: every scenario below shares the SAME two fire-intensity
# series (SOIL_CAMPBELL_FI_WL_NAME / SOIL_CAMPBELL_FI_HS_NAME, written once
# per generation run by generate_soil_campbell_goldens._write_side_files) at
# SOIL_CAMPBELL_N_STEPS=20 steps — a short, clearly-decaying series
# (harness-contract section 7: neither SD_Mngr_New nor SE_Mngr_Array's
# stepping loop has a hard iteration cap of its own). Sharing one series
# keeps the matrix's routing/family/boundary partitions independent of an
# otherwise-arbitrary per-row fire-intensity choice.
#
# Soil-type partition: each of the 5 cr_SoilType values (fof_sh.h:72-76)
# appears exactly once in each route's first 5 scenarios; the 6th scenario
# in each route exercises a boundary (soil_moist_pct at e_SMV_Min/e_SMV_Max,
# fof_sh2.h:11-12, or an explicit non-sentinel efficiency override) rather
# than a 6th soil type, per the "equivalence partitions + meaningful
# boundaries, not Cartesian product" rule.
# ===========================================================================

#: (case_id, soil_type, moist_cond, soil_moist_pct, wl_efficiency,
#:  hs_efficiency) for the 6 BR-SOI-NODUFF (SOI-NOD-01..06) scenarios.
#: duff_dep_pre_in/duff_dep_pos_in are "0"/"0" (selects SE_Mngr_Array,
#: harness-contract section 7's route-selection rule); the 3 duff-only
#: trailing columns are the SD-route-only fields and take their
#: SI_Init-matching defaults (0 / -1 / 0) on this route.
_NODUFF_SCENARIOS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("SOI-NOD-01", "Fine-Silt", "Dry", "10", "-1", "-1"),
    ("SOI-NOD-02", "Loamy-Skeletal", "Wet", "20", "-1", "-1"),
    ("SOI-NOD-03", "Fine", "Moderate", "15", "-1", "-1"),
    ("SOI-NOD-04", "Coarse-Silt", "VeryDry", "5", "-1", "-1"),
    ("SOI-NOD-05", "Coarse-Loamy", "Dry", "12", "-1", "-1"),
    # Boundary + explicit-efficiency-override partition (soil_moist_pct at
    # e_SMV_Min=0.0, fof_sh2.h:12; non-sentinel wl/hs efficiencies instead
    # of the -1 "use SI_Init default" sentinel), on the one soil family
    # (Coarse-Silt) whose physical constants are confirmed identical
    # between C++ and Python (F-51) — the most credible target for a
    # future scheme-only comparison.
    ("SOI-NOD-06", "Coarse-Silt", "VeryDry", "0", "0.25", "0.05"),
)

#: (case_id, soil_type, moist_cond, duff_dep_pre_in, duff_dep_pos_in,
#:  soil_moist_pct, duff_load_tac, duff_consumed_pct, duff_moist_pct) for
#: the 6 BR-SOI-DUFF (SOI-DUF-01..06) scenarios. duff_dep_pre_in > 0
#: selects SD_Mngr_New.
_DUFF_SCENARIOS: Tuple[Tuple[str, str, str, str, str, str, str, str, str], ...] = (
    ("SOI-DUF-01", "Fine-Silt", "Dry", "2", "1", "10", "5", "50", "60"),
    ("SOI-DUF-02", "Loamy-Skeletal", "Wet", "3", "1.5", "20", "8", "40", "70"),
    ("SOI-DUF-03", "Fine", "Moderate", "1.5", "0.5", "15", "3", "60", "55"),
    ("SOI-DUF-04", "Coarse-Silt", "VeryDry", "2", "1", "5", "5", "50", "45"),
    ("SOI-DUF-05", "Coarse-Loamy", "Dry", "2.5", "1", "12", "6", "45", "50"),
    # Boundary partition: near-total duff consumption (duff_consumed_pct
    # close to 100, duff_dep_pos_in close to 0), on the F-51-comparable
    # Coarse-Silt family.
    ("SOI-DUF-06", "Coarse-Silt", "VeryDry", "2", "0.05", "5", "5", "97.5", "45"),
)

#: The single BR-SOI-NOIG (SOI-NOIG-01) scenario: brn_ignited=NO short-
#: circuits SH_Mngr before either route runs (fof_sh.cpp:50-53). CONTRACT-
#: ONLY (07-branch-traceability.csv): no Python soil_heat_campbell code path
#: models "burnup never ignited", so there is no oracle comparison to
#: attempt here, ever — not merely deferred pending F-51. wl/hs efficiency
#: are irrelevant on this path (SHA_Init_0() never reads them) so
#: :func:`soil_campbell_noig_row` does not take them.
_NOIG_SCENARIO: Tuple[str, str, str, str] = (
    "SOI-NOIG-01", "Fine-Silt", "Dry", "10")

#: Dotted ``soil_campbell.<route>`` policy/divergence keys this dataset
#: cites, one per BR-SOI-* branch. No ``_p5``-style dataset suffix here:
#: unlike the other 4 datasets (which reuse mode names shared across
#: datasets, e.g. ``consume``), ``soil_campbell`` is both the harness mode
#: AND the dataset name, so the plain mode name is already unambiguous.
SOIL_CAMPBELL_ROUTE_KEYS: Dict[str, Tuple[str, ...]] = {
    "soil_campbell": ("duff", "nonduff", "noig"),
}
SOIL_CAMPBELL_DIVERGENCE_KEYS: Dict[str, Tuple[str, ...]] = SOIL_CAMPBELL_ROUTE_KEYS

#: Correction-pass item-3 (2026-09-03): centrally-defined regression-
#: stability precision for characterization tests that re-assert a
#: PREVIOUSLY MEASURED Python-vs-C++ divergence (F-52) as a pinned
#: regression value. This is NOT a scientific parity tolerance -- every
#: ``soil_campbell`` route in ``tolerance_policy.json`` stays
#: ``"unverified"``/``"contract_only"`` with null ``atol``/``rtol``
#: regardless of this constant's value. It exists only so that ordinary
#: floating-point reproducibility noise (platform/library-version
#: differences in SciPy's ``solve_ivp``) does not spuriously fail a test
#: whose entire purpose is detecting a REAL change in the measured
#: divergence when either implementation's physics changes. Widening this
#: value to make a failing characterization test pass again is exactly the
#: "loosen the tolerance until green" anti-pattern F-52/F-51 explicitly
#: reject -- if a pinned value stops matching, re-measure and re-pin it
#: (update the literal expected value), never this constant.
CHARACTERIZATION_REGRESSION_PRECISION_DEGC: float = 0.05

#: REMOVED (Phase 5 correction pass part 3, 2026-09-03). Part 2 added a
#: ``CHARACTERIZATION_SANITY_ENVELOPE_DEGC = 150.0`` "physically sane
#: envelope" bound with a dedicated test, described as an independently
#: loose backstop. Independent review found it was not independent: it was
#: numerically EQUAL to ``SOI-DUF-06``'s own measured max divergence, i.e.
#: tuned to the observed result it claimed to police, not derived from any
#: physical/solver-domain source. No independently-derived bound for this
#: comparison exists in any bundled document, and inventing one would be an
#: equally unjustified tuned number. Deleted rather than replaced: every
#: characterization test already asserts ``np.isfinite(...)`` on its own
#: lane output AND pins the exact measured max/mean divergence via
#: ``CHARACTERIZATION_REGRESSION_PRECISION_DEGC`` above, so a genuine
#: blow-up already fails one of those two real checks without a separate
#: envelope. See F-53's sibling correction-pass note in
#: ``test_soil_campbell_characterization.py`` (tests/unit/cpp/) for the full
#: reasoning.


#: Shared golden-file I/O helpers - see ``_dataset_contract_base.py``'s
#: ``make_dataset_helpers`` for the real implementation; these names are
#: kept as module-level bindings because other test modules import them
#: by name from this module.
_required_golden_files = _HELPERS._required_golden_files
golden_dir = _HELPERS.golden_dir
golden_manifest = _HELPERS.golden_manifest
golden_rows = _HELPERS.golden_rows
missing_golden_files = _HELPERS.missing_golden_files


def soil_campbell_divergence_keys(mode: str) -> List[str]:
    """
    Return the dotted policy keys whose divergence status a Phase 5
    manifest for *mode* must document.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>.<route>`` keys in deterministic order (no
        dataset suffix — see :data:`SOIL_CAMPBELL_ROUTE_KEYS`'s docstring).
    :raises KeyError: If *mode* has no Phase 5 contract.
    """
    return [f"{mode}.{route}" for route in SOIL_CAMPBELL_DIVERGENCE_KEYS[mode]]


def soil_campbell_noduff_row(case_id: str, soil_type: str, moist_cond: str,
                       soil_moist_pct: str, wl_efficiency: str = "-1",
                       hs_efficiency: str = "-1") -> List[str]:
    """
    Build one BR-SOI-NODUFF input row in :data:`SOIL_CAMPBELL_HEADER` order.

    :param case_id: Scenario identifier (e.g. ``"SOI-NOD-01"``).
    :param soil_type: ``cr_SoilType`` value.
    :param moist_cond: ``cr_MoistCond`` value.
    :param soil_moist_pct: Soil moisture (percent, e_SMV_Min..e_SMV_Max).
    :param wl_efficiency: Wood-litter fire-intensity efficiency, or
        ``"-1"`` for the built-in default.
    :param hs_efficiency: Herb-shrub fire-intensity efficiency, or
        ``"-1"`` for the built-in default.
    :returns: A 16-field row list.
    """
    return [
        case_id, "0", "YES", soil_type, moist_cond, "0", "0", soil_moist_pct,
        wl_efficiency, hs_efficiency, str(SOIL_CAMPBELL_N_STEPS),
        SOIL_CAMPBELL_FI_WL_NAME, SOIL_CAMPBELL_FI_HS_NAME,
        "0", "-1", "0",
    ]


def soil_campbell_duff_row(case_id: str, soil_type: str, moist_cond: str,
                     duff_dep_pre_in: str, duff_dep_pos_in: str,
                     soil_moist_pct: str, duff_load_tac: str,
                     duff_consumed_pct: str, duff_moist_pct: str) -> List[str]:
    """
    Build one BR-SOI-DUFF input row in :data:`SOIL_CAMPBELL_HEADER` order.

    :param case_id: Scenario identifier (e.g. ``"SOI-DUF-01"``).
    :param soil_type: ``cr_SoilType`` value.
    :param moist_cond: ``cr_MoistCond`` value.
    :param duff_dep_pre_in: Pre-fire duff depth (in, > 0 selects
        ``SD_Mngr_New``).
    :param duff_dep_pos_in: Post-fire duff depth (in).
    :param soil_moist_pct: Soil moisture (percent).
    :param duff_load_tac: Duff load (T/ac; ``d_SI.f_DufLoaPre``).
    :param duff_consumed_pct: Duff consumed (percent; ``d_SI.f_DufConPer``).
    :param duff_moist_pct: Duff moisture (percent; ``d_SI.f_DufMoi``).
    :returns: A 16-field row list.
    """
    return [
        case_id, "0", "YES", soil_type, moist_cond, duff_dep_pre_in,
        duff_dep_pos_in, soil_moist_pct, "-1", "-1",
        str(SOIL_CAMPBELL_N_STEPS), SOIL_CAMPBELL_FI_WL_NAME,
        SOIL_CAMPBELL_FI_HS_NAME, duff_load_tac, duff_consumed_pct,
        duff_moist_pct,
    ]


def soil_campbell_noig_row(case_id: str, soil_type: str, moist_cond: str,
                     soil_moist_pct: str) -> List[str]:
    """
    Build the BR-SOI-NOIG input row (``brn_ignited="NO"``) in
    :data:`SOIL_CAMPBELL_HEADER` order.

    :param case_id: Scenario identifier (``"SOI-NOIG-01"``).
    :param soil_type: ``cr_SoilType`` value.
    :param moist_cond: ``cr_MoistCond`` value.
    :param soil_moist_pct: Soil moisture (percent).
    :returns: A 16-field row list.
    """
    return [
        case_id, "0", "NO", soil_type, moist_cond, "0", "0", soil_moist_pct,
        "-1", "-1", str(SOIL_CAMPBELL_N_STEPS), SOIL_CAMPBELL_FI_WL_NAME,
        SOIL_CAMPBELL_FI_HS_NAME, "0", "-1", "0",
    ]


def soil_campbell_policy_keys(mode: str) -> List[str]:
    """
    Return every tolerance-policy key applicable to *mode*'s Phase 5
    golden.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>.<route>`` keys in deterministic order (no
        dataset suffix — see :data:`SOIL_CAMPBELL_ROUTE_KEYS`'s docstring).
    :raises KeyError: If *mode* has no Phase 5 contract.
    """
    return [f"{mode}.{route}" for route in SOIL_CAMPBELL_ROUTE_KEYS[mode]]


def soil_campbell_rows(mode: str) -> List[List[str]]:
    """
    Build the complete, ordered Phase 5 input rows for *mode*.

    :param mode: Harness mode name.
    :returns: 13 rows (6 BR-SOI-NODUFF + 6 BR-SOI-DUFF + 1 BR-SOI-NOIG), in
        :data:`SOIL_CAMPBELL_HEADER` order.
    :raises KeyError: If *mode* is not one of :data:`SOIL_CAMPBELL_MODES`.
    """
    if mode != "soil_campbell":
        raise KeyError(f"unknown Phase 5 mode: {mode!r}")
    rows = [
        soil_campbell_noduff_row(case_id, soil_type, moist_cond, soil_moist_pct,
                           wl_eff, hs_eff)
        for case_id, soil_type, moist_cond, soil_moist_pct, wl_eff, hs_eff
        in _NODUFF_SCENARIOS
    ]
    rows.extend(
        soil_campbell_duff_row(case_id, soil_type, moist_cond, dep_pre, dep_pos,
                         soil_moist_pct, load_tac, consumed_pct, moist_pct)
        for case_id, soil_type, moist_cond, dep_pre, dep_pos, soil_moist_pct,
            load_tac, consumed_pct, moist_pct in _DUFF_SCENARIOS
    )
    rows.append(soil_campbell_noig_row(*_NOIG_SCENARIO))
    return rows


def soil_campbell_scenario_case_ids(mode: str) -> List[str]:
    """
    Return every scenario ``case_id`` for *mode*, in :func:`soil_campbell_rows`
    order.

    :param mode: Harness mode name.
    :returns: 13 case IDs.
    :raises KeyError: If *mode* is not one of :data:`SOIL_CAMPBELL_MODES`.
    """
    return [row[0] for row in soil_campbell_rows(mode)]


require_golden_tree = _HELPERS.require_golden_tree
