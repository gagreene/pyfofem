#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_phase5_contract.py - Phase 5 golden-dataset contract for the
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
code (``generate_phase2_goldens.GOLDEN_TOLERANCE_KEYS``) scoped to the six
Phase 2 modes.

Why this lives in its own module rather than in ``_output_contract.py`` or
``_phase4_contract.py``: same reasoning as ``_phase4_contract.py``'s own
docstring — ``_output_contract.py`` is hashed into every Phase 2 manifest's
``generator_source_sha256``, and ``_phase4_contract.py`` is hashed into
every Phase 4 manifest's; a new dataset gets a new module so neither
existing dataset's generator-source digest moves.

Scenario-matrix design rule (same as Phase 4's, from the approved plan):
equivalence partitions and meaningful boundaries, never an indiscriminate
Cartesian product. Every scenario below names the
``gate0/07-branch-traceability.csv`` branch ID it exercises.

**Item-7 scientific-risk note (F-17 / F-51 / F-52).** Python integrates the
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

**F-53 (2026-09-03, CONFIRMED 2026-09-04): a separate, distinct
percent-to-ratio unit defect on the duff route, independent of F-17/F-51/
F-52's full-model divergence.** ``_duff_flux_and_duration()`` never
converts the documented whole-percent ``duff_moisture`` input to the ratio
scale its equation requires (Frandsen 1991's ``R_M`` and the pinned C++'s
own ``fof_sd.cpp:100`` conversion both confirm a ratio, e.g. ``0.45``, not
``45.0``), so every realistic ``duff_moisture`` value computes exactly zero
surface flux. This is CONFIRMED current defective behaviour, not an open
scientific question — see F-53 in ``gate0/04-findings.md`` for the full
evidence chain and
``tests/unit/test_phase5_soil_campbell_characterization.py::test_duff_route_should_produce_positive_surface_forcing_at_realistic_moisture``
for the strict-xfail pin of the desired behaviour. This is tracked as a
DISTINCT contract-defect route,
``soil_campbell_p5.duff_moisture_unit`` (``status:
"known_divergent_strict_xfail"``, null ``atol``/``rtol`` — never scored as
C++ parity), not folded into the ``duff`` route's full-model-comparison
status above, which stays exactly as F-52 left it (F-51/F-52's structural
divergence would remain even after F-53's unit conversion is fixed).

Function order: module constants first, then top-level functions alphabetized
private-then-public, per AGENTS.md.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tests._support import PROJECT_ROOT, TEST_GOLDEN_DIR
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
GOLDEN_ROOT: str = os.path.join(TEST_GOLDEN_DIR, "phase5")

#: Dataset identifier stamped into every Phase 5 manifest's ``dataset``
#: field.
DATASET_NAME: str = "phase5"

#: The one Phase 5 harness mode.
PHASE5_MODES: Tuple[str, ...] = ("soil_campbell",)

#: Output-file suffixes ``soil_campbell`` writes, imported through
#: ``_golden_manifest.MODE_OUTPUT_SUFFIXES`` rather than a second hardcoded
#: tuple, so this can never drift from what the generator actually
#: produces.
MODE_OUTPUT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    mode: tuple(_MODE_OUTPUT_SUFFIXES[mode]) for mode in PHASE5_MODES
}
assert MODE_OUTPUT_SUFFIXES["soil_campbell"] == SOIL_CAMPBELL_SUFFIXES

#: This dataset's own generation-time dependencies, hashed into every
#: manifest whenever the repo is dirty. Includes ``test_cpp_harness_contract.py``
#: (the source of ``SOIL_CAMPBELL_HEADER`` and the scenario rows'
#: side-file helpers) and ``generate_phase2_goldens.py`` (the shared
#: promotion/qualification/verification machinery this dataset's own
#: generator reuses verbatim rather than re-implementing).
GENERATOR_SOURCE_FILES_RELATIVE: Tuple[str, ...] = (
    "tests/cpp_parity_live/_golden_manifest.py",
    "tests/cpp_parity_live/_harness_support.py",
    "tests/cpp_parity_live/_phase5_contract.py",
    "tests/cpp_parity_live/_proc.py",
    "tests/cpp_parity_live/generate_phase2_goldens.py",
    "tests/cpp_parity_live/generate_phase5_goldens.py",
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
# per generation run by generate_phase5_goldens._write_side_files) at
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
#: :func:`phase5_noig_row` does not take them.
_NOIG_SCENARIO: Tuple[str, str, str, str] = (
    "SOI-NOIG-01", "Fine-Silt", "Dry", "10")

#: Dotted ``soil_campbell_p5.<route>`` policy/divergence keys this dataset
#: cites, one per BR-SOI-* branch.
PHASE5_ROUTE_KEYS: Dict[str, Tuple[str, ...]] = {
    "soil_campbell": ("duff", "nonduff", "noig"),
}
PHASE5_DIVERGENCE_KEYS: Dict[str, Tuple[str, ...]] = PHASE5_ROUTE_KEYS

#: Correction-pass item-3 (2026-09-03): centrally-defined regression-
#: stability precision for characterization tests that re-assert a
#: PREVIOUSLY MEASURED Python-vs-C++ divergence (F-52) as a pinned
#: regression value. This is NOT a scientific parity tolerance -- every
#: ``soil_campbell_p5`` route in ``tolerance_policy.json`` stays
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
#: ``test_phase5_soil_campbell_characterization.py`` for the full
#: reasoning.


def _required_golden_files(mode: str) -> List[str]:
    """
    Return the absolute paths every committed Phase 5 golden for *mode*
    must contain.

    :param mode: Harness mode name.
    :returns: Absolute paths to the mode's manifest, input CSV, and every
        declared output CSV.
    :raises KeyError: If *mode* is not one of :data:`PHASE5_MODES`.
    """
    directory = golden_dir(mode)
    paths = [
        os.path.join(directory, f"{mode}.manifest.json"),
        os.path.join(directory, f"{mode}_in.csv"),
    ]
    paths.extend(
        os.path.join(directory, f"{mode}{suffix}.csv")
        for suffix in MODE_OUTPUT_SUFFIXES[mode]
    )
    return paths


def golden_dir(mode: str) -> str:
    """
    Return the committed Phase 5 golden directory for *mode*.

    :param mode: Harness mode name.
    :returns: Absolute path (may not exist if the dataset was never
        generated in this checkout).
    """
    return os.path.join(GOLDEN_ROOT, mode)


def golden_manifest(mode: str) -> Optional[Dict[str, Any]]:
    """
    Load *mode*'s committed Phase 5 manifest.

    :param mode: Harness mode name.
    :returns: The parsed manifest, or ``None`` if the file does not exist.
    """
    path = os.path.join(golden_dir(mode), f"{mode}.manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def golden_rows(mode: str, suffix: str = "") -> List[Dict[str, str]]:
    """
    Read a committed Phase 5 golden output CSV as a list of row dicts.

    :param mode: Harness mode name.
    :param suffix: Output-file suffix (e.g. ``"_summary"``/``"_field"``).
    :returns: Every data row, as ``{column: raw string value}``.
    :raises FileNotFoundError: If the golden file is absent.
    """
    path = os.path.join(golden_dir(mode), f"{mode}{suffix}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def missing_golden_files() -> List[str]:
    """
    Return every required Phase 5 golden file that is absent or empty.

    :returns: Repo-relative paths, sorted, of every required file that
        does not exist or exists with zero bytes.
    """
    missing = []
    for mode in PHASE5_MODES:
        for path in _required_golden_files(mode):
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing.append(
                    os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
                )
    return sorted(missing)


def phase5_divergence_keys(mode: str) -> List[str]:
    """
    Return the dotted policy keys whose divergence status a Phase 5
    manifest for *mode* must document.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>_p5.<route>`` keys in deterministic order.
    :raises KeyError: If *mode* has no Phase 5 contract.
    """
    return [f"{mode}_p5.{route}" for route in PHASE5_DIVERGENCE_KEYS[mode]]


def phase5_noduff_row(case_id: str, soil_type: str, moist_cond: str,
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


def phase5_duff_row(case_id: str, soil_type: str, moist_cond: str,
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


def phase5_noig_row(case_id: str, soil_type: str, moist_cond: str,
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


def phase5_policy_keys(mode: str) -> List[str]:
    """
    Return every tolerance-policy key applicable to *mode*'s Phase 5
    golden.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>_p5.<route>`` keys in deterministic order.
    :raises KeyError: If *mode* has no Phase 5 contract.
    """
    return [f"{mode}_p5.{route}" for route in PHASE5_ROUTE_KEYS[mode]]


def phase5_rows(mode: str) -> List[List[str]]:
    """
    Build the complete, ordered Phase 5 input rows for *mode*.

    :param mode: Harness mode name.
    :returns: 13 rows (6 BR-SOI-NODUFF + 6 BR-SOI-DUFF + 1 BR-SOI-NOIG), in
        :data:`SOIL_CAMPBELL_HEADER` order.
    :raises KeyError: If *mode* is not one of :data:`PHASE5_MODES`.
    """
    if mode != "soil_campbell":
        raise KeyError(f"unknown Phase 5 mode: {mode!r}")
    rows = [
        phase5_noduff_row(case_id, soil_type, moist_cond, soil_moist_pct,
                           wl_eff, hs_eff)
        for case_id, soil_type, moist_cond, soil_moist_pct, wl_eff, hs_eff
        in _NODUFF_SCENARIOS
    ]
    rows.extend(
        phase5_duff_row(case_id, soil_type, moist_cond, dep_pre, dep_pos,
                         soil_moist_pct, load_tac, consumed_pct, moist_pct)
        for case_id, soil_type, moist_cond, dep_pre, dep_pos, soil_moist_pct,
            load_tac, consumed_pct, moist_pct in _DUFF_SCENARIOS
    )
    rows.append(phase5_noig_row(*_NOIG_SCENARIO))
    return rows


def phase5_scenario_case_ids(mode: str) -> List[str]:
    """
    Return every scenario ``case_id`` for *mode*, in :func:`phase5_rows`
    order.

    :param mode: Harness mode name.
    :returns: 13 case IDs.
    :raises KeyError: If *mode* is not one of :data:`PHASE5_MODES`.
    """
    return [row[0] for row in phase5_rows(mode)]


def require_golden_tree() -> None:
    """
    Fail closed unless the complete committed Phase 5 golden dataset is
    present.

    :returns: None.
    :raises FileNotFoundError: If any required file is missing or empty.
    """
    missing = missing_golden_files()
    if missing:
        raise FileNotFoundError(
            "the committed Phase 5 golden dataset is incomplete - this is a "
            "repository defect, not a skippable environment difference. "
            "Missing or empty:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nRestore them from git, or regenerate with "
              "tests/cpp_parity_live/generate_phase5_goldens.py (needs the "
              "live MSVC/CMake/Ninja toolchain)."
        )
