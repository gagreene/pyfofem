#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_burnup_extended_contract.py - "Burnup Extended" golden-dataset contract:
additional ``run_burnup``/``burnup`` scenarios beyond existing coverage,
selected from direct Python (``burnup_calcs.py``/``burnup.py``) and
pinned-C++ (``bur_brn.cpp``) control-flow analysis.

Builds **no new harness mode**: every scenario is driven through the
already-qualified ``consume`` mode's existing, unchanged input schema
(the same fire-environment/fuel-loading fields Phase 2/4/6 already use),
exactly as ``_emissions_equivalence_contract.py`` reuses it. This module lives on its
own (not folded into ``_expanded_matrix_contract.py``/``_emissions_equivalence_contract.py``) so
editing it cannot move either of their committed manifests'
``generator_source_sha256``.

**Scenario selection rationale (verified by direct probe execution
against the live harness before being committed to this module, not
assumed).** Every committed Phase 2/4/6 ``consume`` golden row uses
``windspeed_m_s=2``, ``ambient_temp_c=20``, ``ig_time_s=60`` uniformly,
and populates only the AGGREGATE ``dw1000_tac``/``pct_rot`` columns,
never the granular ``snd_dw*_tac``/``rot_dw*_tac`` size-class columns -
confirmed by a direct scan of every committed Phase 2/4/6 ``consume_in.csv``
ahead of writing this module. The four scenarios below were chosen to
exercise real, previously-unmanifested branches in
``_run_burnup_cell()``/``burnup()``:

1. ``rot-snd-mix`` - populates the granular ``snd_dw20_tac``/
   ``rot_dw20_tac`` size-class columns simultaneously (both > 0), forcing
   ``_run_burnup_cell()``'s per-particle loop to construct BOTH a
   ``dwk_20`` (sound, ``tpig=_SOUND_TPIG=327``) and a ``dwk_20_r``
   (rotten, ``tpig=_ROTTEN_TPIG=302``) :class:`FuelParticle` in the SAME
   simulation - never exercised by any prior committed golden, which
   only ever populate the undifferentiated aggregate ``dw1000_tac``.
2. ``calm-wind`` - ``windspeed_m_s=0``, the EXACT lower boundary of
   ``_FIRE_BOUNDS['u']`` (``(0.0, 5.0)``) inside a full stateful burnup
   simulation (as opposed to item B's isolated-function boundary
   characterization) - every prior committed golden uses a nonzero
   windspeed.
3. ``hot-amb-duff`` - ``ambient_temp_c=39.9``, just inside the upper
   boundary of ``_FIRE_BOUNDS['tamb_c']`` (``(-40.0, 40.0)``), combined
   with duff present (``duff_tac=10.0`` unchanged from the base row) -
   every prior committed golden uses ``ambient_temp_c=20``.
4. ``long-igtime`` - ``ig_time_s=199.9``, just inside the upper boundary
   of ``_FIRE_BOUNDS['ti']`` (``(10.0, 200.0)``) - every prior committed
   golden uses ``ig_time_s=60``.

Each was confirmed, by a direct live-harness probe run ahead of writing
this module, to produce ``_summary`` output genuinely different from the
unmodified base row across at least one of ``SndDW1kCon``/``RotDW1kCon``/
``FlaDur``/``SmoDur``/``TotCon`` - i.e. each is scientifically
discriminating, not merely a relabeled duplicate of existing coverage.

Function order: module constants first, then top-level functions
alphabetized private-then-public, per AGENTS.md.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tests._support import PROJECT_ROOT, TEST_GOLDEN_DIR
from tests.cpp_parity_live._dataset_contract_base import make_dataset_helpers
from tests.cpp_parity_live._golden_manifest import (
    load_tolerance_policy,
    MODE_OUTPUT_SUFFIXES as _MODE_OUTPUT_SUFFIXES,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES
GOLDEN_ROOT: str = os.path.join(TEST_GOLDEN_DIR, "burnup_extended")
DATASET_NAME: str = "burnup_extended"
BURNUP_EXTENDED_MODES: Tuple[str, ...] = ("consume",)
CONSUME_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(MODES["consume"]["header"])
}
MODE_OUTPUT_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    mode: tuple(_MODE_OUTPUT_SUFFIXES[mode]) for mode in BURNUP_EXTENDED_MODES
}
#: Shared golden-file I/O helpers (see ``_dataset_contract_base.py``):
#: ``_required_golden_files``/``golden_dir``/``golden_manifest``/
#: ``golden_rows``/``golden_rows_by_case``/``missing_golden_files``/
#: ``require_golden_tree`` below are thin aliases onto this, not
#: independent implementations.
_HELPERS = make_dataset_helpers(
    golden_root=GOLDEN_ROOT,
    dataset_label="Burnup Extended",
    modes=BURNUP_EXTENDED_MODES,
    mode_output_suffixes=MODE_OUTPUT_SUFFIXES,
    generator_hint="tests/cpp_parity_live/generate_burnup_extended_goldens.py",
)
GENERATOR_SOURCE_FILES_RELATIVE: Tuple[str, ...] = (
    "tests/cpp_parity_live/_golden_manifest.py",
    "tests/cpp_parity_live/_harness_support.py",
    "tests/cpp_parity_live/_burnup_extended_contract.py",
    "tests/cpp_parity_live/_proc.py",
    "tests/cpp_parity_live/_scratch.py",
    "tests/cpp_parity_live/generate_canonical_goldens.py",
    "tests/cpp_parity_live/generate_burnup_extended_goldens.py",
    "tests/cpp_parity_live/test_cpp_harness_contract.py",
    "tests/cpp_parity_live/tolerance_policy.json",
)
CONSUME_SCENARIOS: Tuple[Tuple[str, Dict[str, str], str], ...] = (
    # rot-snd-mix exercises a genuinely distinct control-flow branch (the
    # per-particle-loop `is_rotten` conditional in _run_burnup_cell(),
    # constructing BOTH a sound and a rotten 1000hr particle in one
    # simulation) and has its own traceability row, BR-BUP-ROT-SND-MIX.
    # The other three are fire-environment-boundary VALUE variations of
    # the SAME whole-model nominal-burnup branch the pre-existing
    # BR-BRN-NOMINAL row already covers, so they extend that row's own
    # scenario_id list rather than each getting a new row - see
    # gate0/07-branch-traceability.csv.
    #
    # Every scenario also overrides duff_moist_method to "ENTIRE"
    # (Phase 7 correction pass, 2026-09-06, item 3): the canonical base
    # row this dataset started from uses "NFDR" - inherited unchanged,
    # never a deliberate target of any of these 4 scenarios (none of them
    # are about duff-moisture-method routing) - but
    # run_fofem_emissions() has NO parameter to select "NFDR" at all: it
    # always computes duff consumption via consm_duff(duff_moist_cat=
    # "edm"), Python's own spelling of C++'s ENTIRE route (see F-22's
    # "'edm' is the C++ ENTIRE case" note in gate0/04-findings.md).
    # Comparing a NFDR-generated golden against an ENTIRE-computed Python
    # call is not a fair like-for-like comparison; overriding these
    # scenarios' own duff_moist_method to ENTIRE (which none of them
    # target) makes DufCon and every field it feeds into directly
    # comparable. See F-60/F-61 in gate0/04-findings.md for the full
    # investigation (this was originally mis-attributed to F-23, which
    # is Northeast-only and does not apply to these InteriorWest
    # scenarios at all - retracted, see BR-BUP-ROT-SND-MIX's own notes).
    ("rot-snd-mix", {
        "dw1000_tac": "0", "pct_rot": "0",
        "snd_dw20_tac": "1.5", "rot_dw20_tac": "1.5",
        "duff_moist_method": "ENTIRE",
    }, "BR-BUP-ROT-SND-MIX"),
    ("calm-wind", {
        "windspeed_m_s": "0", "duff_moist_method": "ENTIRE",
    }, "BR-BRN-NOMINAL"),
    ("hot-amb-duff", {
        "ambient_temp_c": "39.9", "duff_moist_method": "ENTIRE",
    }, "BR-BRN-NOMINAL"),
    ("long-igtime", {
        "ig_time_s": "199.9", "duff_moist_method": "ENTIRE",
    }, "BR-BRN-NOMINAL"),
)


def _apply(
        base: Sequence[str],
        index: Dict[str, int],
        case_id: str,
        overrides: Dict[str, str],
) -> List[str]:
    """
    Return a copy of *base* with ``case_id`` and *overrides* applied.

    :param base: The mode's canonical all-ok row (field values, in schema
        order).
    :param index: Column-name to position map for that mode's schema.
    :param case_id: Value for the row's ``case_id`` column.
    :param overrides: Column-name to value pairs to substitute.
    :returns: A new field list of the same length as *base*.
    :raises KeyError: If an override names a column the schema does not have
        - fail-closed, so a typo in a scenario definition cannot silently
        produce a row that tests nothing.
    """
    row = list(base)
    row[index["case_id"]] = case_id
    for column, value in overrides.items():
        row[index[column]] = value
    return row


#: Shared golden-file I/O helpers - see ``_dataset_contract_base.py``'s
#: ``make_dataset_helpers`` for the real implementation; these names are
#: kept as module-level bindings because other test modules import them
#: by name from this module.
_required_golden_files = _HELPERS._required_golden_files
golden_dir = _HELPERS.golden_dir
golden_manifest = _HELPERS.golden_manifest
golden_rows = _HELPERS.golden_rows
golden_rows_by_case = _HELPERS.golden_rows_by_case
missing_golden_files = _HELPERS.missing_golden_files


def burnup_extended_divergence_keys(mode: str) -> List[str]:
    """
    Return the dotted policy keys whose divergence status a Phase 7
    manifest for *mode* must document.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>_burnup_extended.<route>`` keys, deterministic order.
    :raises KeyError: If *mode* is not one of :data:`BURNUP_EXTENDED_MODES`.
    """
    return burnup_extended_policy_keys(mode)


#: The complete, fixed set of ``consume_burnup_extended`` tolerance-policy routes
#: (Phase 7 correction pass item 2, 2026-09-06). Unlike the original
#: Phase 7 pass's per-scenario keys (one key per ``case_id``, covering
#: only ``FlaDur``), these are per FIELD-GROUP - the same convention
#: ``consume_expanded_matrix`` already uses - so every one of the 66 real scientific
#: columns in the committed ``consume_summary.csv`` (derived from the
#: real header via ``_output_contract.classify_columns``, not a second
#: hardcoded list) has an explicit, applicable CLASSIFICATION.
#:
#: Phase 7 correction pass (2026-09-07) item 5: having a classification
#: is NOT the same as having an executed numeric comparison - most
#: routes are ``status="verified"``/``"known_divergent_strict_xfail"``
#: and ARE numerically compared in ``test_burnup_extended_parity.py`` (tests/unit/cpp/),
#: but ``duff_percent`` and ``emissions_mismatched_group`` are
#: ``status="unverified"`` and deliberately have NO numeric comparison:
#: ``duff_percent`` because ``run_fofem_emissions()`` exposes no
#: matching ``DufPer``-equivalent field at all, and
#: ``emissions_mismatched_group`` because all 4 Phase 7 scenarios use
#: intentionally mismatched flame/smolder/duff emission-factor groups
#: (inherited unchanged from the Phase 2 canonical base row). See
#: ``tolerance_policy.json``'s ``consume_burnup_extended`` section for each route's
#: own ``status`` field and measured evidence, and
#: ``test_burnup_extended_parity.py`` (tests/unit/cpp/) for exactly which routes are
#: actually exercised by an executed comparison.
BURNUP_EXTENDED_POLICY_ROUTES: Tuple[str, ...] = (
    "woody_small",
    "dw100",
    "woody_1000hr",
    "duff",
    "duff_percent",
    "totals",
    "fladur",
    "flame_smolder_consumption_nominal",
    "flame_smolder_consumption_affected_scenarios",
    "smoldering_duration_nominal",
    "smoldering_duration_affected_scenarios",
    "mineral_soil",
    "emissions_mismatched_group",
)


def burnup_extended_policy_keys(mode: str) -> List[str]:
    """
    Return every tolerance-policy key applicable to *mode*'s Phase 7 golden.

    :param mode: Harness mode name.
    :returns: Dotted ``<mode>_burnup_extended.<route>`` keys, deterministic order.
    :raises KeyError: If *mode* is not one of :data:`BURNUP_EXTENDED_MODES`.
    """
    if mode != "consume":
        raise KeyError(f"unknown Phase 7 mode: {mode!r}")
    return [f"consume_burnup_extended.{route}" for route in BURNUP_EXTENDED_POLICY_ROUTES]


def burnup_extended_rows(mode: str) -> List[List[str]]:
    """
    Build the complete, ordered Phase 7 input rows for *mode*.

    :param mode: Harness mode name.
    :returns: One field list per scenario, in the mode's schema order.
    :raises KeyError: If *mode* is not one of :data:`BURNUP_EXTENDED_MODES`.
    """
    if mode != "consume":
        raise KeyError(f"unknown Phase 7 mode: {mode!r}")
    base = MODES["consume"]["row"]
    return [
        _apply(base, CONSUME_INDEX, case_id, overrides)
        for case_id, overrides, _branches in CONSUME_SCENARIOS
    ]


def burnup_extended_tolerance(mode: str, route: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return the centrally recorded ``(atol, rtol)`` for one Phase 7 output
    route.

    This is the ONLY way a Phase 7 parity test may obtain a tolerance for a
    Python-vs-C++ golden comparison: the number, its unit, and the measured
    evidence that justifies it live in ``tolerance_policy.json``, never as a
    literal in a test module. Unlike ``_expanded_matrix_contract.expanded_matrix_tolerance()``,
    no ``column_group`` sibling-fallback is needed here - every
    ``consume_burnup_extended.*`` entry records its own ``atol``/``rtol`` directly.

    :param mode: Harness mode name; the ``_burnup_extended`` suffix is added here.
    :param route: Route key (scenario ``case_id``) within that mode's
        Phase 7 policy section.
    :returns: ``(atol, rtol)``.
    :raises KeyError: If the section or route does not exist.
    :raises ValueError: If the entry records neither ``atol`` nor ``rtol``.
    """
    section = f"{mode}_burnup_extended"
    policy = load_tolerance_policy()
    if section not in policy:
        raise KeyError(f"no Phase 7 tolerance-policy section {section!r}")
    entries = policy[section]
    if route not in entries:
        raise KeyError(f"no route {route!r} in policy section {section!r}")
    entry = entries[route]
    if entry.get("atol") is None and entry.get("rtol") is None:
        raise ValueError(
            f"{section}.{route} records no atol/rtol - record the measured "
            "evidence in tolerance_policy.json rather than hardcoding a "
            "number in a test"
        )
    return entry.get("atol"), entry.get("rtol")


require_golden_tree = _HELPERS.require_golden_tree
