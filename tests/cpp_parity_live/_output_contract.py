#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_output_contract.py - Authoritative per-mode Phase 2 golden output-column
contract (Phase 2 round 4 correction item 1).

Distinguishes, for every ``(mode, output_suffix)`` file
``_golden_manifest.MODE_OUTPUT_SUFFIXES`` declares:

- METADATA/CONTROL columns: IDs, mode, schema version, outcome, return/
  error fields, equation identifiers, counts used only for reconciliation,
  and hashes. These never need a tolerance-policy classification.
- SCIENTIFIC columns: every remaining column, which MUST have an
  applicable ``tolerance_policy.json`` entry (see
  ``tests/unit/test_tolerance_policy_completeness.py``, which derives the
  REAL CSV headers from the generated goldens rather than trusting a
  second hardcoded list).

Two resolution rules connect a real scientific column to its
``tolerance_policy.json`` key:

1. **Direct (most modes)**: the tolerance_policy entry declares an explicit
   ``covers_columns`` list of the real column names its status/justification
   applies to. A column may legitimately appear in more than one entry's
   ``covers_columns`` ONLY when those entries share the same declared
   ``column_group`` tag (proving an intentional multi-equation-route
   classification of the same output — e.g. mortality's three `mort_equ`
   routes all writing through the single ``prob`` column — rather than an
   accidental duplicate).
2. **Component-indirect (``consume_components.csv`` only)**: this file is
   long-format (one row per component, four GENERIC value columns
   ``pre_tac``/``con_tac``/``pos_tac``/``pct_con`` that repeat for every
   component). :data:`CONSUME_COMPONENT_TO_PREFIX` /
   :data:`CONSUME_COMPONENT_VALUE_COLUMN_TO_SUFFIX` connect a real
   ``(component, value_column)`` pair (read from the real generated file,
   not assumed) to the equivalent ``consume`` mode policy key — e.g.
   ``("Duff", "con_tac")`` resolves to the same ``DufCon`` entry that
   already covers ``consume_summary.csv``'s ``DufCon`` column, per the
   plan's explicit instruction to connect duplicate/derived component
   fields to their summary-field policies rather than duplicating entries.

Function order: private-then-public alphabetized per AGENTS.md (no
private helpers needed here — every top-level name is a public constant/
function used by the completeness tests and by manifest construction).
"""
from __future__ import annotations

import csv
import os
from typing import Dict, FrozenSet, List, Tuple

#: Metadata/control columns per (mode, output_suffix). These never need a
#: tolerance_policy classification. Explicit and tested (see
#: test_tolerance_policy_completeness.py) — never inferred by exclusion of
#: "whatever the policy happens to cover".
METADATA_COLUMNS: Dict[Tuple[str, str], FrozenSet[str]] = {
    ("consume", "_summary"): frozenset({
        "case_id", "mode", "schema_version", "outcome", "ret_code",
        "err_text", "input_sha256",
    }),
    ("consume", "_components"): frozenset({
        "case_id", "component", "equation", "input_sha256",
    }),
    ("litter_eq", ""): frozenset({
        "case_id", "mode", "schema_version", "outcome", "equ_num", "ret",
        "err_text", "input_sha256",
    }),
    ("shrub_herb_eq", ""): frozenset({
        "case_id", "mode", "schema_version", "outcome",
        "shrub_equ", "herb_equ", "fol_equ", "bra_equ",
        "ret", "err_text", "input_sha256",
    }),
    ("mortality", ""): frozenset({
        "case_id", "mode", "schema_version", "outcome", "mort_equ", "ret",
        "err_text", "input_sha256",
    }),
    ("bark_thick", ""): frozenset({
        "case_id", "mode", "schema_version", "outcome", "ret", "err_text",
        "input_sha256",
    }),
    ("canopy_cover", "_trees"): frozenset({
        "case_id", "stand_id", "mode", "schema_version", "outcome",
        "cct_equ_no", "ret", "err_text", "input_sha256",
    }),
    ("canopy_cover", "_stands"): frozenset({
        "stand_id", "mode", "schema_version", "n_trees", "stand_sha256",
    }),
    # canopy_cover_groups.csv is a pure reconciliation/diagnostic file
    # (membership counts, aggregate-emitted flag, suppression reason) —
    # every column is metadata; zero scientific columns, by design.
    ("canopy_cover", "_groups"): frozenset({
        "stand_id", "mode", "schema_version", "n_members", "n_ok",
        "n_expected_model_error", "n_unexpected_failure",
        "aggregate_emitted", "suppression_reason", "group_sha256",
    }),
}

#: consume_components.csv's component-name (as written by the harness) to
#: its consume_summary.csv field-name prefix. Verified against the real
#: generated file's distinct `component` values, not assumed.
CONSUME_COMPONENT_TO_PREFIX: Dict[str, str] = {
    "Litter": "Lit",
    "DW1": "DW1",
    "DW10": "DW10",
    "DW100": "DW100",
    "SndDW1k": "SndDW1k",
    "RotDW1k": "RotDW1k",
    "Duff": "Duf",
    "Herb": "Her",
    "Shrub": "Shr",
    "Foliage": "Fol",
    "Branch": "Bra",
}

#: consume_components.csv's four generic value columns to the summary-field
#: suffix they correspond to. `pct_con` -> `Per` even for the ten groups
#: that have no `<Prefix>Per` column in consume_summary.csv itself (only
#: `DufPer` does) — the quantity is still real and scientifically
#: meaningful, so it still gets its own `tolerance_policy.json` entry,
#: just one that consume_components.csv is its only source for.
CONSUME_COMPONENT_VALUE_COLUMN_TO_SUFFIX: Dict[str, str] = {
    "pre_tac": "Pre",
    "con_tac": "Con",
    "pos_tac": "Pos",
    "pct_con": "Per",
}

#: Policy routes exercised by each canonical Phase 2 golden.  ``consume``
#: is handled specially by :func:`phase2_canonical_policy_keys`: its one
#: output row emits every scientific field, so every consume policy entry
#: applies.  The other modes have alternate equation routes that write the
#: same output columns; only the routes named here apply to the generated
#: canonical row.
PHASE2_CANONICAL_ROUTE_KEYS: Dict[str, Tuple[str, ...]] = {
    "litter_eq": ("997",),
    "shrub_herb_eq": (
        "crown_branch", "crown_foliage", "herb", "shrub",
    ),
    "mortality": ("CroSco",),
    "bark_thick": ("all",),
    "canopy_cover": ("all",),
}

#: Subset of the canonical routes whose known divergence actually applies
#: to the canonical Phase 2 input.  This separate, explicit mapping keeps
#: conditional findings such as consume's Northeast case-6 F-23 and the
#: Pine-Flatwoods herb route from being falsely attributed to the canonical
#: Interior-West rows merely because those findings share output columns.
PHASE2_CANONICAL_DIVERGENCE_KEYS: Dict[str, Tuple[str, ...]] = {
    "consume": (),
    "litter_eq": ("997",),
    "shrub_herb_eq": (),
    "mortality": ("CroSco",),
    "bark_thick": ("all",),
    "canopy_cover": ("all",),
}


def classify_columns(
        mode: str, suffix: str, header: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Split *header* into ``(metadata, scientific)`` column-name lists for
    ``(mode, suffix)``.

    :param mode: Harness mode name.
    :param suffix: Output-file suffix (``""`` for a mode's single primary
        file, else e.g. ``"_summary"``).
    :param header: The real CSV header row (column names, in file order).
    :return: ``(metadata_columns, scientific_columns)``, both in *header*
        order.
    :raises KeyError: If ``(mode, suffix)`` has no registered metadata set.
    """
    metadata_set = METADATA_COLUMNS[(mode, suffix)]
    metadata = [c for c in header if c in metadata_set]
    scientific = [c for c in header if c not in metadata_set]
    return metadata, scientific


def component_field_policy_key(component: str, value_column: str) -> str:
    """
    Resolve a real ``consume_components.csv`` ``(component, value_column)``
    pair to the equivalent ``consume`` mode ``tolerance_policy.json`` key.

    :param component: Real value of the row's ``component`` field (e.g.
        ``"Duff"``).
    :param value_column: Which of the four generic value columns (e.g.
        ``"con_tac"``).
    :return: The ``consume`` mode policy key, e.g. ``"DufCon"``.
    :raises KeyError: If *component* or *value_column* is not a known,
        mapped name (this is intentional fail-closed — an unrecognized
        component name in freshly generated data means this contract is
        stale and must be updated, not silently skipped).
    """
    prefix = CONSUME_COMPONENT_TO_PREFIX[component]
    suffix = CONSUME_COMPONENT_VALUE_COLUMN_TO_SUFFIX[value_column]
    return f"{prefix}{suffix}"


def phase2_canonical_divergence_keys(mode: str) -> List[str]:
    """Return policy keys divergent for *mode*'s canonical Phase 2 row.

    :param mode: Harness mode name.
    :returns: Dotted policy keys in deterministic contract order.
    :raises KeyError: If *mode* has no canonical Phase 2 contract.
    """
    return [
        f"{mode}.{route}"
        for route in PHASE2_CANONICAL_DIVERGENCE_KEYS[mode]
    ]


def phase2_canonical_policy_keys(mode: str, policy: dict) -> List[str]:
    """Return every policy key applicable to *mode*'s canonical golden.

    ``consume`` emits all of its scientific fields in one row, so all
    consume entries apply.  Other modes use the explicitly selected
    equation routes in :data:`PHASE2_CANONICAL_ROUTE_KEYS`.

    :param mode: Harness mode name.
    :param policy: Loaded tolerance-policy object.
    :returns: Dotted policy keys in deterministic contract order.
    :raises KeyError: If the mode or a configured route is absent.
    """
    routes = (
        sorted(policy[mode])
        if mode == "consume"
        else list(PHASE2_CANONICAL_ROUTE_KEYS[mode])
    )
    for route in routes:
        policy[mode][route]
    return [f"{mode}.{route}" for route in routes]


def read_real_header(csv_path: str) -> List[str]:
    """
    Read and return the real header row of a generated golden CSV.

    :param csv_path: Absolute path to the CSV file.
    :return: Column names, in file order.
    :raises FileNotFoundError: If *csv_path* does not exist — callers
        should surface this as a clear skip/failure, never substitute a
        hardcoded assumption for a missing real file.
    """
    with open(csv_path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def real_distinct_components(csv_path: str) -> List[str]:
    """
    Return the distinct, real ``component`` values found in a generated
    ``consume_components.csv``, in first-seen order.

    :param csv_path: Absolute path to ``consume_components.csv``.
    :return: Distinct component names as actually written by the harness.
    """
    seen: List[str] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["component"]
            if name not in seen:
                seen.append(name)
    return seen
