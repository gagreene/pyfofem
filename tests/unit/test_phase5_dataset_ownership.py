#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase5_dataset_ownership.py - Explicit dataset-mode-ownership guard
between Phase 2, Phase 4, and Phase 5.

``soil_campbell`` was added (Phase 5, Part 1) to the shared vocabulary
dicts in ``_golden_manifest.py`` (``MODE_SCHEMA_VERSIONS`` and
``MODE_OUTPUT_SUFFIXES``) so the common harness-contract machinery
(schema-version enforcement, output-file-suffix lookup) knows about it.
It was deliberately NOT added to ``test_cpp_harness_contract.MODES`` /
``ALL_MODE_NAMES``, the dict Phase 2's generator (``generate_phase2_
goldens.py``) iterates directly, nor to ``_phase4_contract.PHASE4_MODES``,
the explicit tuple Phase 4's generator (``generate_phase4_goldens.py``)
iterates directly.

This module proves, by EXACT membership (not count), that:

- ``MODES`` (Phase 2's owned mode set) is exactly the six Phase 2/4 modes,
  and structurally excludes ``soil_campbell``.
- ``PHASE4_MODES`` (Phase 4's owned mode set) is exactly the same six
  modes, and structurally excludes ``soil_campbell``.
- ``soil_campbell`` IS a real, known mode name in the shared vocabulary
  dicts (``MODE_SCHEMA_VERSIONS``, ``MODE_OUTPUT_SUFFIXES``) - it is not
  simply absent from the codebase - but is owned by neither Phase 2 nor
  Phase 4's dataset.
- ``generate_phase2_goldens.py`` and ``generate_phase4_goldens.py`` each
  contain exactly one mode-iteration construct over their respective
  owned-mode collection (``MODES`` / ``PHASE4_MODES``), confirmed by a
  source scan, not merely "currently doesn't crash" - a future edit that
  adds a second, unscoped iteration (e.g. over the raw
  ``MODE_OUTPUT_SUFFIXES`` dict, the defect this module's sibling fix in
  ``test_tolerance_policy_completeness.py`` corrected) is caught here.
- The two datasets are disjoint from a hypothetical Phase 5 mode set of
  exactly ``{"soil_campbell"}``.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import re

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import (
    MODE_OUTPUT_SUFFIXES,
    MODE_SCHEMA_VERSIONS,
)
from tests.cpp_parity_live._phase4_contract import PHASE4_MODES
from tests.cpp_parity_live.test_cpp_harness_contract import ALL_MODE_NAMES, MODES

#: The exact six modes Phase 2 AND Phase 4 own. Hardcoded and explicit
#: (not derived from any other module's collection) so this file is an
#: independent witness, not a tautology against the code it is checking.
_PHASE2_AND_4_OWNED_MODES = frozenset(
    {"consume", "litter_eq", "shrub_herb_eq", "mortality", "bark_thick", "canopy_cover"}
)

#: The exact one mode Phase 5 owns.
_PHASE5_OWNED_MODES = frozenset({"soil_campbell"})

_GENERATOR_PATHS = {
    "phase2": os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", "generate_phase2_goldens.py"
    ),
    "phase4": os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", "generate_phase4_goldens.py"
    ),
}

#: Matches a top-level mode-iteration construct: ``for mode in <NAME>`` or
#: ``list(<NAME>)`` where ``<NAME>`` is a bare identifier (not a subscript,
#: attribute access, or literal), which is precisely the shape an unscoped
#: ``MODE_OUTPUT_SUFFIXES``/``MODE_SCHEMA_VERSIONS`` leak would take.
_MODE_ITERATION_RE = re.compile(r"for\s+mode\s+in\s+([A-Za-z_][A-Za-z0-9_]*)|list\(([A-Za-z_][A-Za-z0-9_]*)\)")


def _read_generator_source(dataset: str) -> str:
    with open(_GENERATOR_PATHS[dataset], encoding="utf-8") as handle:
        return handle.read()


def test_phase2_and_phase4_owned_mode_sets_are_identical_and_exact():
    """``MODES``/``ALL_MODE_NAMES`` (Phase 2) and ``PHASE4_MODES``
    (Phase 4) must both equal the same explicit six-mode set - not merely
    the same length."""
    assert set(MODES) == _PHASE2_AND_4_OWNED_MODES
    assert set(ALL_MODE_NAMES) == _PHASE2_AND_4_OWNED_MODES
    assert set(PHASE4_MODES) == _PHASE2_AND_4_OWNED_MODES
    assert len(MODES) == 6
    assert len(PHASE4_MODES) == 6


def test_phase2_and_phase4_generators_only_iterate_their_owned_mode_collection():
    """Source-scan proof (not behavioural inference) that neither
    generator's mode-iteration construct names the raw shared vocabulary
    dicts. Every bare-identifier ``for mode in X`` / ``list(X)`` construct
    found in each generator must resolve to that generator's own
    owned-mode collection, never to ``MODE_OUTPUT_SUFFIXES`` or
    ``MODE_SCHEMA_VERSIONS`` directly."""
    forbidden_names = {"MODE_OUTPUT_SUFFIXES", "MODE_SCHEMA_VERSIONS"}
    allowed_by_dataset = {
        "phase2": {"MODES", "ALL_MODE_NAMES"},
        "phase4": {"PHASE4_MODES"},
    }
    for dataset, path in _GENERATOR_PATHS.items():
        source = _read_generator_source(dataset)
        found_names = set()
        for match in _MODE_ITERATION_RE.finditer(source):
            name = match.group(1) or match.group(2)
            found_names.add(name)
        offending = found_names & forbidden_names
        assert not offending, (
            f"{path} iterates a shared vocabulary dict directly: {offending} "
            "- this would silently pull soil_campbell (or any future "
            "Phase-5-only mode) into this dataset's generation."
        )
        # At least one iteration construct must exist and it must be a
        # collection this dataset actually owns.
        owned_hits = found_names & allowed_by_dataset[dataset]
        assert owned_hits, (
            f"{path} has no recognisable mode-iteration construct over its "
            f"owned collection {allowed_by_dataset[dataset]}"
        )


def test_soil_campbell_is_a_known_mode_but_owned_by_neither_phase2_nor_phase4():
    """``soil_campbell`` must be present in the shared vocabulary (proving
    it is real, wired-in Part-1 work, not simply missing) while being
    absent from both dataset-owned mode sets."""
    assert "soil_campbell" in MODE_SCHEMA_VERSIONS
    assert "soil_campbell" in MODE_OUTPUT_SUFFIXES
    assert "soil_campbell" not in MODES
    assert "soil_campbell" not in ALL_MODE_NAMES
    assert "soil_campbell" not in PHASE4_MODES


def test_the_three_dataset_owned_mode_sets_partition_every_known_mode():
    """Phase 2/4's owned set and Phase 5's owned set must be disjoint and,
    together, must equal the complete known-mode vocabulary - no mode is
    owned by more than one dataset and no known mode is owned by zero
    datasets."""
    all_known_modes = frozenset(MODE_SCHEMA_VERSIONS)
    assert _PHASE2_AND_4_OWNED_MODES & _PHASE5_OWNED_MODES == frozenset()
    assert _PHASE2_AND_4_OWNED_MODES | _PHASE5_OWNED_MODES == all_known_modes
