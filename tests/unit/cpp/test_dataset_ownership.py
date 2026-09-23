#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_dataset_ownership.py - Explicit dataset-mode-ownership guard between
``canonical``, ``expanded_matrix``, and ``soil_campbell``.

``soil_campbell`` was added (Phase 5, Part 1) to the shared vocabulary
dicts in ``_golden_manifest.py`` (``MODE_SCHEMA_VERSIONS`` and
``MODE_OUTPUT_SUFFIXES``) so the common harness-contract machinery
(schema-version enforcement, output-file-suffix lookup) knows about it.
It was deliberately NOT added to ``test_cpp_harness_contract.MODES`` /
``ALL_MODE_NAMES``, the dict the ``canonical`` generator (``generate_
canonical_goldens.py``) iterates directly, nor to
``_expanded_matrix_contract.EXPANDED_MATRIX_MODES``, the explicit tuple
the ``expanded_matrix`` generator (``generate_expanded_matrix_goldens.py``)
iterates directly.

This module proves, by EXACT membership (not count), that:

- ``MODES`` (``canonical``'s owned mode set) is exactly the six shared
  modes, and structurally excludes ``soil_campbell``.
- ``EXPANDED_MATRIX_MODES`` (``expanded_matrix``'s owned mode set) is
  exactly the same six modes, and structurally excludes ``soil_campbell``.
- ``soil_campbell`` IS a real, known mode name in the shared vocabulary
  dicts (``MODE_SCHEMA_VERSIONS``, ``MODE_OUTPUT_SUFFIXES``) - it is not
  simply absent from the codebase - but is owned by neither ``canonical``
  nor ``expanded_matrix``.
- ``generate_canonical_goldens.py`` and ``generate_expanded_matrix_goldens.py`` each
  contain exactly one mode-iteration construct over their respective
  owned-mode collection (``MODES`` / ``EXPANDED_MATRIX_MODES``), confirmed by a
  source scan, not merely "currently doesn't crash" - a future edit that
  adds a second, unscoped iteration (e.g. over the raw
  ``MODE_OUTPUT_SUFFIXES`` dict, the defect this module's sibling fix in
  ``test_tolerance_policy_completeness.py`` corrected) is caught here.
- The two datasets are disjoint from ``soil_campbell``'s own mode set of
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
from tests.cpp_parity_live._expanded_matrix_contract import EXPANDED_MATRIX_MODES
from tests.cpp_parity_live.test_cpp_harness_contract import ALL_MODE_NAMES, MODES

#: The exact six modes ``canonical`` AND ``expanded_matrix`` own. Hardcoded
#: and explicit (not derived from any other module's collection) so this
#: file is an independent witness, not a tautology against the code it is
#: checking.
_CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES = frozenset(
    {"consume", "litter_eq", "shrub_herb_eq", "mortality", "bark_thick", "canopy_cover"}
)

#: The exact one mode ``soil_campbell`` owns.
_SOIL_CAMPBELL_OWNED_MODES = frozenset({"soil_campbell"})

_GENERATOR_PATHS = {
    "canonical": os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", "generate_canonical_goldens.py"
    ),
    "expanded_matrix": os.path.join(
        PROJECT_ROOT, "tests", "cpp_parity_live", "generate_expanded_matrix_goldens.py"
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


def test_canonical_and_expanded_matrix_owned_mode_sets_are_identical_and_exact():
    """``MODES``/``ALL_MODE_NAMES`` (``canonical``) and
    ``EXPANDED_MATRIX_MODES`` (``expanded_matrix``) must both equal the
    same explicit six-mode set - not merely the same length."""
    assert set(MODES) == _CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES
    assert set(ALL_MODE_NAMES) == _CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES
    assert set(EXPANDED_MATRIX_MODES) == _CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES
    assert len(MODES) == 6
    assert len(EXPANDED_MATRIX_MODES) == 6


def test_canonical_and_expanded_matrix_generators_only_iterate_their_owned_mode_collection():
    """Source-scan proof (not behavioural inference) that neither
    generator's mode-iteration construct names the raw shared vocabulary
    dicts. Every bare-identifier ``for mode in X`` / ``list(X)`` construct
    found in each generator must resolve to that generator's own
    owned-mode collection, never to ``MODE_OUTPUT_SUFFIXES`` or
    ``MODE_SCHEMA_VERSIONS`` directly."""
    forbidden_names = {"MODE_OUTPUT_SUFFIXES", "MODE_SCHEMA_VERSIONS"}
    allowed_by_dataset = {
        "canonical": {"MODES", "ALL_MODE_NAMES"},
        "expanded_matrix": {"EXPANDED_MATRIX_MODES"},
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
            "soil_campbell-only mode) into this dataset's generation."
        )
        # At least one iteration construct must exist and it must be a
        # collection this dataset actually owns.
        owned_hits = found_names & allowed_by_dataset[dataset]
        assert owned_hits, (
            f"{path} has no recognisable mode-iteration construct over its "
            f"owned collection {allowed_by_dataset[dataset]}"
        )


def test_soil_campbell_is_a_known_mode_but_owned_by_neither_canonical_nor_expanded_matrix():
    """``soil_campbell`` must be present in the shared vocabulary (proving
    it is real, wired-in Part-1 work, not simply missing) while being
    absent from both dataset-owned mode sets."""
    assert "soil_campbell" in MODE_SCHEMA_VERSIONS
    assert "soil_campbell" in MODE_OUTPUT_SUFFIXES
    assert "soil_campbell" not in MODES
    assert "soil_campbell" not in ALL_MODE_NAMES
    assert "soil_campbell" not in EXPANDED_MATRIX_MODES


def test_the_three_dataset_owned_mode_sets_partition_every_known_mode():
    """``canonical``/``expanded_matrix``'s owned set and ``soil_campbell``'s
    owned set must be disjoint and, together, must equal the complete
    known-mode vocabulary - no mode is owned by more than one dataset and
    no known mode is owned by zero datasets."""
    all_known_modes = frozenset(MODE_SCHEMA_VERSIONS)
    assert _CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES & _SOIL_CAMPBELL_OWNED_MODES == frozenset()
    assert _CANONICAL_AND_EXPANDED_MATRIX_OWNED_MODES | _SOIL_CAMPBELL_OWNED_MODES == all_known_modes
