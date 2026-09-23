#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_dataset_contract_base.py - Shared golden-file I/O helpers for every
per-dataset contract module (``_expanded_matrix_contract.py``,
``_soil_campbell_contract.py``, ``_emissions_equivalence_contract.py``,
``_burnup_extended_contract.py``).

Every one of those modules previously defined its own byte-for-byte
identical copy of ``golden_dir``/``golden_manifest``/``golden_rows``/
``golden_rows_by_case``/``missing_golden_files``/``_required_golden_files``/
``require_golden_tree``, differing only in which golden root/mode
list/output-suffix table/generator script name they closed over. This
module factors that out into one parametrized builder,
:func:`make_dataset_helpers`, so a future change to the shared I/O
contract only needs to happen once.

Each dataset's own scenario matrices, tolerance-resolution logic, and
``*_rows()``/``*_tolerance()`` functions are NOT here - they differ
genuinely between datasets and stay in each dataset's own contract module.

Function order: the one public builder function, per AGENTS.md (no
private helpers at module scope - everything is a closure inside the
builder).
"""
from __future__ import annotations

import csv
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from tests._support import PROJECT_ROOT


def make_dataset_helpers(
        *,
        golden_root: str,
        dataset_label: str,
        modes: Tuple[str, ...],
        mode_output_suffixes: Dict[str, Tuple[str, ...]],
        generator_hint: str,
) -> SimpleNamespace:
    """
    Build the seven shared golden-file I/O helpers for one dataset.

    :param golden_root: The dataset's ``GOLDEN_ROOT`` (absolute path).
    :param dataset_label: Human-readable label used only in error text
        (e.g. ``"Expanded Matrix"``, ``"Soil Campbell"``).
    :param modes: The dataset's harness modes, in declared order.
    :param mode_output_suffixes: ``{mode: (suffix, ...)}`` — the output-file
        suffixes each mode writes.
    :param generator_hint: The dataset's own generator script path, quoted
        in :func:`require_golden_tree`'s error message.
    :returns: A namespace with attributes ``_required_golden_files``,
        ``golden_dir``, ``golden_manifest``, ``golden_rows``,
        ``golden_rows_by_case``, ``missing_golden_files``,
        ``require_golden_tree`` — identical behavior to the pre-extraction
        per-module copies.
    """

    def golden_dir(mode: str) -> str:
        """Return the committed golden directory for *mode*."""
        return os.path.join(golden_root, mode)

    def _required_golden_files(mode: str) -> List[str]:
        """Return the absolute paths every committed golden for *mode*
        must contain."""
        directory = golden_dir(mode)
        paths = [
            os.path.join(directory, f"{mode}.manifest.json"),
            os.path.join(directory, f"{mode}_in.csv"),
        ]
        paths.extend(
            os.path.join(directory, f"{mode}{suffix}.csv")
            for suffix in mode_output_suffixes[mode]
        )
        return paths

    def golden_manifest(mode: str) -> Optional[Dict[str, Any]]:
        """Load *mode*'s committed manifest, or ``None`` if absent."""
        path = os.path.join(golden_dir(mode), f"{mode}.manifest.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def golden_rows(mode: str, suffix: str = "") -> List[Dict[str, str]]:
        """Read a committed golden output CSV as a list of row dicts."""
        path = os.path.join(golden_dir(mode), f"{mode}{suffix}.csv")
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def golden_rows_by_case(mode: str, suffix: str = "") -> Dict[str, Dict[str, str]]:
        """Read a committed golden output CSV, keyed by ``case_id``."""
        out: Dict[str, Dict[str, str]] = {}
        for row in golden_rows(mode, suffix):
            case_id = row["case_id"]
            if case_id in out:
                raise ValueError(
                    f"duplicate case_id {case_id!r} in {mode}{suffix}.csv"
                )
            out[case_id] = row
        return out

    def missing_golden_files() -> List[str]:
        """Return every required golden file that is absent or empty."""
        missing = []
        for mode in modes:
            for path in _required_golden_files(mode):
                if not os.path.isfile(path) or os.path.getsize(path) == 0:
                    missing.append(
                        os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
                    )
        return sorted(missing)

    def require_golden_tree() -> None:
        """Fail closed unless the complete committed golden dataset is
        present."""
        missing = missing_golden_files()
        if missing:
            raise FileNotFoundError(
                f"the committed {dataset_label} golden dataset is incomplete "
                "- this is a repository defect, not a skippable environment "
                "difference. Missing or empty:\n"
                + "\n".join(f"  - {path}" for path in missing)
                + f"\nRestore them from git, or regenerate with "
                  f"{generator_hint} (needs the live MSVC/CMake/Ninja "
                  "toolchain)."
            )

    return SimpleNamespace(
        _required_golden_files=_required_golden_files,
        golden_dir=golden_dir,
        golden_manifest=golden_manifest,
        golden_rows=golden_rows,
        golden_rows_by_case=golden_rows_by_case,
        missing_golden_files=missing_golden_files,
        require_golden_tree=require_golden_tree,
    )
