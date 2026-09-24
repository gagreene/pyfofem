#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_burnup_extended_goldens.py - Generate the Phase 7 item E additional-
``run_burnup`` golden dataset (a real, manifested ``consume`` scenario
file covering rotten/sound-mix, calm-wind, hot-ambient, and long-residence
-time burnup branches) with a full provenance manifest.

Phase 7 item E adds **no harness mode**. Every scenario in
``_burnup_extended_contract.py`` is driven through the already-qualified
``consume`` mode, using its existing, unchanged input schema.

This module deliberately does NOT re-implement the Phase 2 generator's
safety machinery. It imports and reuses it verbatim, exactly as
``generate_expanded_matrix_goldens.py``/``generate_soil_campbell_goldens.py``/
``generate_emissions_equivalence_goldens.py`` do:

- :func:`~tests.cpp_parity_live.generate_canonical_goldens._qualify_all` - the
  COMPLETE, unfiltered ``test_cpp_harness_contract.py`` module must pass, as
  one real executed subprocess, before any Phase 7 golden is generated.
- :func:`~tests.cpp_parity_live.generate_canonical_goldens._promote` - the
  single-writer-locked, crash-recoverable, whole-tree swap.
- :func:`~tests.cpp_parity_live.generate_canonical_goldens._validate_staged_tree`
  - the complete staged batch must pass :func:`validate_manifest` before the
  committed tree is touched at all.
- :func:`~tests.cpp_parity_live.generate_canonical_goldens.verify_regeneration`
  - the production determinism comparison, shared with ``--verify-only``.

The canonical/expanded_matrix/soil_campbell/emissions_equivalence golden
trees are never read, written, or promoted by this module: ``GOLDEN_ROOT``
here is ``tests/test_data/test_golden_output/burnup_extended/``, a sibling
of ``canonical/``/``expanded_matrix/``/``soil_campbell/``/
``emissions_equivalence/``.

Usage:
    python tests/cpp_parity_live/generate_burnup_extended_goldens.py
    python tests/cpp_parity_live/generate_burnup_extended_goldens.py --verify-only

Function order: private helpers first, then public functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime, timezone

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import (
    build_manifest,
    canonical_policy_keys,
    check_pinned_sha,
    load_tolerance_policy,
    MODE_SCHEMA_VERSIONS,
    validate_manifest,
    write_manifest,
)
from tests.cpp_parity_live._harness_support import (
    HARNESS_EXE,
    ensure_built,
    run_harness,
    toolchain_status,
)
from tests.cpp_parity_live._burnup_extended_contract import (
    DATASET_NAME,
    GOLDEN_ROOT,
    MODE_OUTPUT_SUFFIXES,
    BURNUP_EXTENDED_MODES,
    burnup_extended_rows,
)
from tests.cpp_parity_live._scratch import scratch_tempdir
from tests.cpp_parity_live.generate_canonical_goldens import (
    _build_flags_from_cache,
    _compiler_identity,
    _generator_toolchain_identity,
    _promote,
    _qualify_all,
    verify_regeneration,
)
from tests.cpp_parity_live.test_cpp_harness_contract import MODES

#: Tolerance-policy keys applicable to each Phase 7 mode's golden, resolved
#: through the same shared helper the manifest builder/validator use.
_TOLERANCE_POLICY = load_tolerance_policy()
GOLDEN_TOLERANCE_KEYS = {
    mode: canonical_policy_keys(DATASET_NAME, mode, _TOLERANCE_POLICY)
    for mode in BURNUP_EXTENDED_MODES
}


def _generate_one(mode: str, out_dir: str) -> str:
    """
    Generate *mode*'s Phase 7 golden plus manifest into *out_dir*.

    :param mode: Harness mode name (only ``"consume"`` is supported).
    :param out_dir: Directory to write this mode's files into.
    :returns: Absolute path to the manifest written.
    :raises RuntimeError: If the harness exits nonzero or the generated
        manifest fails validation.
    :raises KeyError: If *mode* is not a Phase 7 mode.
    """
    if mode not in BURNUP_EXTENDED_MODES:
        raise KeyError(f"unknown Phase 7 mode: {mode!r}")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, mode)
    header = MODES[mode]["header"]
    result = run_harness(
        mode, header, burnup_extended_rows(mode), prefix,
        output_suffixes=MODE_OUTPUT_SUFFIXES[mode],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"burnup_extended golden generation for mode={mode!r} failed "
            f"(rc={result.returncode}):\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

    input_csv = [prefix + "_in.csv"]
    output_csvs = [
        prefix + suffix + ".csv" for suffix in MODE_OUTPUT_SUFFIXES[mode]
        if os.path.isfile(prefix + suffix + ".csv")
    ]
    side_files = {}
    if mode == "consume":
        factor_csv = os.path.join(
            PROJECT_ROOT, "reference", "fofem_cpp", "FOF_UNIX",
            "Emission_Factors.csv",
        )
        if os.path.isfile(factor_csv):
            side_files["emission_factor_table"] = factor_csv

    manifest = build_manifest(
        harness_mode=mode,
        schema_version=MODE_SCHEMA_VERSIONS[mode],
        compiler_identity=_compiler_identity(),
        generator_toolchain=_generator_toolchain_identity(),
        platform=platform.platform(),
        architecture=platform.machine(),
        build_type="Debug",
        build_flags=_build_flags_from_cache(),
        generating_command=f"{HARNESS_EXE} {prefix}_in.csv {prefix}",
        input_csv_paths=input_csv,
        output_csv_paths=output_csvs,
        tolerance_policy_keys=GOLDEN_TOLERANCE_KEYS[mode],
        side_files=side_files,
        now_utc_iso=datetime.now(timezone.utc).isoformat(),
        dataset=DATASET_NAME,
    )
    manifest_path = os.path.join(out_dir, f"{mode}.manifest.json")
    write_manifest(manifest_path, manifest)

    errors = validate_manifest(
        manifest, check_against_live_checkout=True, golden_dir=out_dir,
    )
    if errors:
        raise RuntimeError(
            f"generated Phase 7 manifest for mode={mode!r} failed "
            "validation:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return manifest_path


def generate_all(out_root: str, *, qualify: bool = True) -> None:
    """
    Generate every Phase 7 mode's golden + manifest, gated on pinned-SHA,
    build success, and (if *qualify*) the complete harness self-test matrix.

    :param out_root: Destination directory (only touched after full
        success).
    :param qualify: If ``False``, skip the self-test subprocess gate - used
        only by this module's own driver tests.
    :returns: None.
    :raises RuntimeError: If the toolchain is unavailable, the build fails,
        qualification fails, or any mode fails to generate/validate.
    :raises ProvenanceError: If the C++ checkout is not at the pinned SHA.
    """
    check_pinned_sha()
    ok, reason = toolchain_status()
    if not ok:
        raise RuntimeError(f"toolchain unavailable: {reason}")
    ok, reason = ensure_built()
    if not ok:
        raise RuntimeError(f"build failed: {reason}")

    if qualify:
        _qualify_all()

    with scratch_tempdir("generate", prefix="generate-all") as tmp_root:
        for mode in BURNUP_EXTENDED_MODES:
            _generate_one(mode, os.path.join(tmp_root, mode))
        _promote(tmp_root, out_root, list(BURNUP_EXTENDED_MODES))


def main() -> int:
    """
    Command-line entry point.

    :returns: Process exit code - 0 on success, 1 if ``--verify-only``
        detected a difference against the committed Phase 7 tree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Regenerate into a temp dir and diff against the committed "
             "Phase 7 goldens instead of overwriting.",
    )
    args = parser.parse_args()

    if args.verify_only:
        with scratch_tempdir("verify-only", prefix="verify") as tmp:
            generate_all(tmp)
            mismatches = verify_regeneration(
                GOLDEN_ROOT, tmp, list(BURNUP_EXTENDED_MODES)
            )
            if mismatches:
                print("BURNUP_EXTENDED DETERMINISM CHECK FAILED:")
                for message in mismatches:
                    print(" -", message)
                return 1
            print(
                "BURNUP_EXTENDED DETERMINISM CHECK PASSED: fresh regeneration is "
                "byte-identical (SHA-256 compared) to the committed burnup_extended "
                "goldens for every input/output CSV - no missing/extra files "
                "either - and every manifest field matches except "
                "generated_utc, generating_command, pyfofem_commit, "
                "and pyfofem_dirty."
            )
            return 0

    generate_all(GOLDEN_ROOT)
    print(f"Generated burnup_extended goldens under {GOLDEN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
