#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_phase5_goldens.py - Generate the Phase 5 ``soil_campbell``
scenario-matrix golden dataset (BR-SOI-DUFF/NODUFF/NOIG) with a full
provenance manifest.

Unlike Phase 4 (which reuses the six already-qualified Phase 2 modes),
Phase 5 drives its OWN new harness mode, ``soil_campbell`` (Part 1). This
module deliberately does NOT re-implement the Phase 2 generator's safety
machinery — it imports and reuses it verbatim, exactly as
``generate_phase4_goldens.py`` does:

- :func:`~tests.cpp_parity_live.generate_phase2_goldens._qualify_all` - the
  COMPLETE, unfiltered ``test_cpp_harness_contract.py`` module (which
  already includes the ``soil_campbell`` self-test rows) must pass, as one
  real executed subprocess, before any Phase 5 golden is generated.
- :func:`~tests.cpp_parity_live.generate_phase2_goldens._promote` - the
  single-writer-locked, crash-recoverable, whole-tree swap.
- :func:`~tests.cpp_parity_live.generate_phase2_goldens._validate_staged_tree`
  - the complete staged batch must pass :func:`validate_manifest` before the
  committed tree is touched at all.
- :func:`~tests.cpp_parity_live.generate_phase2_goldens.verify_regeneration`
  - the production determinism comparison, shared with ``--verify-only``.

The Phase 2/Phase 4 golden trees are never read, written, or promoted by
this module: ``GOLDEN_ROOT`` here is
``tests/test_data/test_golden_output/phase5/``, a sibling of
``phase2/``/``phase4/``.

Safety/gating contract (identical to Phase 2/4's, applied to the Phase 5
tree): pinned-SHA check before overlay reapply/configure/build and again
inside every manifest build; the complete harness self-test matrix must
pass first; every manifest is validated (re-hashing every referenced file
from disk) before promotion; promotion is whole-tree and recoverable.

**Item-7 scientific-risk note (updated 2026-09-04 — corrected from the
original "not-yet-attempted" framing).** This generator produces the
C++-ORACLE side of the Phase 5 dataset only; no comparison or tolerance is
computed here. Executed cross-implementation numerical CHARACTERIZATION
against this generator's own golden output DOES exist elsewhere
(``tests/unit/test_phase5_soil_campbell_characterization.py``), but it is
not a scientific parity comparison: F-52 established that full-model parity
is an inappropriate classification for the ``duff``/``nonduff`` routes,
since the Python and C++ implementations represent materially different
physics, not one model at two numerical precisions. Accordingly
``tolerance_policy.json``'s ``soil_campbell_p5`` section records ``duff``/
``nonduff`` as ``"status": "unverified"`` (characterized, not scientifically
validated; null ``atol``/``rtol``) and ``noig`` as ``"contract_only"`` (no
Python code path exists to compare against at all) — see
``_phase5_contract.py``'s module docstring for the full evidence chain.
Separately, F-53 identified an independent, CONFIRMED Python defect (a
missing percent-to-ratio conversion in ``_duff_flux_and_duration()``): that
is tracked as its own ``soil_campbell_p5.duff_moisture_unit`` route,
``"status": "known_divergent_strict_xfail"`` (also null ``atol``/``rtol``,
since a binary contract defect cannot honestly be bounded by a tolerance
either), pinned by a strict ``xfail`` rather than folded into the
``duff``/``nonduff`` full-model-characterization routes above.

Usage:
    python tests/cpp_parity_live/generate_phase5_goldens.py
    python tests/cpp_parity_live/generate_phase5_goldens.py --verify-only

Function order: private helpers first, then public functions, each group
alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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
    ensure_built,
    HARNESS_EXE,
    run_harness,
    toolchain_status,
)
from tests.cpp_parity_live._phase5_contract import (
    DATASET_NAME,
    GOLDEN_ROOT,
    MODE_OUTPUT_SUFFIXES,
    PHASE5_MODES,
    phase5_rows,
)
from tests.cpp_parity_live.generate_phase2_goldens import (
    _build_flags_from_cache,
    _compiler_identity,
    _generator_toolchain_identity,
    _promote,
    _qualify_all,
    verify_regeneration,
)
from tests.cpp_parity_live.test_cpp_harness_contract import (
    SOIL_CAMPBELL_FI_HS_NAME,
    SOIL_CAMPBELL_FI_WL_NAME,
    SOIL_CAMPBELL_HEADER,
    SOIL_CAMPBELL_N_STEPS,
)

#: Tolerance-policy keys applicable to each Phase 5 mode's golden, resolved
#: through the same shared helper the manifest builder/validator use, so
#: the generator can never cite a key set the validator would reject.
_TOLERANCE_POLICY = load_tolerance_policy()
GOLDEN_TOLERANCE_KEYS = {
    mode: canonical_policy_keys(DATASET_NAME, mode, _TOLERANCE_POLICY)
    for mode in PHASE5_MODES
}


def _generate_one(mode: str, out_dir: str) -> str:
    """
    Generate *mode*'s Phase 5 golden plus manifest into *out_dir*.

    :param mode: Harness mode name (only ``"soil_campbell"`` is supported).
    :param out_dir: Directory to write this mode's files into (also where
        the two shared fire-intensity side files are written for the
        harness to read, and removed from again before this function
        returns — see the note on side files below).
    :returns: Absolute path to the manifest written.
    :raises RuntimeError: If the harness exits nonzero or the generated
        manifest fails validation.
    :raises KeyError: If *mode* is not a Phase 5 mode.

    **Why no ``side_file_sha256`` manifest entry.** Unlike Phase 2/4's
    species/emission-factor tables (stable, pre-existing repository
    files), ``soil_campbell``'s two fire-intensity series are GENERATED
    fresh into a directory that, during staging, lives outside the repo
    (``generate_all``'s ``tempfile.TemporaryDirectory()``) — a path
    ``_golden_manifest.to_repo_relative``/``check_pinned_sha``'s sibling
    safety check (``_is_safe_repo_relative_path``) correctly rejects, and
    ``validate_manifest`` separately rejects any ``side_file_sha256`` role
    for a mode not in ``MODES_REQUIRING_SPECIES_TABLE``/
    ``MODES_REQUIRING_EMISSION_FACTOR_TABLE`` (``soil_campbell`` is in
    neither). This is not a gap: harness-contract section 7's own text —
    confirmed directly in ``run_soil_campbell`` (``test_harness.cpp``,
    "input_sha256: normalised fields with the two path columns replaced
    by the referenced files' own content hash") — already folds each side
    file's SHA-256 into every row's own ``input_sha256`` output column,
    so :func:`~tests.cpp_parity_live.generate_phase2_goldens.verify_regeneration`'s
    ordinary CSV content comparison already detects any side-file drift
    without a separate manifest field. The two files are therefore
    write-then-delete scratch inputs, not committed golden artifacts —
    committing them would also make ``_validate_staged_tree``'s exact
    per-mode file-set check reject them as unexpected extras.
    """
    if mode not in PHASE5_MODES:
        raise KeyError(f"unknown Phase 5 mode: {mode!r}")
    os.makedirs(out_dir, exist_ok=True)
    wl_path = os.path.join(out_dir, SOIL_CAMPBELL_FI_WL_NAME)
    hs_path = os.path.join(out_dir, SOIL_CAMPBELL_FI_HS_NAME)
    _write_side_files(out_dir)
    try:
        prefix = os.path.join(out_dir, mode)
        result = run_harness(
            mode, SOIL_CAMPBELL_HEADER, phase5_rows(mode), prefix,
            output_suffixes=MODE_OUTPUT_SUFFIXES[mode],
        )
    finally:
        # Scratch inputs only — see the docstring's "why no side_file_sha256"
        # note. Removed unconditionally, even on a harness failure below, so
        # a partially-generated out_dir never leaves them behind either.
        for path in (wl_path, hs_path):
            if os.path.isfile(path):
                os.remove(path)
    if result.returncode != 0:
        raise RuntimeError(
            f"Phase 5 golden generation for mode={mode!r} failed "
            f"(rc={result.returncode}):\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )

    input_csv = [prefix + "_in.csv"]
    output_csvs = [
        prefix + suffix + ".csv" for suffix in MODE_OUTPUT_SUFFIXES[mode]
        if os.path.isfile(prefix + suffix + ".csv")
    ]

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
            f"generated Phase 5 manifest for mode={mode!r} failed "
            "validation:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return manifest_path


def _write_side_files(out_dir: str) -> None:
    """
    Write the two shared fire-intensity side files into *out_dir*.

    Same short, clearly-decaying formula
    ``test_cpp_harness_contract._write_soil_side_files`` uses for
    qualification (a load-bearing choice, not cosmetic: neither
    ``SD_Mngr_New`` nor ``SE_Mngr_Array``'s stepping loop has a hard
    iteration cap independent of ``SHA_Get``'s own ``eC_Tim(10000)`` table
    bound, harness-contract section 7) — reimplemented rather than
    imported since it is a private test-fixture helper of a different
    module, not shared provenance/manifest/promotion machinery.

    :param out_dir: Directory to write the two side files into.
    :returns: None.
    """
    wl_path = os.path.join(out_dir, SOIL_CAMPBELL_FI_WL_NAME)
    hs_path = os.path.join(out_dir, SOIL_CAMPBELL_FI_HS_NAME)
    wl_values = [max(0.0, 50.0 - i * 3.0) for i in range(SOIL_CAMPBELL_N_STEPS)]
    hs_values = [max(0.0, 10.0 - i * 0.5) for i in range(SOIL_CAMPBELL_N_STEPS)]
    with open(wl_path, "w", newline="\n") as f:
        f.write("\n".join(str(v) for v in wl_values) + "\n")
    with open(hs_path, "w", newline="\n") as f:
        f.write("\n".join(str(v) for v in hs_values) + "\n")


def generate_all(out_root: str, *, qualify: bool = True) -> None:
    """
    Generate every Phase 5 mode's golden + manifest, gated on pinned-SHA,
    build success, and (if *qualify*) the complete harness self-test
    matrix.

    Writes to a fresh temporary directory and promotes to *out_root* as
    one whole-tree transaction only after every mode has generated AND the
    complete staged batch has validated.

    :param out_root: Destination directory (only touched after full
        success).
    :param qualify: If ``False``, skip the self-test subprocess gate -
        used only by this module's own driver tests, which must exercise
        generation failure paths independently of the self-test suite.
    :returns: None.
    :raises RuntimeError: If the toolchain is unavailable, the build
        fails, qualification fails, or any mode fails to generate/
        validate.
    :raises ProvenanceError: If the C++ checkout is not at the pinned
        SHA.
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

    with tempfile.TemporaryDirectory() as tmp_root:
        for mode in PHASE5_MODES:
            _generate_one(mode, os.path.join(tmp_root, mode))
        _promote(tmp_root, out_root, list(PHASE5_MODES))


def main() -> int:
    """
    Command-line entry point.

    :returns: Process exit code - 0 on success, 1 if ``--verify-only``
        detected a difference against the committed Phase 5 tree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Regenerate into a temp dir and diff against the committed "
             "Phase 5 goldens instead of overwriting.",
    )
    args = parser.parse_args()

    if args.verify_only:
        with tempfile.TemporaryDirectory() as tmp:
            generate_all(tmp)
            mismatches = verify_regeneration(
                GOLDEN_ROOT, tmp, list(PHASE5_MODES)
            )
            if mismatches:
                print("PHASE 5 DETERMINISM CHECK FAILED:")
                for message in mismatches:
                    print(" -", message)
                return 1
            print(
                "PHASE 5 DETERMINISM CHECK PASSED: fresh regeneration is "
                "byte-identical (SHA-256 compared) to the committed Phase 5 "
                "goldens for every input/output CSV - no missing/extra files "
                "either - and every manifest field matches except "
                "generated_utc, generating_command, and "
                "pyfofem_dirty.porcelain."
            )
            return 0

    generate_all(GOLDEN_ROOT)
    print(f"Generated Phase 5 goldens under {GOLDEN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
