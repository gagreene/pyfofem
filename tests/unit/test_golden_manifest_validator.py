#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_golden_manifest_validator.py - Deterministic tests for the Phase 2
golden-dataset provenance manifest builder/validator.

These tests need no live C++ build: they exercise
``tests.cpp_parity_live._golden_manifest`` directly (git/hashing only).
They specifically prove the Codex-demonstrated gap is fixed — a manifest
whose output digest was replaced with zeros (or any other single-field
tamper: input, side-file, individual overlay file, combined overlay,
upstream SHA) must now be rejected, because every referenced file is
re-hashed from disk rather than trusted at face value.

Function order: top-level functions/fixtures are alphabetized,
private-then-public, per AGENTS.md.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from tests._support import PROJECT_ROOT
from tests.cpp_parity_live._golden_manifest import (
    EXPECTED_SPECIES_TABLE_REPO_PATH,
    GENERATOR_SOURCE_FILES,
    MODE_SCHEMA_VERSIONS,
    PINNED_UPSTREAM_SHA,
    REQUIRED_FIELDS,
    REQUIRED_OVERLAY_FILES,
    VALID_HARNESS_MODES,
    build_manifest,
    check_pinned_sha,
    compute_overlay_digests,
    current_upstream_sha,
    divergences_for_keys,
    git_dirty_status,
    load_tolerance_policy,
    to_repo_relative,
    validate_manifest,
    write_manifest,
)

#: The exact generator-source set item 3 (round 4 correction) requires:
#: every first-party file whose content directly changes generation,
#: manifest construction/validation, policy selection, qualification, or
#: deterministic verification.
_EXPECTED_GENERATOR_SOURCE_REPO_RELATIVE = frozenset({
    "tests/cpp_parity_live/_golden_manifest.py",
    "tests/cpp_parity_live/_harness_support.py",
    "tests/cpp_parity_live/_output_contract.py",
    "tests/cpp_parity_live/_proc.py",
    "tests/cpp_parity_live/generate_phase2_goldens.py",
    "tests/cpp_parity_live/test_cpp_harness_contract.py",
    "tests/cpp_parity_live/tolerance_policy.json",
})

# ``sample_manifest`` uses "mortality" (not "litter_eq") specifically
# because mortality is one of the modes in MODES_REQUIRING_SPECIES_TABLE —
# this lets the fixture exercise the real, exact species-table side-file
# contract (tracked FOF_SPP.CSV path + hash) rather than a synthetic one.
_SAMPLE_MODE = "mortality"


@pytest.fixture
def side_file():
    # The exact, single tracked species table every species-driven mode's
    # golden must reference (see EXPECTED_SPECIES_TABLE_REPO_PATH) — not an
    # arbitrary synthetic CSV, since validate_manifest() now checks the
    # side file's recorded repo-relative path against this exact value.
    return os.path.join(PROJECT_ROOT, *EXPECTED_SPECIES_TABLE_REPO_PATH.split("/"))


@pytest.fixture
def sample_manifest(tmp_path, side_file):
    input_csv = tmp_path / f"{_SAMPLE_MODE}_in.csv"
    input_csv.write_text(
        "#fofem-harness,%s,%s\n"
        % (_SAMPLE_MODE, MODE_SCHEMA_VERSIONS[_SAMPLE_MODE]),
        encoding="utf-8",
    )
    output_csv = tmp_path / f"{_SAMPLE_MODE}.csv"
    output_csv.write_text("case_id,outcome\nc1,ok\n", encoding="utf-8")
    return build_manifest(
        harness_mode=_SAMPLE_MODE,
        schema_version=MODE_SCHEMA_VERSIONS[_SAMPLE_MODE],
        compiler_identity="test-compiler",
        generator_toolchain="test-toolchain",
        platform="test-platform",
        architecture="test-arch",
        build_type="Debug",
        build_flags="-flags-",
        generating_command="fofem_test in.csv out",
        input_csv_paths=[str(input_csv)],
        output_csv_paths=[str(output_csv)],
        tolerance_policy_keys=["mortality.CroSco"],
        side_files={"species_table": side_file},
        now_utc_iso="2026-01-01T00:00:00+00:00",
    )


def test_build_manifest_empty_tolerance_keys_is_rejected(tmp_path):
    input_csv = tmp_path / "in.csv"
    input_csv.write_text("x", encoding="utf-8")
    output_csv = tmp_path / "out.csv"
    output_csv.write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance_policy_keys"):
        build_manifest(
            harness_mode="litter_eq", schema_version="1",
            compiler_identity="c", generator_toolchain="t", platform="p",
            architecture="a", build_type="Debug", build_flags="f",
            generating_command="cmd",
            input_csv_paths=[str(input_csv)], output_csv_paths=[str(output_csv)],
            tolerance_policy_keys=[],
        )


def test_build_manifest_has_every_required_field(sample_manifest):
    for field in REQUIRED_FIELDS:
        assert field in sample_manifest, f"missing {field}"


def test_check_pinned_sha_passes_on_pinned_checkout():
    # This repo's reference/fofem_cpp must be at the pinned SHA for any
    # Phase 2 work to proceed at all; if this fails, every other live-build
    # test in the suite would also be blocked, so failing loudly here first
    # is the more useful diagnostic.
    check_pinned_sha()  # must not raise
    assert current_upstream_sha() == PINNED_UPSTREAM_SHA


def test_compute_overlay_digests_covers_all_required_files():
    d = compute_overlay_digests()
    assert set(d["per_file"].keys()) == REQUIRED_OVERLAY_FILES
    assert len(d["combined"]) == 64


def test_compute_overlay_digests_is_deterministic():
    a = compute_overlay_digests()
    b = compute_overlay_digests()
    assert a == b


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_corrupted_manifest_missing_field_is_rejected(sample_manifest, missing_field):
    corrupted = copy.deepcopy(sample_manifest)
    del corrupted[missing_field]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any(missing_field in e for e in errors)


def test_corrupted_manifest_wrong_type_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["input_csv_sha256"] = "not-a-dict"
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("input_csv_sha256" in e for e in errors)


def test_git_dirty_status_detects_untracked_file(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "new_file.txt").write_text("x", encoding="utf-8")
    status = git_dirty_status(str(tmp_path))
    assert status["dirty"] is True
    assert status["untracked"] is True
    assert status["staged"] is False


def test_git_dirty_status_detects_staged_file(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True)
    status = git_dirty_status(str(tmp_path))
    assert status["dirty"] is True
    assert status["staged"] is True


def test_load_tolerance_policy_has_every_mode():
    policy = load_tolerance_policy()
    for mode in ("consume", "litter_eq", "shrub_herb_eq", "mortality", "bark_thick", "canopy_cover"):
        assert mode in policy


def test_manifest_dirty_true_requires_generator_source_hashes(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["pyfofem_dirty"] = {"dirty": True}
    corrupted["generator_source_sha256"] = {}
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("generator_source_sha256" in e for e in errors)


def test_divergences_for_keys_excludes_verified_and_unverified():
    policy = load_tolerance_policy()
    out = divergences_for_keys(policy, ["mortality.CroDam", "shrub_herb_eq.crown_foliage"])
    assert out == [], (
        f"'unverified' (CroDam) and 'verified_equivalent' (crown_foliage) "
        f"must not be reported as divergences: {out}"
    )


def test_divergences_for_keys_includes_known_divergent():
    policy = load_tolerance_policy()
    out = divergences_for_keys(policy, ["litter_eq.997"])
    assert len(out) == 1
    assert out[0].startswith("litter_eq.997: known_divergent_strict_xfail")


def test_manifest_empty_divergences_rejected_when_scenario_has_one(sample_manifest):
    """Deleting CroSco's applicable limitation must fail reconciliation."""
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["documented_expected_divergences"] = []
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("scenario-applicable divergences" in e for e in errors)


def test_manifest_non_list_divergences_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["documented_expected_divergences"] = "not-a-list"
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("documented_expected_divergences must be a list" in e for e in errors)


def test_manifest_non_string_divergence_entry_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["documented_expected_divergences"] = [123]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("must be a non-empty string" in e for e in errors)


def test_manifest_empty_tolerance_reference_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["tolerance_policy_reference"] = []
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("tolerance_policy_reference" in e for e in errors)


def test_manifest_not_a_dict_is_rejected():
    errors = validate_manifest("just a string", check_against_live_checkout=False)
    assert errors and "not a JSON object" in errors[0]


def test_manifest_unknown_tolerance_key_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["tolerance_policy_reference"] = ["not_a_real_mode.not_a_real_scenario"]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("not found in tolerance_policy.json" in e for e in errors)


def test_manifest_wrong_mode_tolerance_key_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["tolerance_policy_reference"] = ["litter_eq.997"]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("canonical Phase 2 scenario contract" in e for e in errors)


def test_tamper_altered_combined_overlay_digest_is_rejected(sample_manifest):
    """Structural check (no live checkout needed): the recorded combined
    digest must match the recomputation from the manifest's OWN per-file
    digests."""
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["overlay_combined_digest"] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("overlay_combined_digest does not match" in e for e in errors)


def test_tamper_altered_individual_overlay_digest_is_rejected(sample_manifest):
    # Corrupting one per-file digest also makes the manifest's own combined
    # digest internally inconsistent, so the structural check (which runs
    # before any live-checkout cross-check) already rejects it — both are
    # correct rejections of the same tamper; either is acceptable evidence.
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["overlay_file_digests"]))
    corrupted["overlay_file_digests"][key] = "1" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any(
        "overlay_combined_digest does not match" in e
        or f"overlay file {key!r} digest mismatch" in e
        for e in errors
    )


def test_tamper_altered_individual_overlay_digest_vs_live_tree_is_rejected(sample_manifest):
    """Same tamper as above, but with the combined digest recomputed to
    stay internally consistent, so this specifically exercises the
    live-checkout per-file cross-check rather than the structural one."""
    import hashlib

    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["overlay_file_digests"]))
    corrupted["overlay_file_digests"][key] = "1" * 64
    lines = sorted(f"{rel}:{h}" for rel, h in corrupted["overlay_file_digests"].items())
    corrupted["overlay_combined_digest"] = hashlib.sha256(
        "\n".join(lines).encode("utf-8")
    ).hexdigest()
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any(f"overlay file {key!r} digest mismatch" in e for e in errors)


def test_tamper_altered_input_hash_is_rejected(sample_manifest, tmp_path):
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["input_csv_sha256"]))
    corrupted["input_csv_sha256"][key] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True,
                                golden_dir=str(tmp_path))
    assert any("input CSV" in e and "digest mismatch" in e for e in errors)


def test_tamper_altered_output_hash_is_rejected(sample_manifest, tmp_path):
    """The exact scenario Codex demonstrated: replacing an output digest
    with zeros must now be caught (previously returned no errors)."""
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["output_csv_sha256"]))
    corrupted["output_csv_sha256"][key] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True,
                                golden_dir=str(tmp_path))
    assert any("output CSV" in e and "digest mismatch" in e for e in errors)


def test_tamper_altered_side_file_hash_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["side_file_sha256"]["species_table"]["sha256"] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any("side file 'species_table'" in e and "digest mismatch" in e for e in errors)


def test_tamper_altered_upstream_sha_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["upstream_cpp_sha"] = "1" * 40
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("only approved pinned SHA" in e for e in errors)


def test_tamper_extra_overlay_file_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["overlay_file_digests"]["not_a_real_file.txt"] = "a" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("unexpected extra files" in e for e in errors)


def test_tamper_missing_overlay_file_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["overlay_file_digests"]))
    del corrupted["overlay_file_digests"][key]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("missing required files" in e for e in errors)


def test_tamper_missing_output_file_is_rejected(sample_manifest):
    """A renamed/dropped output file must be rejected outright, not merely
    left un-cross-checked (the exact gap Codex flagged: a missing golden
    output currently passes as long as the remaining digests match)."""
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["output_csv_sha256"]))
    del corrupted["output_csv_sha256"][key]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("output_csv_sha256 missing required files" in e for e in errors)


def test_tamper_extra_output_file_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["output_csv_sha256"]["mortality_extra.csv"] = "a" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("output_csv_sha256 has unexpected extra files" in e for e in errors)


def test_tamper_renamed_input_file_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    key = next(iter(corrupted["input_csv_sha256"]))
    corrupted["input_csv_sha256"]["wrong_name.csv"] = corrupted["input_csv_sha256"].pop(key)
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("input_csv_sha256" in e and "must be exactly" in e for e in errors)


def test_tamper_side_file_missing_required_role_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    del corrupted["side_file_sha256"]["species_table"]
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("side_file_sha256 missing required role" in e for e in errors)


def test_tamper_side_file_extra_role_is_rejected(sample_manifest):
    """A mode carrying a side-file role it does not need (e.g. an
    emission-factor entry tacked onto a species-driven mode) must be
    rejected, not silently accepted."""
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["side_file_sha256"]["emission_factor_table"] = {
        "path": "reference/fofem_cpp/FOF_UNIX/Emission_Factors.csv",
        "sha256": "0" * 64,
    }
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("side_file_sha256 has unexpected role" in e for e in errors)


def test_tamper_side_file_wrong_species_table_path_is_rejected(sample_manifest):
    """The species-table side file must reference the ONE tracked
    FOF_SPP.CSV path exactly — a different (even if otherwise valid,
    existing) repo file must be rejected."""
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["side_file_sha256"]["species_table"]["path"] = (
        "reference/fofem_cpp/FOF_UNIX/Emission_Factors.csv"
    )
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("species_table']['path'] must be exactly" in e for e in errors)


def test_tamper_side_file_path_traversal_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["side_file_sha256"]["species_table"]["path"] = "../../../etc/passwd"
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("not a safe repo-relative path" in e for e in errors)


def test_tamper_side_file_absolute_path_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    corrupted["side_file_sha256"]["species_table"]["path"] = os.path.join(
        PROJECT_ROOT, "src", "pyfofem", "supporting_data", "FOFEM6.7", "FOF_SPP.CSV"
    )
    errors = validate_manifest(corrupted, check_against_live_checkout=False)
    assert any("not a safe repo-relative path" in e for e in errors)


def test_generator_source_files_contains_exactly_expected_set():
    """Every expected generation-critical source is present, and nothing
    extra — item 3 of the round 4 correction pass required _proc.py and
    tolerance_policy.json specifically (both previously omitted), plus any
    new output-column/policy-contract file this same pass introduces."""
    actual = {to_repo_relative(p) for p in GENERATOR_SOURCE_FILES}
    assert actual == _EXPECTED_GENERATOR_SOURCE_REPO_RELATIVE


@pytest.mark.parametrize("rel_path", sorted(_EXPECTED_GENERATOR_SOURCE_REPO_RELATIVE))
def test_tamper_generator_source_digest_mismatch_is_rejected_per_file(sample_manifest, rel_path):
    """Modifying ANY single generator-source file (not just an arbitrary
    one) must cause live validation to fail — parametrized explicitly over
    every file in the expected set, including the three item 3 flagged as
    previously untested: _proc.py, tolerance_policy.json, and the new
    _output_contract.py."""
    corrupted = copy.deepcopy(sample_manifest)
    assert corrupted["generator_source_sha256"], (
        "fixture precondition: repo must be dirty in this test environment "
        "for generator_source_sha256 to be populated"
    )
    assert rel_path in corrupted["generator_source_sha256"], (
        f"{rel_path!r} missing from a real generator_source_sha256 — "
        "GENERATOR_SOURCE_FILES and this test's expected set have drifted"
    )
    corrupted["generator_source_sha256"][rel_path] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any(f"generator source {rel_path!r}" in e and "digest mismatch" in e for e in errors)


def test_tamper_generator_source_extra_key_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    assert corrupted["generator_source_sha256"], (
        "fixture precondition: repo must be dirty in this test environment "
        "for generator_source_sha256 to be populated"
    )
    corrupted["generator_source_sha256"]["tests/not_a_real_generator.py"] = "a" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any("generator_source_sha256 key set does not match" in e for e in errors)


def test_tamper_generator_source_digest_mismatch_is_rejected(sample_manifest):
    corrupted = copy.deepcopy(sample_manifest)
    assert corrupted["generator_source_sha256"]
    key = next(iter(corrupted["generator_source_sha256"]))
    corrupted["generator_source_sha256"][key] = "0" * 64
    errors = validate_manifest(corrupted, check_against_live_checkout=True)
    assert any(f"generator source {key!r}" in e and "digest mismatch" in e for e in errors)


def test_live_checkout_validation_without_golden_dir_is_rejected(sample_manifest):
    """check_against_live_checkout=True with golden_dir omitted must fail
    clearly instead of silently skipping input/output authentication (the
    exact gap Codex flagged)."""
    errors = validate_manifest(sample_manifest, check_against_live_checkout=True)
    assert any("golden_dir is required" in e for e in errors)


def test_valid_manifest_passes_live_checkout_cross_check(sample_manifest, tmp_path):
    errors = validate_manifest(sample_manifest, check_against_live_checkout=True,
                                golden_dir=str(tmp_path))
    assert errors == []


def test_valid_manifest_passes_structural_validation(sample_manifest):
    errors = validate_manifest(sample_manifest, check_against_live_checkout=False)
    assert errors == []


def test_wrong_upstream_sha_manifest_is_rejected_before_any_hashing(monkeypatch, tmp_path):
    """A different checkout must fail closed at build_manifest() time, not
    merely be recorded and validated later."""
    import tests.cpp_parity_live._golden_manifest as gm

    monkeypatch.setattr(gm, "current_upstream_sha", lambda: "f" * 40)
    input_csv = tmp_path / "in.csv"
    input_csv.write_text("x", encoding="utf-8")
    output_csv = tmp_path / "out.csv"
    output_csv.write_text("y", encoding="utf-8")
    with pytest.raises(gm.ProvenanceError, match="pinned-SHA check failed"):
        gm.build_manifest(
            harness_mode="litter_eq", schema_version="1",
            compiler_identity="c", generator_toolchain="t", platform="p",
            architecture="a", build_type="Debug", build_flags="f",
            generating_command="cmd",
            input_csv_paths=[str(input_csv)], output_csv_paths=[str(output_csv)],
            tolerance_policy_keys=["litter_eq.998"],
        )


def test_write_manifest_round_trips_through_json(tmp_path, sample_manifest):
    path = os.path.join(str(tmp_path), "sub", "case.manifest.json")
    write_manifest(path, sample_manifest)
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == sample_manifest
    assert validate_manifest(loaded, check_against_live_checkout=False) == []


def test_mode_schema_versions_covers_every_valid_harness_mode():
    """Every mode the validator accepts must declare a schema version, or the
    per-mode equality check would silently not apply to it."""
    assert set(MODE_SCHEMA_VERSIONS) == set(VALID_HARNESS_MODES)


def test_schema_version_must_match_the_mode_that_declares_it(sample_manifest):
    """A version string that is valid for SOME mode is still wrong provenance
    if it is not THIS mode's own declared version.

    This is what makes ``mortality``'s v2 a genuine per-mode revision rather
    than a silent redefinition of "v1": a manifest claiming
    ``harness_mode=mortality, schema_version=1`` is rejected outright, not
    quietly accepted because "1" is a version some other mode uses.
    """
    others = sorted(
        set(MODE_SCHEMA_VERSIONS.values())
        - {MODE_SCHEMA_VERSIONS[_SAMPLE_MODE]}
    )
    assert others, "expected at least one other declared schema version"
    for version in others:
        bad = copy.deepcopy(sample_manifest)
        bad["schema_version"] = version
        errors = validate_manifest(bad)
        assert any("schema_version" in e for e in errors), (
            "%s manifest wrongly accepted schema_version %r"
            % (_SAMPLE_MODE, version)
        )
