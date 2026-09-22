#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_duff_forcing_correction.py - Python-side coverage for the Campbell
duff-forcing correction (``src/pyfofem/components/soil_heating.py``'s
``_duff_burn_rate``/``_duff_heat_fraction``/``_duff_burn_profile``), which
resolves two previously-undocumented soil-model defects (informally
labelled SOI-01: a static pre-fire duff depth used for heat transmission
over the whole burn, and SOI-02: a load-independent empirical burn rate)
and the F-53 percent-to-ratio conversion, now performed exactly once at
this forcing boundary. See ``gate0/04-findings.md`` F-69 (which records
this correction) and F-53's own RESOLVED update.

**Assertion classes (required module declaration):**

- Class **(a) Python contract/source-relation tests** - hand-derived
  checks against the pinned C++ ``DuffBurn()`` (``bur_brn.cpp:1950-1986``)
  and ``SD_HeatAdj()`` (``fof_sd.cpp:294-313``) formulas, percent-to-ratio
  conversion, zero-forcing edge cases, remaining-depth/consumed-percent
  boundaries, and the TIME-VARYING heat-transmission proof (the specific
  defect this pass fixes). These make no C++ EXECUTABLE parity claim on
  their own - see class (c) below and the companion harness self-tests in
  ``tests/cpp_parity_live/test_cpp_harness_contract.py`` (``test_soil_
  duff_burn_*``) for that.
- Class **(c) Campbell outcome characterization** - one test comparing the
  corrected (time-varying) forcing against a locally-reconstructed
  pre-correction (constant, pre-fire-depth) forcing at the full
  ``soil_heat_campbell()`` level, proving the change moves resulting soil
  temperatures in the physically-expected direction. This is explicitly a
  characterization of an in-repo behavior change, NOT a C++ parity
  acceptance test - no tolerance against a C++ golden is asserted here.

``_duff_burn_rate``/``_duff_heat_fraction``/``_duff_burn_profile`` are
private, scalar-only helpers with no array-broadcast contract; this module
does not claim or test array support for them.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pyfofem.components.soil_heating as sh
from pyfofem.components.soil_heating import (
    _CPP_INCH_TO_CM,
    _CPP_TPA_TO_KGM2,
    _duff_burn_profile,
    _duff_burn_rate,
    _duff_heat_fraction,
    soil_heat_campbell,
)

pytestmark = pytest.mark.soil_solver

_DEPTHS = list(range(1, 14))


def _cpp_duffburn(wdf_kgm2: float, dfm_ratio: float, pct_consumed: float) -> tuple:
    """
    Independent hand transcription of C++ ``DuffBurn()``
    (``bur_brn.cpp:1950-1986``), kept deliberately separate from
    :func:`pyfofem.components.soil_heating._duff_burn_rate` so these tests
    do not just re-assert the production function against itself.

    :param wdf_kgm2: Duff dry load (kg/m^2).
    :param dfm_ratio: Duff moisture as a ratio.
    :param pct_consumed: Duff consumed, percent, in [0, 100].
    :returns: ``(intensity_kw, duration_s, consumed_rate_kgm2_s)``.
    """
    if wdf_kgm2 <= 0.0 or dfm_ratio >= 1.96:
        return 0.0, 0.0, 0.0
    dfi = 11.25 - 4.05 * dfm_ratio
    ff = pct_consumed / 100.0
    denom = 7.5 - 2.7 * dfm_ratio
    tdf = 1.0e4 * ff * wdf_kgm2 / denom if denom != 0.0 else 0.0
    amt = (ff * wdf_kgm2) / tdf if tdf != 0.0 else 0.0
    return dfi, tdf, amt


def _duff_params(**overrides) -> dict:
    """
    Build a valid ``duff_params`` dict for :func:`_duff_burn_profile` /
    :func:`soil_heat_campbell`'s ``model="duff"`` path.

    :param overrides: Fields to override in the base dict.
    :returns: A ``duff_params`` dict.
    """
    base = dict(duff_load=5.0, duff_depth=2.0, duff_moisture=60.0, pct_consumed=50.0)
    base.update(overrides)
    return base


def _soil_params(soil_family: str = "coarse-silty", start_water: float = 0.10,
                  start_temp: float = 21.0, **overrides) -> dict:
    """
    Build a valid ``soil_params`` dict for :func:`soil_heat_campbell`.

    :param soil_family: One of ``_SOIL_FAMILY_DEFAULTS``'s keys.
    :param start_water: Starting volumetric water content (m3/m3).
    :param start_temp: Starting soil temperature (degC).
    :param overrides: Additional fields to override in the base dict.
    :returns: A ``soil_params`` dict.
    """
    base = dict(soil_family=soil_family, start_water=start_water, start_temp=start_temp)
    base.update(overrides)
    return base


def test_conversion_constants_match_cpp_source_relation():
    """Class (a). ``_CPP_TPA_TO_KGM2`` == 1/4.46 (``fof_util.cpp:543-549``,
    NOT pyfofem's own usual 4.4609) and ``_CPP_INCH_TO_CM`` ==
    100/39.37 (``fof_sd.cpp:298-299`` via ``fof_sh.cpp:209-215``'s
    ``InchtoMeter``, NOT the idealised 2.54)."""
    assert _CPP_TPA_TO_KGM2 == pytest.approx(1.0 / 4.46, rel=0, abs=1e-15)
    assert _CPP_INCH_TO_CM == pytest.approx(100.0 / 39.37, rel=0, abs=1e-12)
    assert _CPP_INCH_TO_CM != pytest.approx(2.54, abs=1e-6)


def test_duff_burn_profile_final_remaining_depth_matches_consumed_percent():
    """Class (a). ``post_depth_cm`` == ``pre_depth_cm * (1 - pct/100)`` for
    a burning case, and remaining depth never goes negative even at full
    (100%) consumption."""
    profile = _duff_burn_profile(_duff_params(duff_depth=3.0, pct_consumed=80.0))
    expected_post = profile["pre_depth_cm"] * 0.20
    assert profile["post_depth_cm"] == pytest.approx(expected_post)
    assert profile["post_depth_cm"] >= 0.0

    full = _duff_burn_profile(_duff_params(duff_depth=3.0, pct_consumed=100.0))
    assert full["post_depth_cm"] == pytest.approx(0.0, abs=1e-9)
    assert full["remaining_depth_fn"](full["duration_s"]) >= 0.0


def test_duff_burn_profile_flux_fn_finite_nonnegative_over_and_after_duration():
    """Class (a). ``flux_fn`` is finite and non-negative throughout
    ``[0, duration_s]`` and exactly zero for any t beyond it."""
    profile = _duff_burn_profile(_duff_params(pct_consumed=60.0))
    duration_s = profile["duration_s"]
    assert duration_s > 0.0
    for t in np.linspace(0.0, duration_s, 11):
        val = profile["flux_fn"](float(t))
        assert math.isfinite(val)
        assert val >= 0.0
    assert profile["flux_fn"](duration_s + 1.0) == 0.0
    assert profile["flux_fn"](duration_s * 100.0) == 0.0


def test_duff_burn_profile_heat_fraction_constant_at_full_consumption_endpoint():
    """Class (a) boundary. At 100% consumption, remaining depth reaches
    0 cm exactly at t=duration_s -- heat_fraction_fn there equals
    ``_duff_heat_fraction(0.0)``, the maximum transmission fraction."""
    profile = _duff_burn_profile(_duff_params(duff_depth=2.0, pct_consumed=100.0))
    duration_s = profile["duration_s"]
    assert profile["remaining_depth_fn"](duration_s) == pytest.approx(0.0, abs=1e-9)
    assert profile["heat_fraction_fn"](duration_s) == pytest.approx(
        _duff_heat_fraction(0.0), abs=1e-9,
    )


def test_duff_burn_profile_heat_fraction_increases_as_duff_thins_time_varying_proof():
    """Class (a). THE defect this pass fixes (SOI-01): the heat-
    transmission fraction must be evaluated at the TIME-VARYING remaining
    duff depth, not held constant at the pre-fire value for the whole
    burn. Since ``_duff_heat_fraction`` is monotonically decreasing in
    depth (a thicker duff layer insulates more), and remaining depth
    monotonically decreases as the duff burns down, the fraction (and
    hence flux, since intensity_w is time-invariant) must strictly
    increase from t=0 to t=duration_s for any partial-consumption case."""
    profile = _duff_burn_profile(_duff_params(duff_depth=3.0, pct_consumed=70.0))
    duration_s = profile["duration_s"]
    frac_start = profile["heat_fraction_fn"](0.0)
    frac_mid = profile["heat_fraction_fn"](duration_s / 2.0)
    frac_end = profile["heat_fraction_fn"](duration_s)
    assert frac_start < frac_mid < frac_end

    flux_start = profile["flux_fn"](0.0)
    flux_end = profile["flux_fn"](duration_s - 1e-6)
    assert flux_start < flux_end


def test_duff_burn_profile_missing_duff_load_raises_value_error():
    """Class (a). ``duff_load`` is a genuine required input (item 2 of the
    task): a caller omitting it must get a precise ValueError, never the
    old silent load-independent fallback."""
    params = _duff_params()
    del params["duff_load"]
    with pytest.raises(ValueError, match="duff_load"):
        _duff_burn_profile(params)


def test_duff_burn_profile_non_finite_duff_load_raises_value_error():
    """Class (a). NaN/inf ``duff_load`` is rejected, not silently
    propagated into zero-forcing (that is reserved for finite <= 0)."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="duff_load"):
            _duff_burn_profile(_duff_params(duff_load=bad))


def test_duff_burn_profile_pct_consumed_out_of_range_raises_value_error():
    """Class (a). C++'s own out-of-range fallback (``ff=0.837-0.426*dfm``,
    ``bur_brn.cpp:1974-1975``) exists only for a standalone-Burnup-
    without-FOFEM path with no Campbell-contract equivalent, so an
    out-of-range ``pct_consumed`` is rejected outright rather than
    silently ported or clamped -- per the task's explicit "do not invent a
    clamp" instruction."""
    for bad in (-1.0, 100.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pct_consumed"):
            _duff_burn_profile(_duff_params(pct_consumed=bad))


def test_duff_burn_profile_percent_moisture_converted_to_ratio_exactly_once():
    """Class (a). F-53's fix: a realistic whole-percent duff_moisture
    (45%) must reach ``_duff_burn_rate`` as the ratio 0.45, not 45.0 or
    0.0045 -- verified by comparing the profile's own intensity_w against
    an independently hand-derived DuffBurn call using ratio 0.45
    explicitly (never re-deriving via the production ratio conversion
    itself)."""
    duff_load = 6.0
    profile = _duff_burn_profile(
        _duff_params(duff_load=duff_load, duff_moisture=45.0, pct_consumed=55.0),
    )
    wdf = duff_load * (1.0 / 4.46)
    dfi_kw, tdf, _ = _cpp_duffburn(wdf, 0.45, 55.0)
    assert profile["intensity_w"] == pytest.approx(dfi_kw * 1000.0, rel=1e-9)
    assert profile["duration_s"] == pytest.approx(tdf, rel=1e-9)

    # An accidental unconverted-percent bug (ratio=45.0) would trip
    # DuffBurn's own dfm>=1.96 zero-forcing guard -- confirm that would
    # look completely different from the real, correctly-converted result.
    dfi_wrong_kw, tdf_wrong, _ = _cpp_duffburn(wdf, 45.0, 55.0)
    assert (dfi_wrong_kw, tdf_wrong) == (0.0, 0.0)
    assert profile["intensity_w"] != 0.0


def test_duff_burn_profile_pre_depth_uses_cpp_inch_to_cm_conversion():
    """Class (a). ``pre_depth_cm`` uses the exact C++ ``100/39.37``
    conversion, not the idealised 2.54."""
    profile = _duff_burn_profile(_duff_params(duff_depth=2.0))
    assert profile["pre_depth_cm"] == pytest.approx(2.0 * (100.0 / 39.37), rel=1e-12)
    assert profile["pre_depth_cm"] != pytest.approx(2.0 * 2.54, abs=1e-6)


def test_duff_burn_profile_zero_or_negative_load_still_requires_other_fields():
    """Class (a) contract note. Zero-forcing from load does not bypass
    accessing the other required fields: unlike ``duff_load`` (the one
    field with an explicit presence check, item 2 of the task),
    ``pct_consumed``/``duff_depth``/``duff_moisture`` are read via direct
    dict indexing, so a missing key surfaces as ``KeyError`` -- not a
    silently-defaulted or zero-forced value."""
    params = _duff_params(duff_load=0.0)
    del params["pct_consumed"]
    with pytest.raises(KeyError, match="pct_consumed"):
        _duff_burn_profile(params)


def test_duff_burn_profile_zero_or_negative_load_yields_zero_forcing_not_error():
    """Class (a). A finite ``duff_load <= 0`` (including negative) is a
    VALID input yielding zero forcing -- matching C++'s own
    ``if (wdf <= 0.0 ...) return;`` guard exactly -- not an error case
    (item 4 of the task: only missing/non-finite duff_load raises)."""
    for load in (0.0, -3.0):
        profile = _duff_burn_profile(_duff_params(duff_load=load))
        assert profile["duration_s"] == 0.0
        assert profile["intensity_w"] == 0.0
        assert profile["consumed_rate_kgm2_s"] == 0.0
        assert profile["flux_fn"](0.0) == 0.0
        assert profile["post_depth_cm"] == profile["pre_depth_cm"]


def test_duff_burn_profile_zero_percent_consumed_yields_zero_forcing():
    """Class (a) boundary. C++ ``SD_Mngr_New`` supplies duff heat only
    while ``i_ClockSec < i_BurnTime`` (``fof_sd.cpp:123-131``). Therefore
    a 0%-consumed case has a zero burn duration and zero flux even at t=0;
    it must not leave a numerical impulse for ``solve_ivp`` to sample."""
    profile = _duff_burn_profile(_duff_params(pct_consumed=0.0))
    assert profile["duration_s"] == 0.0
    assert profile["flux_fn"](0.0) == 0.0
    assert profile["flux_fn"](1.0) == 0.0


def test_duff_burn_rate_matches_hand_derived_dufburn_formula():
    """Class (a). A normal (non-boundary) case matches an independently
    hand-transcribed DuffBurn() formula for all three outputs."""
    wdf, dfm, pct = 5.0 / 4.46, 0.45, 55.0
    intensity_kw, duration_s, rate = _duff_burn_rate(wdf, dfm, pct)
    exp_intensity, exp_duration, exp_rate = _cpp_duffburn(wdf, dfm, pct)
    assert intensity_kw == pytest.approx(exp_intensity, rel=1e-12)
    assert duration_s == pytest.approx(exp_duration, rel=1e-12)
    assert rate == pytest.approx(exp_rate, rel=1e-12)


def test_duff_burn_rate_zero_at_moisture_threshold():
    """Class (a). C++'s exact ``dfm >= 1.96`` non-burning boundary
    (``bur_brn.cpp:1960``) -- at and above the threshold, all zero; just
    below, nonzero."""
    assert _duff_burn_rate(5.0, 1.96, 50.0) == (0.0, 0.0, 0.0)
    assert _duff_burn_rate(5.0, 2.5, 50.0) == (0.0, 0.0, 0.0)
    below = _duff_burn_rate(5.0, np.nextafter(1.96, 0.0), 50.0)
    assert below != (0.0, 0.0, 0.0)


def test_duff_burn_rate_zero_at_zero_or_negative_load():
    """Class (a). C++'s exact ``wdf <= 0`` guard (``bur_brn.cpp:1960``)."""
    assert _duff_burn_rate(0.0, 0.45, 50.0) == (0.0, 0.0, 0.0)
    assert _duff_burn_rate(-1.0, 0.45, 50.0) == (0.0, 0.0, 0.0)


def test_duff_heat_fraction_clips_to_0_100_percent_range():
    """Class (a). ``SD_HeatAdj``'s regression is clipped to [0, 100]
    before scaling to a [0, 1] fraction -- a very thin (near-zero) depth
    must not exceed a fraction of 1.0, and a very thick depth must not go
    negative."""
    assert 0.0 <= _duff_heat_fraction(0.0) <= 1.0
    assert 0.0 <= _duff_heat_fraction(1000.0) <= 1.0
    assert _duff_heat_fraction(1000.0) == pytest.approx(0.0, abs=1e-6)


def test_duff_heat_fraction_matches_hand_derived_sdheatadj_formula():
    """Class (a). Matches an independently hand-transcribed SD_HeatAdj()
    regression (``fof_sd.cpp:294-313``) at several depths, not just re-
    deriving the production formula against itself."""
    for depth_cm in (0.5, 2.0, 5.0, 10.0):
        y0, a, b, c, d = -1.6996, 32.7652, 7.4601, 68.9349, 0.6077
        r = y0 + a * math.exp(-b * depth_cm) + c * math.exp(-d * depth_cm)
        expected = min(max(r, 0.0), 100.0) * 0.01
        assert _duff_heat_fraction(depth_cm) == pytest.approx(expected, rel=1e-12)


def test_duff_heat_fraction_monotonically_decreasing_in_depth():
    """Class (a). A thicker duff layer insulates more -- the fraction
    strictly decreases as depth increases, over the physically meaningful
    range. This monotonicity is what makes the time-varying-forcing fix
    (item 1) actually matter: a thinning layer transmits more heat over
    time, not the same amount throughout."""
    # Excludes the top (depth 0, saturates the clip at exactly 1.0) and
    # bottom (depth >~6cm, saturates the clip at exactly 0.0) of the
    # curve's clipped range, where flat rather than strictly-decreasing
    # is the correct, expected shape.
    depths = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0]
    fractions = [_duff_heat_fraction(d) for d in depths]
    assert all(earlier > later for earlier, later in zip(fractions, fractions[1:]))


def test_partial_consumption_temperature_differs_from_constant_prefire_forcing_characterization(monkeypatch):
    """
    Class (c). Campbell OUTCOME characterization, not C++ parity.

    Proves, at the full public ``soil_heat_campbell()`` level, that a
    partial-consumption duff scenario's resulting soil temperatures
    genuinely differ from -- and are never lower than -- what the
    PRE-CORRECTION constant-pre-fire-depth forcing (SOI-01) would have
    produced, in the physically-expected direction: since
    ``_duff_heat_fraction`` decreases with depth and the corrected profile
    's remaining depth only decreases over time, the corrected time-
    varying fraction is always >= a fraction pinned at the (thicker)
    pre-fire depth for the whole burn -- so the corrected forcing never
    under-delivers heat relative to the old behavior.

    Reconstructs the OLD constant-forcing behavior locally via
    monkeypatching ``_duff_heat_fraction`` to always return the pre-fire
    depth's fraction -- it does NOT revive the removed
    ``_duff_flux_and_duration``/``_make_duff_flux_fn`` functions, and
    touches no production code.
    """
    duff_params = _duff_params(duff_load=5.0, duff_depth=2.0,
                                duff_moisture=45.0, pct_consumed=60.0)
    soil_params = _soil_params()

    df_new = soil_heat_campbell("duff", duff_params, soil_params, _DEPTHS)

    profile = _duff_burn_profile(duff_params)
    prefire_fraction = _duff_heat_fraction(profile["pre_depth_cm"])
    assert profile["duration_s"] > 0.0  # a real, non-trivial burn

    monkeypatch.setattr(sh, "_duff_heat_fraction", lambda depth_cm: prefire_fraction)
    df_old_style = soil_heat_campbell("duff", duff_params, soil_params, _DEPTHS)

    new_vals = df_new.to_numpy()
    old_vals = df_old_style.to_numpy()
    assert np.isfinite(new_vals).all()
    assert np.isfinite(old_vals).all()
    # F-70 (third round) fixed a real bulk_density/particle_density unit
    # transcription error (see gate0/04-findings.md) that also corrected
    # the coupled solver's own cool-down timing -- the two runs' trajectory
    # LENGTHS (n_time_indices, driven by _soi_done_duff's cooldown check)
    # can now legitimately differ, so the elementwise comparisons below
    # are taken over the shared overlapping window, not the full arrays.
    n_common = min(len(new_vals), len(old_vals))
    new_common = new_vals[:n_common]
    old_common = old_vals[:n_common]
    assert not np.allclose(new_common, old_common)
    # Corrected forcing delivers heat at least as fast everywhere -- never
    # colder than the pre-correction constant-forcing reconstruction.
    assert (new_common >= old_common - 1e-9).all()
    assert new_vals.max() > old_vals.max()
