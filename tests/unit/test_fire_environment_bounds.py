#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fire_environment_bounds.py - Phase 7 item B: characterization of the
three currently inconsistent fire-environment bounds-handling paths
(``docs/CODEBASE.md`` Gotcha #23):

1. ``burnup.py::_check_fire()`` - fully defined, symmetric
   reject-both-sides-of-every-bound, but confirmed here (by a real,
   repository-wide grep at test time, not a hardcoded belief) to be
   NEVER CALLED anywhere in production.
2. ``burnup_calcs.py::_run_burnup_cell()``'s own inline pre/post checks -
   asymmetric clip-upper (``burnup_limit_adjust`` codes 1-6)/reject-lower
   (``burnup_error`` codes 10-14) for ``fistart``/``ti``/``u``/``tamb_c``;
   ``dfm`` inverted (clipped low, rejected high, and only checked at all
   when duff is present); ``d`` (fuel bed depth) clipped on *both* sides
   with no rejection path at all.
3. ``burnup_calcs.py::gen_burnup_in_file()`` - unconditional clip-both-
   sides (``max(lo, min(val, hi))``) for 5 of the 6 fire bounds, with NO
   rejection path and, notably, **no ``dfm`` parameter at all** - the
   function cannot even express a duff-moisture value, so path 3 never
   applies that bound.

Per the approved Phase 7 plan: this module documents and asserts the
PRESENT divergence as characterization (so a later consolidation has a
reliable before/after baseline) - it does **not** normalize or fix
production behavior. No strict xfail is added here because no target
"correct" consolidated contract has been approved yet (Gotcha #23 and
the Next Steps list both record this as an open decision requiring
explicit sign-off, not a decided desired behavior a test could pin).

The path-3 (``gen_burnup_in_file()``) tests use the ``tmp_path`` fixture
NAME for compatibility with pytest's own idioms, but this module
OVERRIDES it (see :func:`tmp_path` below) with a repository-local
directory, per the Phase 7 correction pass's filesystem boundary
(2026-09-06) - never the system/user temporary directory pytest's own
built-in ``tmp_path`` resolves to.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pytest
import sys
import pyfofem.components.burnup  # noqa: F401  (ensures sys.modules entry exists)
_burnup_module = sys.modules['pyfofem.components.burnup']
from pyfofem.components import burnup_calcs as bc
from pyfofem.components.burnup import BurnupValidationError, _FIRE_BOUNDS, _check_fire
from tests.cpp_parity_live._scratch import scratch_tempdir
_CHECK_FIRE_PARAMS = (
    ('fistart', 'fistart'),
    ('ti', 'ti'),
    ('u', 'u'),
    ('d', 'd'),
    ('tamb_c', 'tamb_c'),
)
_BRN_FIELD_NAME = {
    'intensity': 'INTENSITY',
    'ig_time': 'IG_TIME',
    'windspeed': 'WINDSPEED',
    'depth': 'DEPTH',
    'ambient_temp': 'AMBIENT_TEMP',
}


def _nominal_check_fire_kwargs(**overrides):
    """Build a fully in-bounds ``_check_fire()`` call, one override at a
    time isolating exactly one bound."""
    kwargs = dict(fistart=1000.0, ti=60.0, u=1.0, d=0.3, tamb_c=20.0,
                  wdf_load=0.0, dfm=0.5)
    kwargs.update(overrides)
    return kwargs


def _nominal_ckw(**overrides):
    """Build a minimal, otherwise fully in-bounds ``ckw`` dict for
    ``_run_burnup_cell()`` - identical shape to
    ``test_run_burnup_cell_error_codes.py``'s own ``_base_ckw()`` (kept
    independent/duplicated intentionally: that module's fixture is a
    private implementation detail this one should not import and couple
    to)."""
    ckw = dict(
        fuel_loadings_bu={'litter': 1.0},
        fuel_moistures_bu={'litter': 0.10},
        rotten_keys={},
        density_map={},
        intensity_kw=1000.0,
        frt_s=60.0,
        ws=1.0,
        fb_depth=0.3,
        amb_temp=20.0,
        duf_loading_si=0.0,
        duf_moist_frac=0.5,
        duf_pct_consumed=-1.0,
        hsf_consumed_si=0.0,
        brafol_consumed_si=0.0,
        burnup_dt=15.0,
        bkw=dict(r0=1.83, dr=0.4, max_times=3000, fint_switch=15.0, validate=True),
    )
    ckw.update(overrides)
    return ckw


def _parse_brn(content: str) -> dict:
    """Parse a generated ``#NAME value`` .brn text blob into a
    ``{name: float}`` dict - shared by this module's tests and
    ``test_gen_burnup_in_file.py``'s own richer content assertions.

    :param content: Raw file text as produced by
        :func:`~pyfofem.components.burnup_calcs.gen_burnup_in_file`.
    :return: Dict mapping each ``#NAME`` token to its parsed float value.
    """
    parsed = {}
    for line in content.splitlines():
        match = re.match(r'#(\S+)\s+(.+)$', line)
        assert match, f'unparseable .brn line: {line!r}'
        parsed[match.group(1)] = float(match.group(2))
    return parsed


@pytest.mark.parametrize('bound_key,kwarg', _CHECK_FIRE_PARAMS)
def test_check_fire_boundary_values_are_valid_inclusive(bound_key, kwarg):
    """Path 1 exact-boundary: ``_check_fire()`` uses strict ``<``/``>``,
    so a value exactly AT either bound is VALID (inclusive range) -
    unlike ``_check_fuel()``'s ``<=``/``>=`` (exclusive range) on the
    same kind of bound tuple. Both boundary values are exercised."""
    lo, hi, _label = _FIRE_BOUNDS[bound_key]
    _check_fire(**_nominal_check_fire_kwargs(**{kwarg: lo}))  # no raise
    _check_fire(**_nominal_check_fire_kwargs(**{kwarg: hi}))  # no raise


def test_check_fire_dfm_only_checked_when_duff_present():
    """Path 1's ``dfm`` bound is gated on ``wdf_load > 0.0`` - out-of-range
    duff moisture with zero duff loading must not raise."""
    _check_fire(**_nominal_check_fire_kwargs(wdf_load=0.0, dfm=99.0))  # no raise
    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(wdf_load=5.0, dfm=99.0))


def test_check_fire_fistart_exact_boundary_and_float_neighbors():
    """``_FIRE_BOUNDS['fistart']`` at the exact 40.0/1e5 boundary plus the
    immediate float neighbors (``np.nextafter``), as item B explicitly
    requires."""
    lo, hi, _label = _FIRE_BOUNDS['fistart']
    _check_fire(**_nominal_check_fire_kwargs(fistart=lo))  # exact lower: valid
    _check_fire(**_nominal_check_fire_kwargs(fistart=hi))  # exact upper: valid
    just_above_lo = float(np.nextafter(lo, np.inf))
    just_below_hi = float(np.nextafter(hi, -np.inf))
    _check_fire(**_nominal_check_fire_kwargs(fistart=just_above_lo))  # valid
    _check_fire(**_nominal_check_fire_kwargs(fistart=just_below_hi))  # valid
    just_below_lo = float(np.nextafter(lo, -np.inf))
    just_above_hi = float(np.nextafter(hi, np.inf))
    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(fistart=just_below_lo))
    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(fistart=just_above_hi))


def test_check_fire_invalid_type_raises_typeerror():
    """Path 1 has no explicit type validation - a non-numeric value falls
    straight into the native ``<``/``>`` comparison and raises
    ``TypeError``, not a domain-specific error."""
    with pytest.raises(TypeError):
        _check_fire(**_nominal_check_fire_kwargs(fistart='not-a-number'))


def test_check_fire_is_never_called_anywhere_in_production():
    """Confirms Gotcha #23's "never called" claim directly, at test time,
    via a real repository grep - not an inherited/trusted belief. Only
    the ``def _check_fire(`` line itself (and this test's own reference)
    may mention the name as a call; scans every first-party source file
    under ``src/pyfofem/``."""
    src_root = os.path.join(os.path.dirname(_burnup_module.__file__), '..')
    call_sites = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, 'r', encoding='utf-8') as fh:
                for lineno, line in enumerate(fh, start=1):
                    if '_check_fire(' in line and 'def _check_fire(' not in line:
                        call_sites.append(f'{path}:{lineno}')
    assert call_sites == [], (
        f"_check_fire() now has real call site(s): {call_sites} - "
        "Gotcha #23/this module's dead-code characterization is stale, "
        "update it rather than silently leaving this test broken"
    )


def test_check_fire_length_one_array_is_accepted_longer_array_raises():
    """Path 1's comparisons were written for scalars: a length-1 NumPy
    array coerces fine via ``__bool__`` on the resulting single-element
    boolean array, but a genuinely multi-element array raises NumPy's own
    ambiguous-truth-value ``ValueError`` - there is no vectorized/shape
    contract here at all (unlike the public consm_*/mort_* API family in
    item C's coverage)."""
    _check_fire(**_nominal_check_fire_kwargs(fistart=np.array([1000.0])))  # no raise
    with pytest.raises(ValueError):
        _check_fire(**_nominal_check_fire_kwargs(fistart=np.array([1000.0, 2000.0])))


def test_check_fire_nominal_accepts_and_returns_fistart_ti_unchanged():
    """Path 1 nominal: an in-bounds call returns ``(fistart, ti)``
    unchanged, per its own documented contract."""
    result = _check_fire(**_nominal_check_fire_kwargs())
    assert result == (1000.0, 60.0)


@pytest.mark.parametrize('bound_key,kwarg', _CHECK_FIRE_PARAMS)
def test_check_fire_rejects_just_below_and_just_above(bound_key, kwarg):
    """Path 1 just-below/just-above: every bound is rejected symmetrically
    on both sides - the defining trait that distinguishes path 1 from
    paths 2 and 3."""
    lo, hi, _label = _FIRE_BOUNDS[bound_key]
    step = (hi - lo) * 1e-6 if hi > lo else 1e-6
    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(**{kwarg: lo - step}))
    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(**{kwarg: hi + step}))


@pytest.mark.parametrize('kwarg,bound_key', [
    ('intensity', 'fistart'),
    ('ig_time', 'ti'),
    ('windspeed', 'u'),
    ('depth', 'd'),
    ('ambient_temp', 'tamb_c'),
])
def test_gen_burnup_in_file_clips_both_sides_unconditionally(tmp_path, kwarg, bound_key):
    """Path 3 characterization: every one of the 5 fire-environment
    parameters ``gen_burnup_in_file()`` accepts is silently clipped on
    BOTH sides (``max(lo, min(val, hi))``) - never rejected, and with no
    adjust/error code returned at all (the function returns ``None``)."""
    lo, hi, _label = _FIRE_BOUNDS[bound_key]
    out_path = tmp_path / 'below.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path), **{kwarg: lo - 100.0})
    content = out_path.read_text()
    written = _parse_brn(content)
    assert written[_BRN_FIELD_NAME[kwarg]] == pytest.approx(lo)

    out_path2 = tmp_path / 'above.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path2), **{kwarg: hi + 100.0})
    content2 = out_path2.read_text()
    written2 = _parse_brn(content2)
    assert written2[_BRN_FIELD_NAME[kwarg]] == pytest.approx(hi)


def test_gen_burnup_in_file_fistart_exact_boundary_and_float_neighbors(tmp_path):
    """``_FIRE_BOUNDS['fistart']`` exact-boundary + float-neighbor
    characterization for path 3."""
    lo, hi, _label = _FIRE_BOUNDS['fistart']
    for label, value in (
        ('at_lo', lo), ('at_hi', hi),
        ('just_below_lo', float(np.nextafter(lo, -np.inf))),
        ('just_above_lo', float(np.nextafter(lo, np.inf))),
        ('just_below_hi', float(np.nextafter(hi, -np.inf))),
        ('just_above_hi', float(np.nextafter(hi, np.inf))),
    ):
        out_path = tmp_path / f'{label}.brn'
        bc.gen_burnup_in_file(out_brn_path=str(out_path), intensity=value)
        written = _parse_brn(out_path.read_text())
        expected = min(max(value, lo), hi)
        assert written['INTENSITY'] == pytest.approx(expected)


def test_gen_burnup_in_file_has_no_duff_moisture_parameter_at_all():
    """Path 3 cannot even express ``dfm`` - unlike paths 1 and 2, which
    both validate/adjust duff moisture, ``gen_burnup_in_file()``'s
    signature has no such parameter, so this bound is silently absent
    from the third path entirely (not merely handled differently)."""
    import inspect
    sig = inspect.signature(bc.gen_burnup_in_file)
    assert 'dfm' not in sig.parameters
    assert 'duff_moist' not in sig.parameters


def test_gen_burnup_in_file_invalid_type_raises_typeerror(tmp_path):
    """Path 3 also has no explicit type validation - a non-numeric value
    fails inside ``max(lo, min(val, hi))`` with a native ``TypeError``,
    matching paths 1 and 2's equivalent failure mode - the third
    documented instance of the SAME underlying gap (no shared input
    validation across any of the three paths)."""
    out_path = tmp_path / 'bad.brn'
    with pytest.raises(TypeError):
        bc.gen_burnup_in_file(out_brn_path=str(out_path), intensity='not-a-number')
    assert not out_path.exists()


def test_run_burnup_cell_d_is_clipped_on_both_sides_with_no_rejection_path():
    """Path 2's ``d`` (fuel bed depth) is the one bound with NO rejection
    path at all - both below-minimum and above-maximum values are
    silently clipped, always recording adjust code 4, never a
    ``burnup_error``."""
    lo, hi, _label = _FIRE_BOUNDS['d']
    below = bc._run_burnup_cell(_nominal_ckw(fb_depth=lo - 0.05))
    assert below['burnup_error'] == 0
    assert '4' in str(below['burnup_limit_adjust'])
    above = bc._run_burnup_cell(_nominal_ckw(fb_depth=hi + 1.0))
    assert above['burnup_error'] == 0
    assert '4' in str(above['burnup_limit_adjust'])


def test_run_burnup_cell_dfm_bound_is_inverted_clip_low_reject_high():
    """Path 2's ``dfm`` is the one bound with INVERTED clip/reject
    polarity versus fistart/ti/u/tamb_c: below-minimum duff moisture is
    silently clipped up (adjust code 6), while above-maximum duff
    moisture is rejected (error code 14) - and both only apply when duff
    is actually present."""
    lo, hi, _label = _FIRE_BOUNDS['dfm']
    clipped = bc._run_burnup_cell(_nominal_ckw(
        duf_loading_si=5.0, duf_moist_frac=lo - 0.05,
    ))
    assert clipped['burnup_error'] == 0
    assert '6' in str(clipped['burnup_limit_adjust'])

    rejected = bc._run_burnup_cell(_nominal_ckw(
        duf_loading_si=5.0, duf_moist_frac=hi + 0.05,
    ))
    assert rejected['burnup_error'] == 14

    # Absent duff: neither the clip nor the reject branch fires, even for
    # the exact same out-of-range moisture fraction values.
    absent_low = bc._run_burnup_cell(_nominal_ckw(
        duf_loading_si=0.0, duf_moist_frac=lo - 0.05,
    ))
    assert absent_low['burnup_error'] == 0
    assert absent_low['burnup_limit_adjust'] == 0
    absent_high = bc._run_burnup_cell(_nominal_ckw(
        duf_loading_si=0.0, duf_moist_frac=hi + 0.05,
    ))
    assert absent_high['burnup_error'] == 0
    assert absent_high['burnup_limit_adjust'] == 0


def test_run_burnup_cell_fistart_exact_boundary_and_float_neighbors():
    """``_FIRE_BOUNDS['fistart']`` exact-boundary + float-neighbor
    characterization for path 2, as item B explicitly requires."""
    lo, hi, _label = _FIRE_BOUNDS['fistart']
    at_lo = bc._run_burnup_cell(_nominal_ckw(intensity_kw=lo))
    assert at_lo['burnup_error'] == 0 and at_lo['burnup_limit_adjust'] == 0
    at_hi = bc._run_burnup_cell(_nominal_ckw(intensity_kw=hi))
    assert at_hi['burnup_error'] == 0 and at_hi['burnup_limit_adjust'] == 0

    just_below_lo = float(np.nextafter(lo, -np.inf))
    rejected = bc._run_burnup_cell(_nominal_ckw(intensity_kw=just_below_lo))
    assert rejected['burnup_error'] == 10

    just_above_hi = float(np.nextafter(hi, np.inf))
    clipped = bc._run_burnup_cell(_nominal_ckw(intensity_kw=just_above_hi))
    assert clipped['burnup_error'] == 0
    assert '1' in str(clipped['burnup_limit_adjust'])

    just_above_lo = float(np.nextafter(lo, np.inf))
    ok = bc._run_burnup_cell(_nominal_ckw(intensity_kw=just_above_lo))
    assert ok['burnup_error'] == 0 and ok['burnup_limit_adjust'] == 0

    just_below_hi = float(np.nextafter(hi, -np.inf))
    ok2 = bc._run_burnup_cell(_nominal_ckw(intensity_kw=just_below_hi))
    assert ok2['burnup_error'] == 0 and ok2['burnup_limit_adjust'] == 0


def test_run_burnup_cell_invalid_type_raises_typeerror():
    """Path 2 also has no explicit type validation on fire-environment
    fields - a string value falls straight into the native comparison
    (``intensity > _fi_hi``) and raises ``TypeError``, matching path 1's
    equivalent failure mode."""
    with pytest.raises(TypeError):
        bc._run_burnup_cell(_nominal_ckw(intensity_kw='not-a-number'))


@pytest.mark.parametrize('field,bound_key,error_code', [
    ('intensity_kw', 'fistart', 10),
    ('frt_s', 'ti', 11),
    ('ws', 'u', 12),
    ('amb_temp', 'tamb_c', 13),
])
def test_run_burnup_cell_lower_bound_is_rejected_not_clipped(field, bound_key, error_code):
    """Path 2 lower-bound characterization: the same 4 fields are REJECTED
    (never clipped) when below the lower bound - the asymmetry that
    defines path 2 as distinct from path 1 (symmetric reject) and path 3
    (symmetric clip)."""
    lo, hi, _label = _FIRE_BOUNDS[bound_key]
    result = bc._run_burnup_cell(_nominal_ckw(**{field: lo - 1.0}))
    assert result['burnup_error'] == error_code


@pytest.mark.parametrize('field,bound_key,adjust_code', [
    ('intensity_kw', 'fistart', 1),
    ('frt_s', 'ti', 2),
    ('ws', 'u', 3),
    ('amb_temp', 'tamb_c', 5),
])
def test_run_burnup_cell_upper_bound_is_clipped_not_rejected(field, bound_key, adjust_code):
    """Path 2 upper-bound characterization: fistart/ti/u/tamb_c are
    silently CLIPPED (not rejected) when above the upper bound, recording
    the documented ``burnup_limit_adjust`` code - the simulation still
    runs to completion (``burnup_error == 0``)."""
    lo, hi, _label = _FIRE_BOUNDS[bound_key]
    result = bc._run_burnup_cell(_nominal_ckw(**{field: hi + 1.0}))
    assert result['burnup_error'] == 0
    assert str(adjust_code) in str(result['burnup_limit_adjust'])


def test_three_paths_disagree_on_the_same_out_of_range_fistart_value(tmp_path):
    """The single most concrete demonstration of Gotcha #23: the SAME
    just-below-lower-bound ``fistart`` value is handled three
    incompatible ways by the three paths - raised (path 1), rejected as
    a numeric error code with no exception (path 2), and silently
    clipped with no signal at all (path 3). No unified contract is
    asserted here; the divergence itself is the characterized fact."""
    lo, _hi, _label = _FIRE_BOUNDS['fistart']
    bad_value = lo - 1.0

    with pytest.raises(BurnupValidationError):
        _check_fire(**_nominal_check_fire_kwargs(fistart=bad_value))

    cell_result = bc._run_burnup_cell(_nominal_ckw(intensity_kw=bad_value))
    assert cell_result['burnup_error'] == 10

    out_path = tmp_path / 'divergent.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path), intensity=bad_value)
    written = _parse_brn(out_path.read_text())
    assert written['INTENSITY'] == pytest.approx(lo)


@pytest.fixture
def tmp_path(request):
    """
    Override pytest's built-in ``tmp_path`` fixture for every test in
    this module: a repository-local, collision-safe directory under
    ``tests/cpp_parity_live/_scratch.py``'s scratch root, never the
    system/user temporary directory, per the Phase 7 correction pass's
    explicit filesystem boundary (2026-09-06). Returns a real
    ``pathlib.Path`` so existing test bodies (``tmp_path / 'x.brn'``,
    ``.read_text()``) work completely unchanged. Removed on exit
    regardless of test outcome.

    :param request: The requesting test node (used only to namespace the
        directory name for readability under the shared scratch root).
    :return: Yields a ``pathlib.Path`` to the created directory.
    """
    with scratch_tempdir("fire_environment_bounds", prefix=request.node.name) as path:
        yield Path(path)
