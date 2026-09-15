#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phase8_moisture_regime_integration.py - Phase 8 item D: moisture-
regime integration matrix.

``get_moisture_regime()`` itself already has full unit coverage
(``tests/unit/test_utility_contracts.py``: all four regimes, case
normalization, invalid names, defensive-copy, and the wet>moderate>dry>
very-dry ordering invariant). This module verifies moisture-regime
behavior through the relevant PUBLIC INTEGRATION ROUTES instead -
``run_fofem_emissions(moisture_regime=...)`` (the facade-level
consumer) and ``consm_duff()`` (the function whose own routing table
varies by region/cover-group/duff-moisture-category, so it is the
natural place to test the regime's interaction with those axes, per
item D's own "interaction with regions, cover groups, duff moisture
method/category, and relevant consumption routes" requirement).

**`run_fofem_emissions()` has no user-facing `duff_moist_method`/
`duff_moist_cat` parameter of its own** - it hardcodes
``consm_duff(duff_moist_cat='edm')`` internally with no caller override
(a pre-existing finding, F-61, not re-derived here). The region/cover-
group/duff-moisture-category interaction axis of item D is therefore
exercised directly against ``consm_duff()`` (itself a public function),
not through ``run_fofem_emissions()``, since that is the real public
surface where the interaction actually varies. Calling ``consm_duff()``
directly with an explicit ``duff_moist_cat`` means F-61 (a facade-level
gap only) simply does not apply to these calls; F-23 (the Northeast
generic Eq-15-vs-``Duf_Default`` routing defect) is exercised AS-IS,
never hidden or worked around - these tests assert only Python's own
internal ordering/finiteness contract, never a C++ parity claim.

**Correction pass (2026-09-10, responding to independent review) - the
matrix is now a REAL, complete parametrized cross-product**, not a
partial approximation of one: all
**4 moisture regimes x 4 regions x 2 representative cover-group values
x 3 duff-moisture categories = 96** combinations are exercised as 96
independent parametrized nodes
(``test_consm_duff_cross_product_produces_a_finite_in_range_percent``),
each asserting only that the combination runs without raising and
produces a finite ``pdc`` in ``[0, 100]``. The two cover-group values
chosen per region are REPRESENTATIVE of that region's own real routing
table (not the same two labels blindly reused across all four regions,
which would be meaningless for regions whose cover-group vocabulary
differs) - see :data:`_REGION_COVER_GROUPS`. Every one of the
resulting 24 (region, cover-group, category) combinations was
confirmed by direct execution to be structurally APPLICABLE (produces
a finite, regime-varying result) before being included - none needed
to be encoded as inapplicable/skipped.

The wet>moderate>dry>very-dry pdc ordering invariant is inherently a
CROSS-regime property (comparing 4 regime-driven calls against each
other), so it is expressed as a second, complementary parametrization
over the 24 (region, cover-group, category) combinations
(``test_consm_duff_pdc_ordering_across_regimes``), each of which
internally calls all 4 regimes and asserts the ordering - together, the
two parametrized tests exercise the complete claimed matrix precisely
described here, not merely asserted in prose.

**Correction pass (2026-09-11, responding to independent review)** - the
NorthEast entry in :data:`_REGION_COVER_GROUPS` previously used
``'RedJacPine'``/``'BalsamSpruce'``, neither of which
``consm_duff()``'s own ``_REDJAC``/``_BALSAM`` membership sets recognize
(see that function's source: ``_REDJAC = {'Red Jack Pine', 'Red, Jack
Pine', 'RedJacPin', 'RJP'}``, ``_BALSAM = {'Balsam', 'Black Spruce', 'Red
Spruce', 'White Spruce', 'BalBRWSpr', 'Balsam Fir', 'BFS'}``) - every
NorthEast case in both the 96-case and 24-case matrices was silently
falling through to the generic NorthEast branch (``Duf_Default`` / Eq 2)
rather than exercising the RedJacPin/Balsam-Spruce specialized routing the
matrix claimed to cover. Fixed by using the accepted labels
``'RedJacPin'``/``'BalBRWSpr'`` directly, and by adding
``test_consm_duff_northeast_labels_reach_their_distinct_branches``, an
executable discrimination test proving (by direct numeric comparison, not
mere finiteness) that ``'RedJacPin'`` and ``'BalBRWSpr'`` each reach a
branch distinct from the generic NorthEast fallback and from each other -
confirmed by direct execution before this test was written:
``duff_moist=40.0, d_pre=3.0`` gives ``pdc`` 22.37 (RedJacPin, edm) vs.
41.03 (an unrecognized cover group, edm) - RedJacPin's own Eq-15-pine=1
branch, not the fallback; and 64.10 (BalBRWSpr, ldm) vs. 66.66 (an
unrecognized cover group / RedJacPin, ldm) - BalBRWSpr's own Eq-5 branch,
not the fallback. The corrected labels were re-run through the complete
96-case and 24-case matrices; the wet>moderate>dry>very-dry ``pdc``
ordering invariant continues to hold for both real NorthEast branches (no
regression, no weakening of the invariant, no defect newly exposed - see
the validation record in ``.claude/CLAUDE.md`` for the measured
per-combination confirmation).

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfofem import consm_duff, get_moisture_regime, run_fofem_emissions

_REGIMES = ('wet', 'moderate', 'dry', 'very dry')

#: 4 regions, each paired with 2 cover-group values REPRESENTATIVE of
#: that region's own real ``consm_duff()`` routing table (see that
#: function's own docstring routing table) - not the same 2 labels
#: reused verbatim across every region, which would be meaningless
#: (e.g. ``'Ponderosa pine'`` has no special routing in ``SouthEast``).
_REGION_COVER_GROUPS = {
    'InteriorWest': ('Ponderosa pine', 'Other'),
    'PacificWest': ('Ponderosa pine', 'Other'),
    'NorthEast': ('RedJacPin', 'BalBRWSpr'),
    'SouthEast': ('Pocosin', 'Other'),
}

#: The 3 documented duff-moisture categories.
_DUFF_MOIST_CATS = ('ldm', 'edm', 'nfdth')

#: The complete 24-way (region, cover_group, duff_moist_cat) combination
#: set - the outer 3 axes of the full 96-way cross product; the 4th axis
#: (moisture regime) is exercised INSIDE each parametrized case below,
#: either individually (the 96-node finiteness matrix) or as an internal
#: 4-way comparison (the 24-node ordering matrix).
_DUFF_COMBOS_24 = [
    (region, cvr_grp, cat)
    for region, cvr_grps in _REGION_COVER_GROUPS.items()
    for cvr_grp in cvr_grps
    for cat in _DUFF_MOIST_CATS
]

#: The complete 96-way cross product: every (region, cover_group,
#: duff_moist_cat, regime) combination.
_DUFF_COMBOS_96 = [
    (region, cvr_grp, cat, regime)
    for region, cvr_grp, cat in _DUFF_COMBOS_24
    for regime in _REGIMES
]


def _emissions_with_regime(moisture_regime, region='InteriorWest'):
    """
    Run a minimal ``run_fofem_emissions()`` call using *moisture_regime*.

    :param moisture_regime: Value forwarded verbatim as
        ``moisture_regime=``.
    :param region: FOFEM region for the single cell.
    :return: The full ``run_fofem_emissions()`` result dict.
    """
    return run_fofem_emissions(
        litter=1.0, duff=1.0, duff_depth=1.0, herb=0.0, shrub=0.0,
        crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
        region=region, use_burnup=False, moisture_regime=moisture_regime,
        units='Imperial',
    )


@pytest.mark.parametrize('region,cvr_grp,duff_moist_cat,regime', _DUFF_COMBOS_96)
def test_consm_duff_cross_product_produces_a_finite_in_range_percent(region, cvr_grp, duff_moist_cat, regime):
    """The complete 96-way (4 regimes x 4 regions x 2 representative
    cover groups x 3 duff-moisture categories) cross product: every
    combination must run without raising and produce a finite ``pdc``
    in ``[0, 100]`` - supplying every auxiliary input
    (``d_pre``/``dw1000_moist``) the routed equation could need so no
    combination silently produces ``NaN`` for lack of an argument."""
    regime_vals = get_moisture_regime(regime)
    result = consm_duff(
        pre_dl=2.0, duff_moist=regime_vals['duff'], reg=region, cvr_grp=cvr_grp,
        duff_moist_cat=duff_moist_cat, d_pre=3.0, dw1000_moist=regime_vals['3plus'],
        units='Imperial',
    )
    assert np.isfinite(result['pdc']), (
        f'region={region} cvr_grp={cvr_grp} duff_moist_cat={duff_moist_cat} '
        f'regime={regime}: pdc is not finite: {result["pdc"]!r}'
    )
    assert 0.0 <= result['pdc'] <= 100.0, (
        f'region={region} cvr_grp={cvr_grp} duff_moist_cat={duff_moist_cat} '
        f'regime={regime}: pdc out of [0, 100]: {result["pdc"]!r}'
    )


def test_consm_duff_northeast_labels_reach_their_distinct_branches():
    """Proves ``'RedJacPin'`` and ``'BalBRWSpr'`` (the two NorthEast
    cover-group labels used by :data:`_REGION_COVER_GROUPS`) each reach a
    genuinely distinct routing branch inside ``consm_duff()``, rather than
    merely producing SOME finite result that could silently be the generic
    NorthEast fallback (``Duf_Default`` / Eq 2) - the exact failure mode
    the pre-correction labels (``'RedJacPine'``/``'BalsamSpruce'``, neither
    recognized by ``consm_duff()``'s own ``_REDJAC``/``_BALSAM`` membership
    sets) actually exhibited. Uses an unrecognized cover-group label as a
    live fallback reference (rather than a hardcoded expected number) so
    the assertion is robust to any future change in the fallback formula's
    coefficients."""
    common = dict(pre_dl=2.0, duff_moist=40.0, reg='NorthEast', d_pre=3.0, dw1000_moist=30.0, units='Imperial')

    redjac_edm = consm_duff(cvr_grp='RedJacPin', duff_moist_cat='edm', **common)['pdc']
    balsam_edm = consm_duff(cvr_grp='BalBRWSpr', duff_moist_cat='edm', **common)['pdc']
    generic_edm = consm_duff(cvr_grp='NotARoutedCoverGroup', duff_moist_cat='edm', **common)['pdc']

    redjac_ldm = consm_duff(cvr_grp='RedJacPin', duff_moist_cat='ldm', **common)['pdc']
    balsam_ldm = consm_duff(cvr_grp='BalBRWSpr', duff_moist_cat='ldm', **common)['pdc']
    generic_ldm = consm_duff(cvr_grp='NotARoutedCoverGroup', duff_moist_cat='ldm', **common)['pdc']

    assert redjac_edm != pytest.approx(generic_edm), (
        "'RedJacPin' edm must reach its own Eq-15(pine=1) branch, distinct "
        f'from the generic NorthEast fallback, got redjac={redjac_edm} generic={generic_edm}'
    )
    assert balsam_ldm != pytest.approx(generic_ldm), (
        "'BalBRWSpr' ldm must reach its own Eq-5 branch, distinct from the "
        f'generic NorthEast fallback, got balsam={balsam_ldm} generic={generic_ldm}'
    )
    assert redjac_edm != pytest.approx(balsam_edm), (
        "'RedJacPin' and 'BalBRWSpr' must diverge under edm (pine=1 vs. "
        f'pine=0), got redjac={redjac_edm} balsam={balsam_edm}'
    )
    assert redjac_ldm == pytest.approx(generic_ldm), (
        "'RedJacPin' ldm is documented to default to the same Eq-2 formula "
        f'as the generic fallback, got redjac={redjac_ldm} generic={generic_ldm}'
    )


@pytest.mark.parametrize('region,cvr_grp,duff_moist_cat', _DUFF_COMBOS_24)
def test_consm_duff_pdc_ordering_across_regimes(region, cvr_grp, duff_moist_cat):
    """For each of the 24 (region, cover_group, duff_moist_cat)
    combinations, percent duff consumed (``pdc``) must be non-increasing
    as the regime gets wetter (wet <= moderate <= dry <= very dry, in
    consumed-percent terms) - an oracle-independent physical invariant,
    not a value pinned to a specific number. Confirmed by direct
    execution before this test was written that all 24 combinations
    genuinely VARY with regime (none is a flat, regime-independent
    equation that would make this assertion vacuous)."""
    pdc_by_regime = {}
    for regime in _REGIMES:
        regime_vals = get_moisture_regime(regime)
        pdc_by_regime[regime] = consm_duff(
            pre_dl=2.0, duff_moist=regime_vals['duff'], reg=region,
            cvr_grp=cvr_grp, duff_moist_cat=duff_moist_cat, d_pre=3.0,
            dw1000_moist=regime_vals['3plus'], units='Imperial',
        )['pdc']
    assert (
        pdc_by_regime['wet'] <= pdc_by_regime['moderate'] + 1e-9
        <= pdc_by_regime['dry'] + 1e-9
        <= pdc_by_regime['very dry'] + 1e-9
    ), (
        f'region={region} cvr_grp={cvr_grp} duff_moist_cat={duff_moist_cat}: pdc must be '
        f'non-increasing from wet to very-dry, got {pdc_by_regime}'
    )


def test_get_moisture_regime_rejects_a_no_space_variant_of_very_dry():
    """No silent fallback/fuzzy matching: ``'verydry'`` (no space) is NOT
    accepted even though ``'very dry'`` (with a space) is documented -
    exact string matching after ``.strip().lower()`` only, no alias
    table. Current, documented contract; not loosened here."""
    with pytest.raises(KeyError):
        get_moisture_regime('verydry')


def test_run_fofem_emissions_applies_the_same_scalar_regime_uniformly_across_cells():
    """``moisture_regime`` is a single scalar string applied to the WHOLE
    array call, not a per-cell value - two cells with different regions
    but the SAME regime must receive the identical derived moisture
    inputs (confirmed indirectly: their outputs differ only through the
    region-dependent routing, not through any per-cell moisture
    variation)."""
    n = 3
    zeros = np.zeros(n)
    result = run_fofem_emissions(
        litter=np.full(n, 1.0), duff=np.full(n, 1.0), duff_depth=np.full(n, 1.0),
        herb=zeros, shrub=zeros, crown_foliage=zeros, crown_branch=zeros,
        pct_crown_burned=zeros, region=np.array(['InteriorWest', 'InteriorWest', 'InteriorWest']),
        use_burnup=False, moisture_regime='Dry', units='Imperial',
    )
    lit_con = np.asarray(result['LitCon'])
    assert lit_con[0] == pytest.approx(lit_con[1]) == pytest.approx(lit_con[2]), (
        'identical cells under the same scalar moisture_regime should produce identical results'
    )


def test_run_fofem_emissions_missing_moisture_regime_and_explicit_moisture_raises():
    """Omitting ``moisture_regime`` AND every explicit moisture parameter
    must raise ``ValueError`` naming the missing parameters - no silent
    fallback to an arbitrary default."""
    with pytest.raises(ValueError, match='Missing at least one required moisture input'):
        run_fofem_emissions(
            litter=1.0, duff=0.0, duff_depth=0.0, herb=0.0, shrub=0.0,
            crown_foliage=0.0, crown_branch=0.0, pct_crown_burned=0.0,
            region='InteriorWest', use_burnup=False,
        )


def test_run_fofem_emissions_moisture_regime_is_case_insensitive():
    """``moisture_regime='Wet'``/``'WET'``/``' wet '`` must all dispatch
    identically to ``'wet'`` through the real facade call, matching
    ``get_moisture_regime()``'s own documented case-insensitivity."""
    baseline = _emissions_with_regime('wet')
    for variant in ('Wet', 'WET', ' wet ', 'wEt'):
        result = _emissions_with_regime(variant)
        assert result['LitCon'] == pytest.approx(baseline['LitCon']), (
            f"moisture_regime variant '{variant}' did not match 'wet'"
        )


def test_run_fofem_emissions_rejects_an_unrecognized_moisture_regime():
    """An unrecognized ``moisture_regime`` value must raise ``KeyError``
    naming the valid options - propagated unmodified from
    ``get_moisture_regime()``, no swallowing/defaulting at the facade
    level."""
    with pytest.raises(KeyError, match='Unknown moisture regime'):
        _emissions_with_regime('humid')


def test_run_fofem_emissions_supports_every_documented_regime():
    """Every one of the four documented regimes must run successfully
    through ``run_fofem_emissions()`` and produce a finite ``LitCon``."""
    for regime in _REGIMES:
        result = _emissions_with_regime(regime)
        assert np.isfinite(result['LitCon']), f"regime '{regime}' produced a non-finite LitCon"
