#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Named regression tests for four historical bugs found and fixed in
pyfofem's consumption equations, split out of ``test_equations_golden.py``
during the Phase 1 directory restructure (the golden-CSV-driven parametrized
coverage stayed in ``tests/unit/test_consumption_golden.py``):

- Fix A: ``consm_duff`` Eq 3/7 (nfdth) must use ``dw1000_moist``, not
  ``duff_moist``.
- Fix B: ``consm_herb`` GrassGroup Eq 221 (90%) applies only in Spring.
- Fix C: ``consm_duff`` pile burning (Eq 17) must return ``pdc=10%``, not
  90%.
- Fix D: ``consm_duff`` low-moisture floor — ``duff_moist <= 10`` forces
  ``pdc=100%``.
"""

import pytest

from pyfofem import consm_duff, consm_herb

pytestmark = pytest.mark.regression


class TestFixA_Eq3UsesCorrectMoisture:
    """Fix A: Eq 3 (nfdth) must use dw1000_moist, not duff_moist."""

    def test_eq3_uses_dw1000_moist(self):
        """Result must differ when dw1000_moist ≠ duff_moist."""
        result_correct = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=20.0, units='Imperial',
        )
        result_wrong = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=None,  # falls back to duff_moist=80 → 114.7-4.2*80 → clamped 0
            units='Imperial',
        )
        # Correct: 114.7 - 4.2*20 = 30.7
        assert abs(result_correct['pdc'] - 30.7) < 0.01
        # Fallback: 114.7 - 4.2*80 = -221.3 → clamped to 0 by np.clip
        assert result_wrong['pdc'] == pytest.approx(0.0, abs=0.01)

    def test_eq7_uses_dw1000_moist(self):
        """nfdth depth (``ddc``) must use dw1000_moist, via its effect on
        ``pdc`` (Eq 3).

        CORRECTED 2026-09-21 (F-23 fix pass): ``ddc`` is no longer computed
        via the abandoned Eq 7 depth-reduction regression -- C++ ``DUF_Mngr``
        itself never uses per-region depth equations for its returned
        output (``fof_duf.cpp`` Note-5, unconditional override at
        ``:395``: ``f_Red = f_DufDep * (f_Per / 100.0)`` for every region).
        ``ddc`` is now ALWAYS percent-derived from the final ``pdc``, so
        dw1000_moist's effect on depth is now indirect, through Eq 3's own
        ``pdc``, not a separate Eq 7 formula.
        """
        result = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=20.0, d_pre=3.0, units='Imperial',
        )
        # pdc (Eq 3, unaffected by this pass): 114.7 - 4.2*20 = 30.7
        # ddc = d_pre * (pdc/100) = 3.0 * 0.307 = 0.921 (fof_duf.cpp:395)
        expected_pdc = 114.7 - 4.2 * 20.0
        expected_ddc = 3.0 * (expected_pdc / 100.0)
        assert abs(result['pdc'] - expected_pdc) < 0.001
        assert abs(result['ddc'] - expected_ddc) < 0.001


class TestFixB_GrassHerbSeason:
    """Fix B: GrassGroup Eq 221 (90%) applies only in Spring.

    CORRECTED 2026-09-18 (F-70 comprehensive-suite reconciliation pass,
    see gate0/04-findings.md F-35): this class's original premise --
    that Eq 221 consumes 10% of the herb load in Spring -- was itself
    backwards. The pinned C++ ``Herb_Eq221`` (``fof_hsf.cpp:349``)
    computes ``f_Herb * 0.9``: 90% consumed, not 10%. Production
    ``consm_herb`` already matches this (``pre_hl * 0.9``); only this
    test's expected values were stale.
    """

    def test_grass_spring_is_90pct(self):
        """GrassGroup herb consumption in Spring is 90% (Eq 221).

        :return: None. Raises via ``assert`` on mismatch.
        """
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Spring', units='Imperial')
        assert abs(hlc - 1.8) < 0.001  # 2.0 * 0.9 = 1.8

    def test_grass_summer_is_100pct(self):
        """GrassGroup herb consumption in Summer is 100%.

        :return: None. Raises via ``assert`` on mismatch.
        """
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Summer', units='Imperial')
        assert abs(hlc - 2.0) < 0.001  # 100 % consumed

    def test_grass_fall_is_100pct(self):
        """GrassGroup herb consumption in Fall is 100%.

        :return: None. Raises via ``assert`` on mismatch.
        """
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Fall', units='Imperial')
        assert abs(hlc - 2.0) < 0.001

    def test_grass_winter_is_100pct(self):
        """GrassGroup herb consumption in Winter is 100%.

        :return: None. Raises via ``assert`` on mismatch.
        """
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Winter', units='Imperial')
        assert abs(hlc - 2.0) < 0.001

    def test_grass_no_season_is_100pct(self):
        """No season provided → defaults to non-Spring behaviour (100%)."""
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         units='Imperial')
        assert abs(hlc - 2.0) < 0.001


class TestFixC_PileBurning:
    """Fix C: pile burning (Eq 17) must return pdc=10%, not 90%."""

    def test_pile_pdc_is_10_percent(self):
        result = consm_duff(pre_dl=10.0, duff_moist=50.0,
                            pile=True, units='Imperial')
        assert abs(result['pdc'] - 10.0) < 0.01, (
            f"Pile burning pdc={result['pdc']:.2f}, expected 10.0"
        )

    def test_pile_consumed_amount(self):
        """Consumed amount = pre_dl * 10% = 1.0 T/ac."""
        result = consm_duff(pre_dl=10.0, duff_moist=50.0,
                            pile=True, units='Imperial')
        pdc = result['pdc']
        consumed = 10.0 * pdc / 100.0
        assert abs(consumed - 1.0) < 0.01, (
            f'Pile consumed={consumed:.3f} T/ac, expected 1.0'
        )


class TestFixD_LowMoistureFloor:
    """Fix D: duff_moist ≤ 10 forces pdc=100%."""

    def test_floor_at_exactly_10(self):
        result = consm_duff(pre_dl=10.0, duff_moist=10.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        assert abs(result['pdc'] - 100.0) < 0.01

    def test_floor_below_10(self):
        result = consm_duff(pre_dl=10.0, duff_moist=5.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        assert abs(result['pdc'] - 100.0) < 0.01

    def test_no_floor_above_10(self):
        """pdc should NOT be forced to 100% when duff_moist > 10."""
        result = consm_duff(pre_dl=10.0, duff_moist=11.0,
                            reg='InteriorWest', duff_moist_cat='edm',
                            units='Imperial')
        # Eq 2: 83.7 - 0.426*11 = 79.014
        expected = 83.7 - 0.426 * 11.0
        assert abs(result['pdc'] - expected) < 0.01
        assert result['pdc'] < 100.0
