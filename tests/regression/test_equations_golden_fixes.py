#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Named regression tests for four historical bugs found and fixed in
pyfofem's consumption equations, split out of ``test_equations_golden.py``
during the Phase 1 directory restructure (the golden-CSV-driven parametrized
coverage stayed in ``tests/unit/test_consumption_golden.py``):

- Fix A: ``consm_duff`` Eq 3/7 (nfdth) must use ``dw1000_moist``, not
  ``duff_moist``.
- Fix B: ``consm_herb`` GrassGroup Eq 221 (10%) applies only in Spring.
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
        """Eq 7 depth (nfdth) must use dw1000_moist."""
        result = consm_duff(
            pre_dl=10.0, duff_moist=80.0,
            reg='InteriorWest', duff_moist_cat='nfdth',
            dw1000_moist=20.0, d_pre=3.0, units='Imperial',
        )
        # Eq 7: 1.773 - 0.1051*20 + 0.399*3 = 1.773 - 2.102 + 1.197 = 0.868
        expected = 1.773 - 0.1051 * 20.0 + 0.399 * 3.0
        assert abs(result['ddc'] - expected) < 0.001


class TestFixB_GrassHerbSeason:
    """Fix B: GrassGroup Eq 221 (10%) applies only in Spring."""

    def test_grass_spring_is_10pct(self):
        """GrassGroup herb consumption in Spring is 10% (Eq 221).

        :return: None. Raises via ``assert`` on mismatch.
        """
        hlc = consm_herb('InteriorWest', 'GrassGroup', 2.0, 2.0,
                         season='Spring', units='Imperial')
        assert abs(hlc - 0.2) < 0.001  # 2.0 * 0.1 = 0.2

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
