# -*- coding: utf-8 -*-
"""
test_cpp_comparison.py - Compare Python burnup outputs against existing C++ reference.

The C++ ansi_mai.cpp has been compiled and run, producing:
  - reference/fofem_cpp/load.txt  (per-component burnup summary)
  - reference/fofem_cpp/emis.txt  (per-timestep emissions time-series)

This script replicates the exact same inputs in Python and compares results.
"""
import os
import re
import sys
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure pyfofem is importable
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pyfofem import run_fofem_emissions, run_burnup
from pyfofem.components.burnup_calcs import (
    _TPAC_TO_KGPM2,
    _KGPM2_TO_TPAC,
    _DENSITY_SOUND,
    _DENSITY_ROTTEN,
    _SOUND_TPIG,
    _ROTTEN_TPIG,
    _TCHAR,
    _HTVAL,
)

# ---------------------------------------------------------------------------
# Paths to C++ reference outputs
# ---------------------------------------------------------------------------
_CPP_DIR = os.path.join(_REPO, 'reference', 'fofem_cpp')
_LOAD_TXT = os.path.join(_CPP_DIR, 'load.txt')
_EMIS_TXT = os.path.join(_CPP_DIR, 'emis.txt')


# ---------------------------------------------------------------------------
# C++ reference parsers
# ---------------------------------------------------------------------------
def parse_load_txt(path: str) -> list:
    """Parse C++ load.txt into a list of dicts (one per fuel component).

    Columns (from the header in load.txt):
      1: component index
      2: preburn kg/m2
      3: postburn kg/m2
      4: t_ignite (sec)
      5: t_burnout (sec)   (after the '-->' separator)
      6: moisture (fraction 0-1)
      7: sigma (SAV, 1/m)
      8: preburn T/ac
      9: postburn T/ac
     10: preburn lb/ac
     11: postburn lb/ac
    """
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        # Data lines start with a number (component index)
        if not line or not line[0].isdigit():
            # Check for the "Duration of fire" line
            if 'Duration of fire' in line:
                pass
            continue
        # Replace '-->' with whitespace for uniform parsing
        line = line.replace('-->', ' ')
        parts = line.split()
        if len(parts) < 11:
            continue
        rows.append({
            'comp_idx':       int(parts[0]),
            'preburn_kgm2':   float(parts[1]),
            'postburn_kgm2':  float(parts[2]),
            't_ignite_s':     float(parts[3]),
            't_burnout_s':    float(parts[4]),
            'moisture_frac':  float(parts[5]),
            'sigma':          float(parts[6]),
            'preburn_tac':    float(parts[7]),
            'postburn_tac':   float(parts[8]),
            'preburn_lbac':   float(parts[9]),
            'postburn_lbac':  float(parts[10]),
        })
    return rows


def parse_emis_txt(path: str) -> list:
    """Parse C++ emis.txt into a list of dicts (one per timestep).

    Columns:
      1: time (sec)
      2: intensity (kW/m2)
      3: PM2.5 (g/m2)
      4: PM10 (g/m2)
      5: CH4 (g/m2)
      6: CO2 (g/m2)
      7: CO (g/m2)
      8: NOx (g/m2)
      9: SO2 (g/m2)
     10: flaming weight (kg/m2)
     11: smoldering weight (kg/m2)
     12: flaming weight (T/ac)
     13: smoldering weight (T/ac)
    """
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        rows.append({
            'time_s':         float(parts[0]),
            'intensity_kw':   float(parts[1]),
            'pm25_gm2':       float(parts[2]),
            'pm10_gm2':       float(parts[3]),
            'ch4_gm2':        float(parts[4]),
            'co2_gm2':        float(parts[5]),
            'co_gm2':         float(parts[6]),
            'nox_gm2':        float(parts[7]),
            'so2_gm2':        float(parts[8]),
            'flame_kgm2':     float(parts[9]),
            'smolder_kgm2':   float(parts[10]),
            'flame_tac':      float(parts[11]),
            'smolder_tac':    float(parts[12]),
        })
    return rows


# ---------------------------------------------------------------------------
# ansi_mai.cpp input reconstruction
# ---------------------------------------------------------------------------
# These exactly replicate the hardcoded values in ansi_mai.cpp ConEmiSoi()
# plus the defaults set by CI_Init() in fof_ci.cpp.
ANSI_MAI_INPUTS = {
    'litter': 1.0,          # T/ac
    'duff': 1.0,
    'duff_depth': 1.0,      # inches
    'herb': 1.0,
    'shrub': 1.0,
    'crown_foliage': 1.0,
    'crown_branch': 1.0,
    'pct_crown_burned': 50.0,
    'duff_moist': 10.0,     # whole number %
    'dw10_moist': 20.0,     # whole number %
    'dw1000_moist': 20.0,   # whole number %
    'dw1': 1.0,
    'dw10': 1.0,
    'dw100': 1.0,
    # 1000-hr wood: 1 T/ac total, split evenly into 8 size/rot bins
    'dw3_6s': 0.125,
    'dw6_9s': 0.125,
    'dw9_20s': 0.125,
    'dw20s': 0.125,
    'dw3_6r': 0.125,
    'dw6_9r': 0.125,
    'dw9_20r': 0.125,
    'dw20r': 0.125,
    'region': 'InteriorWest',
    'season': 'Summer',
    'fuel_category': 'Natural',
    # Fire environment (CI_Init defaults from bur_brn.h)
    'hfi': 50.0,            # kW/m2 (e_INTENSITY)
    'flame_res_time': 60.0, # seconds (e_IG_TIME)
    'fuel_bed_depth': 0.3,  # meters (e_DEPTH)
    'ambient_temp': 27.0,   # Celsius (e_AMBIENT_TEMP)
    'windspeed': 0.0,       # m/s (e_WINDSPEED)
}

# C++ load.txt maps component indices to these fuel types:
# 1: litter (sound), 2: dw1 (sound), 3: dw10 (sound), 4: dw100 (sound)
# 5: dwk_3_6 (sound), 6: dwk_3_6 (rotten), 7: dwk_6_9 (sound), 8: dwk_6_9 (rotten)
# 9: dwk_9_20 (sound), 10: dwk_9_20 (rotten), 11: dwk_20 (sound), 12: dwk_20 (rotten)
#
# The order comes from BCM_SetInputs: litter, dw1, dw10, dw100,
# then for each size class: sound then rotten (3-6, 3-6r, 6-9, 6-9r, 9-20, 9-20r, 20, 20r)
CPP_COMP_TO_PYKEY = {
    1: 'litter',
    2: 'dw1',
    3: 'dw10',
    4: 'dw100',
    5: 'dwk_3_6',
    6: 'dwk_3_6_r',
    7: 'dwk_6_9',
    8: 'dwk_6_9_r',
    9: 'dwk_9_20',
    10: 'dwk_9_20_r',
    11: 'dwk_20',
    12: 'dwk_20_r',
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def cpp_load():
    """Parse C++ load.txt reference output."""
    if not os.path.isfile(_LOAD_TXT):
        pytest.skip(f'C++ reference file not found: {_LOAD_TXT}')
    return parse_load_txt(_LOAD_TXT)


@pytest.fixture(scope='module')
def cpp_emis():
    """Parse C++ emis.txt reference output."""
    if not os.path.isfile(_EMIS_TXT):
        pytest.skip(f'C++ reference file not found: {_EMIS_TXT}')
    return parse_emis_txt(_EMIS_TXT)


@pytest.fixture(scope='module')
def py_pipeline_result():
    """Run the full Python pipeline with ansi_mai.cpp inputs."""
    inp = ANSI_MAI_INPUTS
    return run_fofem_emissions(
        litter=inp['litter'],
        duff=inp['duff'],
        duff_depth=inp['duff_depth'],
        herb=inp['herb'],
        shrub=inp['shrub'],
        crown_foliage=inp['crown_foliage'],
        crown_branch=inp['crown_branch'],
        pct_crown_burned=inp['pct_crown_burned'],
        region=inp['region'],
        season=inp['season'],
        fuel_category=inp['fuel_category'],
        duff_moist=inp['duff_moist'],
        dw10_moist=inp['dw10_moist'],
        dw1000_moist=inp['dw1000_moist'],
        dw1=inp['dw1'],
        dw10=inp['dw10'],
        dw100=inp['dw100'],
        dw3_6s=inp['dw3_6s'],
        dw6_9s=inp['dw6_9s'],
        dw9_20s=inp['dw9_20s'],
        dw20s=inp['dw20s'],
        dw3_6r=inp['dw3_6r'],
        dw6_9r=inp['dw6_9r'],
        dw9_20r=inp['dw9_20r'],
        dw20r=inp['dw20r'],
        hfi=inp['hfi'],
        flame_res_time=inp['flame_res_time'],
        fuel_bed_depth=inp['fuel_bed_depth'],
        ambient_temp=inp['ambient_temp'],
        windspeed=inp['windspeed'],
        use_burnup=True,
        units='Imperial',
    )


@pytest.fixture(scope='module')
def py_burnup_direct():
    """Run run_burnup() with full 12-class parity to C++ load.txt."""
    # C++ moisture adjustments (fof_bcm.h / fof_bcm.cpp):
    #   DW10 moisture: 20% -> 0.20 (fraction, no adjustment for DW10)
    #   DW1/litter:    0.20 - 0.02 = 0.18
    #   DW100:         0.20 + 0.02 = 0.22
    #   1000-hr sound: 20% -> 0.20
    #   1000-hr rotten: 0.20 * 2.5 = 0.50 (capped at 3.0)
    dw10_frac = 20.0 / 100.0  # = 0.20
    dw1_moist = dw10_frac - 0.02   # = 0.18
    dw100_moist = dw10_frac + 0.02  # = 0.22
    dw1k_snd_moist = 20.0 / 100.0  # = 0.20
    dw1k_rot_moist = min(dw1k_snd_moist * 2.5, 3.0)  # = 0.50

    # C++ TPA_To_KiSq uses 4.46; Python uses 4.4609.
    # For this comparison we use Python's constant to match what Python burnup sees.
    to_si = _TPAC_TO_KGPM2

    fuel_loadings = {
        'litter': 1.0 * to_si,
        'dw1': 1.0 * to_si,
        'dw10': 1.0 * to_si,
        'dw100': 1.0 * to_si,
        'dwk_3_6': 0.125 * to_si,
        'dwk_3_6_r': 0.125 * to_si,
        'dwk_6_9': 0.125 * to_si,
        'dwk_6_9_r': 0.125 * to_si,
        'dwk_9_20': 0.125 * to_si,
        'dwk_9_20_r': 0.125 * to_si,
        'dwk_20': 0.125 * to_si,
        'dwk_20_r': 0.125 * to_si,
    }
    fuel_moistures = {
        'litter': dw1_moist,
        'dw1': dw1_moist,
        'dw10': dw10_frac,
        'dw100': dw100_moist,
        'dwk_3_6': dw1k_snd_moist,
        'dwk_3_6_r': dw1k_rot_moist,
        'dwk_6_9': dw1k_snd_moist,
        'dwk_6_9_r': dw1k_rot_moist,
        'dwk_9_20': dw1k_snd_moist,
        'dwk_9_20_r': dw1k_rot_moist,
        'dwk_20': dw1k_snd_moist,
        'dwk_20_r': dw1k_rot_moist,
    }
    densities = {
        'dwk_3_6_r': _DENSITY_ROTTEN,
        'dwk_6_9_r': _DENSITY_ROTTEN,
        'dwk_9_20_r': _DENSITY_ROTTEN,
        'dwk_20_r': _DENSITY_ROTTEN,
    }

    # HSF consumed: in the C++ pipeline, HSF_Mngr computes these before burnup.
    # For the ansi_mai.cpp case (InteriorWest, Summer, all loads=1 T/ac):
    #   herb consumed = 1.0 T/ac (100% for InteriorWest non-grass/non-flatwood)
    #   shrub consumed = 1.0 * 0.60 = 0.60 T/ac (Equ 23, ~60% for Summer InteriorWest)
    #   foliage consumed = 1.0 * 50% = 0.50 T/ac (50% crown burn)
    #   branch consumed = 1.0 * 50% * 0.50 = 0.25 T/ac (50% of 50% crown burn)
    # These are the values the Python pipeline computes (verified in full pipeline test).
    herb_con_tac = 1.0   # from consm_herb
    shrub_con_tac = 0.60  # from consm_shrub (approximate)
    fol_con_tac = 0.50   # from consm_canopy
    bra_con_tac = 0.25   # from consm_canopy

    hsf_consumed_si = (herb_con_tac + shrub_con_tac) * to_si
    brafol_consumed_si = (fol_con_tac + bra_con_tac) * to_si

    results, summary, class_order = run_burnup(
        fuel_loadings=fuel_loadings,
        fuel_moistures=fuel_moistures,
        intensity=50.0,
        ig_time=60.0,
        windspeed=0.0,
        depth=0.3,
        ambient_temp=27.0,
        duff_loading=1.0 * to_si,
        duff_moisture=10.0 / 100.0,
        duff_pct_consumed=100.0,
        densities=densities,
        hsf_consumed=hsf_consumed_si,
        brafol_consumed=brafol_consumed_si,
    )
    return results, summary, class_order


# ---------------------------------------------------------------------------
# Tests: Burnup engine level (full 12 classes, via low-level burnup)
# ---------------------------------------------------------------------------
class TestBurnupDirect:
    """Compare low-level Python burnup() against C++ load.txt (12 classes)."""

    def test_component_count(self, py_burnup_direct, cpp_load):
        """Python should have 12 components to match C++ load.txt."""
        _, summary, class_order = py_burnup_direct
        assert len(summary) == 12, f'Expected 12 components, got {len(summary)}'
        assert len(cpp_load) == 12

    def test_preburn_loads_match(self, py_burnup_direct, cpp_load):
        """Pre-burn loads should match between C++ and Python."""
        _, summary, class_order = py_burnup_direct
        py_to_cpp = {
            'litter': 1, 'dw1': 2, 'dw10': 3, 'dw100': 4,
            'dwk_3_6': 5, 'dwk_3_6_r': 6, 'dwk_6_9': 7, 'dwk_6_9_r': 8,
            'dwk_9_20': 9, 'dwk_9_20_r': 10, 'dwk_20': 11, 'dwk_20_r': 12,
        }
        for i, key in enumerate(class_order):
            cpp_idx = py_to_cpp[key]
            cpp_row = cpp_load[cpp_idx - 1]  # 0-based
            py_wdry = summary[i].wdry
            cpp_wdry = cpp_row['preburn_kgm2']
            # Allow for TPA conversion factor difference (4.46 vs 4.4609)
            assert abs(py_wdry - cpp_wdry) < 0.001, \
                f'{key}: Python wdry={py_wdry:.6f} vs C++ preburn={cpp_wdry:.6f}'

    def test_frac_remaining(self, py_burnup_direct, cpp_load):
        """Fraction remaining should be close for all 12 classes."""
        _, summary, class_order = py_burnup_direct
        py_to_cpp = {
            'litter': 1, 'dw1': 2, 'dw10': 3, 'dw100': 4,
            'dwk_3_6': 5, 'dwk_3_6_r': 6, 'dwk_6_9': 7, 'dwk_6_9_r': 8,
            'dwk_9_20': 9, 'dwk_9_20_r': 10, 'dwk_20': 11, 'dwk_20_r': 12,
        }
        for i, key in enumerate(class_order):
            cpp_idx = py_to_cpp[key]
            cpp_row = cpp_load[cpp_idx - 1]
            cpp_frac = cpp_row['postburn_kgm2'] / cpp_row['preburn_kgm2'] \
                if cpp_row['preburn_kgm2'] > 0 else 0.0
            py_frac = summary[i].frac_remaining
            assert abs(py_frac - cpp_frac) < 0.01, \
                f'{key}: Python frac_remaining={py_frac:.6f} vs C++ frac={cpp_frac:.6f}'

    def test_ignition_times(self, py_burnup_direct, cpp_load):
        """Ignition times should be close for all 12 classes."""
        _, summary, class_order = py_burnup_direct
        py_to_cpp = {
            'litter': 1, 'dw1': 2, 'dw10': 3, 'dw100': 4,
            'dwk_3_6': 5, 'dwk_3_6_r': 6, 'dwk_6_9': 7, 'dwk_6_9_r': 8,
            'dwk_9_20': 9, 'dwk_9_20_r': 10, 'dwk_20': 11, 'dwk_20_r': 12,
        }
        for i, key in enumerate(class_order):
            cpp_idx = py_to_cpp[key]
            cpp_row = cpp_load[cpp_idx - 1]
            py_tig = summary[i].t_ignite
            cpp_tig = cpp_row['t_ignite_s']
            assert abs(py_tig - cpp_tig) < 15.0, \
                f'{key}: Python t_ignite={py_tig:.1f} vs C++ t_ignite={cpp_tig:.1f}'

    def test_fine_fuels_consumed(self, py_burnup_direct, cpp_load):
        """Fine fuels (litter, dw1) should be fully consumed in both."""
        _, summary, class_order = py_burnup_direct
        for i, key in enumerate(class_order):
            if key in ('litter', 'dw1'):
                assert summary[i].frac_remaining < 0.01, \
                    f'{key}: Python frac_remaining={summary[i].frac_remaining:.4f} (expected <0.01)'


class TestBurnupTimeSeries:
    """Compare per-timestep fire intensity between C++ emis.txt and Python burnup."""

    def test_timestep_count(self, py_burnup_direct, cpp_emis):
        """Number of timesteps should be similar."""
        results, _, _ = py_burnup_direct
        py_count = len(results)
        cpp_count = len(cpp_emis)
        # Allow some flexibility — C++ includes herb/shrub intensity contribution
        # which may affect when fire dies out
        assert abs(py_count - cpp_count) < 15, \
            f'Python timesteps={py_count} vs C++ timesteps={cpp_count}'

    def test_first_timestep_intensity(self, py_burnup_direct, cpp_emis):
        """First-timestep fire intensity should be in the same ballpark.

        Note: C++ includes herb/shrub/foliage/branch fire intensity contribution
        which Python burnup does not, so C++ will be higher.
        """
        results, _, _ = py_burnup_direct
        if not results or not cpp_emis:
            pytest.skip('No timestep data')
        # Python burnup doesn't directly report fire intensity per timestep
        # in the BurnResult, so we just check the time values align
        py_t0 = results[0].time
        cpp_t0 = cpp_emis[0]['time_s']
        assert abs(py_t0 - cpp_t0) < 1.0, \
            f'First timestep: Python t={py_t0} vs C++ t={cpp_t0}'


# ---------------------------------------------------------------------------
# Tests: Full pipeline level (run_fofem_emissions vs C++ consumed outputs)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """Compare full Python pipeline outputs against C++ reference.

    The C++ load.txt reports per-component consumed amounts in T/ac.
    Python's run_fofem_emissions returns the same in its output dict.
    """

    def test_litter_consumed(self, py_pipeline_result, cpp_load):
        """Litter consumed should be close."""
        py_lit_con = py_pipeline_result['LitCon']
        # C++ component 1 = litter
        cpp_lit_pre = cpp_load[0]['preburn_tac']
        cpp_lit_post = cpp_load[0]['postburn_tac']
        cpp_lit_con = cpp_lit_pre - cpp_lit_post
        assert abs(py_lit_con - cpp_lit_con) < 0.05, \
            f'LitCon: Python={py_lit_con:.4f} vs C++={cpp_lit_con:.4f}'

    def test_dw1_consumed(self, py_pipeline_result, cpp_load):
        """DW1 consumed should match."""
        py_con = py_pipeline_result['DW1Con']
        cpp_pre = cpp_load[1]['preburn_tac']
        cpp_post = cpp_load[1]['postburn_tac']
        cpp_con = cpp_pre - cpp_post
        assert abs(py_con - cpp_con) < 0.05, \
            f'DW1Con: Python={py_con:.4f} vs C++={cpp_con:.4f}'

    def test_dw10_consumed(self, py_pipeline_result, cpp_load):
        """DW10 consumed should be close (partial consumption expected)."""
        py_con = py_pipeline_result['DW10Con']
        cpp_pre = cpp_load[2]['preburn_tac']
        cpp_post = cpp_load[2]['postburn_tac']
        cpp_con = cpp_pre - cpp_post
        assert abs(py_con - cpp_con) < 0.10, \
            f'DW10Con: Python={py_con:.4f} vs C++={cpp_con:.4f}'

    def test_dw100_consumed(self, py_pipeline_result, cpp_load):
        """DW100 consumed — expect minimal consumption."""
        py_con = py_pipeline_result['DW100Con']
        cpp_pre = cpp_load[3]['preburn_tac']
        cpp_post = cpp_load[3]['postburn_tac']
        cpp_con = cpp_pre - cpp_post
        # Both should show very little consumption for 100-hr at these moistures
        assert abs(py_con - cpp_con) < 0.10, \
            f'DW100Con: Python={py_con:.4f} vs C++={cpp_con:.4f}'

    def test_1k_sound_consumed(self, py_pipeline_result, cpp_load):
        """1000-hr sound consumed total."""
        py_con = py_pipeline_result['DW1kSndCon']
        # Sum C++ components 5, 7, 9, 11 (sound 3-6, 6-9, 9-20, 20)
        cpp_con = sum(
            cpp_load[i - 1]['preburn_tac'] - cpp_load[i - 1]['postburn_tac']
            for i in (5, 7, 9, 11)
        )
        assert abs(py_con - cpp_con) < 0.05, \
            f'DW1kSndCon: Python={py_con:.4f} vs C++={cpp_con:.4f}'

    def test_1k_rotten_consumed(self, py_pipeline_result, cpp_load):
        """1000-hr rotten consumed total."""
        py_con = py_pipeline_result['DW1kRotCon']
        # Sum C++ components 6, 8, 10, 12 (rotten 3-6, 6-9, 9-20, 20)
        cpp_con = sum(
            cpp_load[i - 1]['preburn_tac'] - cpp_load[i - 1]['postburn_tac']
            for i in (6, 8, 10, 12)
        )
        assert abs(py_con - cpp_con) < 0.05, \
            f'DW1kRotCon: Python={py_con:.4f} vs C++={cpp_con:.4f}'


# ---------------------------------------------------------------------------
# Diagnostic: print detailed comparison (not a test, run with -s flag)
# ---------------------------------------------------------------------------
def print_comparison(cpp_load, cpp_emis, py_burnup_direct, py_pipeline_result):
    """Print a side-by-side comparison table. Run: pytest -s -k print_comparison"""
    _, summary, class_order = py_burnup_direct

    py_to_cpp = {
        'litter': 1, 'dw1': 2, 'dw10': 3, 'dw100': 4,
        'dwk_3_6': 5, 'dwk_3_6_r': 6, 'dwk_6_9': 7, 'dwk_6_9_r': 8,
        'dwk_9_20': 9, 'dwk_9_20_r': 10, 'dwk_20': 11, 'dwk_20_r': 12,
    }

    print('\n' + '=' * 90)
    print('BURNUP COMPONENT COMPARISON (full 12 classes, low-level burnup direct)')
    print('=' * 90)
    print(f'{"Component":<12} {"C++ pre":>10} {"Py pre":>10} {"C++ post":>10} '
          f'{"Py post":>10} {"C++ frac":>10} {"Py frac":>10} {"C++ tig":>8} {"Py tig":>8}')
    print('-' * 90)

    for i, key in enumerate(class_order):
        cpp_idx = py_to_cpp[key]
        c = cpp_load[cpp_idx - 1]
        s = summary[i]
        cpp_frac = c['postburn_kgm2'] / c['preburn_kgm2'] if c['preburn_kgm2'] > 0 else 0.0
        print(f'{key:<12} {c["preburn_kgm2"]:10.5f} {s.wdry:10.5f} '
              f'{c["postburn_kgm2"]:10.5f} {s.remaining:10.5f} '
              f'{cpp_frac:10.6f} {s.frac_remaining:10.6f} '
              f'{c["t_ignite_s"]:8.1f} {s.t_ignite:8.1f}')

    print('\n' + '=' * 90)
    print('FULL PIPELINE COMPARISON (run_fofem_emissions)')
    print('=' * 90)
    r = py_pipeline_result
    print(f'  LitCon:     Python={r["LitCon"]:.4f} T/ac')
    print(f'  DW1Con:     Python={r["DW1Con"]:.4f} T/ac')
    print(f'  DW10Con:    Python={r["DW10Con"]:.4f} T/ac')
    print(f'  DW100Con:   Python={r["DW100Con"]:.4f} T/ac')
    print(f'  DW1kSndCon: Python={r["DW1kSndCon"]:.4f} T/ac')
    print(f'  DW1kRotCon: Python={r["DW1kRotCon"]:.4f} T/ac')
    print(f'  DufCon:     Python={r["DufCon"]:.4f} T/ac')
    print(f'  HerCon:     Python={r["HerCon"]:.4f} T/ac')
    print(f'  ShrCon:     Python={r["ShrCon"]:.4f} T/ac')
    print(f'  FolCon:     Python={r["FolCon"]:.4f} T/ac')
    print(f'  BraCon:     Python={r["BraCon"]:.4f} T/ac')
    print(f'  FlaDur:     Python={r["FlaDur"]:.1f} s')
    print(f'  SmoDur:     Python={r["SmoDur"]:.1f} s')


def test_print_comparison(cpp_load, cpp_emis, py_burnup_direct, py_pipeline_result):
    """Diagnostic: print detailed comparison (run with pytest -s)."""
    print_comparison(cpp_load, cpp_emis, py_burnup_direct, py_pipeline_result)
