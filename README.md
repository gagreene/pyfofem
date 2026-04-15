
# pyfofem

**pyfofem** is a Python library for modelling fire effects on forest vegetation, porting the science of the First Order Fire Effects Model (FOFEM) to Python. It provides vectorized, array and DataFrame-friendly functions for estimating tree mortality, fuel consumption, smoke emissions, soil heating, and related fire effects using published models and species-specific parameters.

## Directory Structure

```
pyfofem/
├── src/pyfofem/                    # Python library
│   ├── __init__.py                 # Public API re-exports
│   ├── pyfofem.py                  # Core orchestrators: run_fofem_mortality, run_fofem_emissions
│   └── components/                 # Specialized computation modules
│       ├── tree_flame_calcs.py     # Tree geometry, scorch height, bark thickness, canopy cover
│       ├── mortality_calcs.py      # Crown scorch, bole char, cambium kill mortality models
│       ├── consumption_calcs.py    # Fuel consumption (litter, herb, shrub, duff, canopy, soil)
│       ├── burnup.py               # Albini & Reinhardt post-frontal combustion engine
│       ├── burnup_calcs.py         # Burnup wrapper and per-cell dispatch functions
│       ├── emission_calcs.py       # Smoke emissions from consumption
│       ├── soil_heating.py         # Soil temperature models (Campbell, Massman)
│       ├── _component_helpers.py   # Shared scalar/array utilities
│       └── supporting_data/        # Species codes, emission factors, FOFEM 6.7 data
├── tests/                          # Unit and golden tests
│   ├── fofem_emissions_example.py  # Batch-processing example (60,840-row CSV)
│   ├── test_equations_golden.py    # Golden tests for consumption equations
│   ├── test_burnup_golden.py       # Burnup regression tests vs. C++ baseline
│   ├── test_burnup_array.py        # Burnup array-mode tests
│   ├── test_consumption_calcs_array.py
│   └── test_data/                  # Input CSVs and expected outputs for validation
├── dependencies/fofem_cpp/         # C++ FOFEM reference source (read-only)
├── docs/reference/                 # Published literature and papers
├── CODEBASE.md                     # Detailed architecture and science mapping
└── README.md
```

## Features

- **Comprehensive fire-effects modeling**: tree mortality (crown scorch, bole char, cambium kill), fuel consumption (herb, shrub, litter, duff, woody debris, canopy), smoke emissions, and soil heating.
- **Vectorized, array-friendly API**: all functions accept scalars or NumPy arrays; batch thousands of cells in a single call.
- **Published, species-specific models**: implements FOFEM equations and species lookups for 60+ tree species.
- **Burnup post-frontal combustion engine**: Python port of Albini & Reinhardt's (1989) model, validated against C++ outputs.
- **Soil heating models**: Campbell (1D equilibrium) and Massman HMV (non-equilibrium heat-moisture-vapor) using `scipy.integrate.solve_ivp`.
- **Smoke emissions**: three modes are supported: `legacy` (original C++ `ES_Calc` parity), `default` (single EF-group), and `expanded` (flame/coarse-smolder/duff groups) using the bundled emission-factor database.
- **Parallel execution**: `run_fofem_emissions` dispatches per-cell burnup in parallel via `concurrent.futures.ProcessPoolExecutor`.
- **Data-driven**: bundled species code lookups, emission factors, and FOFEM 6.7 reference data.
- **Test suite**: golden tests, batch input/output validation, and science regression tests.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gagreene/pyfofem.git
cd pyfofem
pip install numpy pandas scipy tqdm
```

There is no `setup.py` or `pyproject.toml`. Add `src/` to your `PYTHONPATH` before importing:

```bash
export PYTHONPATH=/path/to/pyfofem/src:$PYTHONPATH
```

## Usage

Import from the top-level package:

```python
from pyfofem import (
    run_fofem_emissions, run_fofem_mortality,
    calc_scorch_ht, calc_flame_length, mort_crnsch,
    consm_duff, calc_smoke_emissions,
)
```

### High-level orchestrators

**Tree mortality** — select a model by name; accepts scalar or array inputs:

```python
from pyfofem import run_fofem_mortality

# Crown scorch model (conifers/general)
Pm = run_fofem_mortality(
    'crnsch',
    spp=['PIPO', 'PSME'],
    dbh=[30, 25],
    ht=[20, 18],
    crown_depth=[8, 7],
    fire_intensity=[5000, 4000],
)
print(Pm)  # array of mortality probabilities [0, 1]
```

Available mortality functions: `'crnsch'` (crown scorch), `'bolchar'` (bole char; hardwoods), `'crcabe'` (cambium kill; conifers).

**Full fuel consumption and emissions pipeline:**

```python
from pyfofem import run_fofem_emissions

results = run_fofem_emissions(
    litter=2.5, duff=10.0, duff_depth=3.0, herb=0.5, shrub=0.2,
    crown_foliage=0.1, crown_branch=0.2, pct_crown_burned=50,
    region='InteriorWest', cvr_grp='Ponderosa pine', season='Summer',
    fuel_category='Natural', duff_moist=40, l_moist=10,
    dw10_moist=12, dw1000_moist=20,
    dw1=0.1, dw10=0.2, dw100=0.3, dw1000s=0.4, dw1000r=0.1,
)
print(results['PM10F'])   # PM10 flaming emissions (g/m² or lb/ac)
print(results['DufCon'])  # duff consumed (kg/m² or T/ac)
```

To match legacy FOFEM GUI/C++ emissions behavior, pass `em_mode='legacy'`.

Returns a dict with 70+ keys covering pre/consumed/post loads for every fuel class, emissions (PM10, PM2.5, CH4, CO, CO2, NOx, SO2), burnup durations, and carbon.

### Emissions modes (`em_mode`)

`run_fofem_emissions(..., em_mode=...)` supports three emissions modes:

- `legacy`
  Replicates the original C++ Burnup `ES_Calc` pathway (`f_CriInt < 0` behavior). Uses fixed combustion-efficiency-derived factors (not the emissions-factor CSV groups). Use this when matching legacy FOFEM GUI/C++ outputs.
- `default`
  Uses one emissions-factor group (`ef_group`, default group 3) for both flaming and smoldering emissions. This is the simplest CSV-driven mode.
- `expanded`
  Uses three emissions-factor groups, matching the newer C++ architecture:
  `ef_group` for flaming, `ef_smoldering_group` for coarse-wood smoldering, and `ef_duff_group` for duff smoldering. Duff is split from smoldering and reported separately in `*_Duff` outputs.

Typical choices:

- Use `legacy` for parity checks against historical GUI runs.
- Use `expanded` for modern factor-group workflows where duff/coarse/flaming factors are separated.
- Use `default` for simple single-group analyses.

### Component-level functions

All component functions are also importable directly:

```python
from pyfofem import (
    # Tree geometry and fire behavior
    calc_scorch_ht, calc_flame_length, calc_char_ht,
    calc_bark_thickness, calc_crown_length_vol_scorched, calc_canopy_cover,

    # Mortality models
    mort_crnsch, mort_bolchar, mort_crcabe,

    # Fuel consumption
    consm_litter, consm_herb, consm_shrub,
    consm_duff, consm_canopy, consm_mineral_soil,

    # Carbon and emissions
    calc_carbon, calc_smoke_emissions,

    # Burnup engine
    run_burnup, FuelParticle, BurnResult, BurnSummaryRow,

    # Soil heating
    soil_heat_campbell, soil_heat_massman,

    # Data lookups
    SPP_CODES, CONSUMPTION_VARS,
    REGION_CODES, CVR_GRP_CODES, SEASON_CODES, FUEL_CATEGORY_CODES,
    get_moisture_regime,
)
```

### Named moisture regimes

```python
from pyfofem import get_moisture_regime

regime = get_moisture_regime('dry')
# Returns dict: {'duff_moist': ..., 'dw10_moist': ..., 'dw1000_moist': ...}
```

Available regimes: `'wet'`, `'moderate'`, `'dry'`, `'very dry'`.

## Public API Summary

| Module | Key functions / classes |
|---|---|
| `pyfofem.py` | `run_fofem_mortality`, `run_fofem_emissions` |
| `tree_flame_calcs` | `calc_scorch_ht`, `calc_flame_length`, `calc_char_ht`, `calc_bark_thickness`, `calc_crown_length_vol_scorched`, `calc_canopy_cover` |
| `mortality_calcs` | `mort_crnsch`, `mort_bolchar`, `mort_crcabe` |
| `consumption_calcs` | `consm_litter`, `consm_herb`, `consm_shrub`, `consm_duff`, `consm_canopy`, `consm_mineral_soil`, `calc_carbon`, `get_moisture_regime` |
| `burnup_calcs` | `run_burnup`, `gen_burnup_in_file` |
| `burnup` | `FuelParticle`, `BurnResult`, `BurnSummaryRow`, `BurnupValidationError`, `burnup` |
| `emission_calcs` | `calc_smoke_emissions` |
| `soil_heating` | `soil_heat_campbell`, `soil_heat_massman` |

## Testing

Run tests with pytest from the repository root:

```bash
pytest tests/
```

- **`test_equations_golden.py`** — parametrized unit tests for consumption equations (duff, herb, shrub, canopy) against golden CSV outputs.
- **`test_burnup_golden.py`** — regression tests comparing the Python burnup port against C++ baseline outputs.
- **`test_burnup_array.py`** — array-mode burnup tests.
- **`test_consumption_calcs_array.py`** — vectorized consumption function tests.
- **`fofem_emissions_example.py`** — batch-processing demo over a 60,840-row input CSV; prints per-row timing and summary statistics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
