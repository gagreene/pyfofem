
# pyfofem

**pyfofem** is a Python library for modeling fire effects on forest vegetation, porting the science of the First Order Fire Effects Model (FOFEM) to Python. It provides vectorized, DataFrame-friendly functions for estimating tree mortality, fuel consumption, emissions, soil heating, and related fire effects using published models and species-specific parameters.

## Directory Structure

```
pyfofem/
├── src/pyfofem/           # Python library (core + components)
│   ├── pyfofem.py         # Core API: fire effects, mortality, emissions, orchestrator
│   ├── __init__.py        # Public API re-exports
│   └── components/        # Submodules: burnup, mortality, soil, helpers
│   └── supporting_data/   # Species codes, emission factors, FOFEM data
├── tests/                 # Unit and golden tests, test data
├── CODEBASE.md            # Full codebase and science mapping
├── README.md
```

## Features

- **Comprehensive fire effects modeling**: tree mortality (crown scorch, bole char, cambium kill), fuel consumption (herb, shrub, litter, duff, woody), smoke emissions, and soil heating.
- **Vectorized, array-friendly API**: All functions accept scalars or NumPy arrays for batch processing.
- **Published, species-specific models**: Implements FOFEM equations and species lookups.
- **Burnup post-frontal combustion engine**: Python port of Albini & Reinhardt's model, matching C++ outputs.
- **Soil heating models**: Campbell (1D equilibrium) and Massman HMV (non-equilibrium) with `scipy.integrate`.
- **Smoke emissions**: Default and expanded emission factor systems, CSV-driven.
- **Data-driven**: Bundled species code lookups, emission factors, and FOFEM 6.7 reference data.
- **Test suite**: Golden tests, batch input/output, and science validation.

## Installation

Clone the repository:

```bash
git clone https://github.com/gagreene/pyfofem.git
cd pyfofem
# No requirements.txt or setup.py; install dependencies manually:
pip install numpy pandas scipy tqdm
```

## Usage

Import from the top-level package (after adding `src/` to your `PYTHONPATH`):

```python
from pyfofem import (
    run_fofem_emissions, run_fofem_mortality,
    calc_scorch_ht, calc_flame_length, mort_crnsch
)

# Example: Estimate crown scorch mortality for two trees
Pm = mort_crnsch(
    spp=['PIPO', 'PSME'],
    dbh=[30, 25],
    ht=[20, 18],
    crown_depth=[8, 7],
    fire_intensity=[5000, 4000]
)
print(Pm)

# Example: Run full fuel consumption and emissions model
results = run_fofem_emissions(
    litter=2.5, duff=10.0, duff_depth=3.0, herb=0.5, shrub=0.2,
    crown_foliage=0.1, crown_branch=0.2, pct_crown_burned=50,
    region='InteriorWest', cvr_grp='Ponderosa pine', season='Summer',
    fuel_category='Natural', duff_moist=40, l_moist=10, dw10_moist=12, dw1000_moist=20,
    dw1=0.1, dw10=0.2, dw100=0.3, dw1000s=0.4, dw1000r=0.1
)
print(results['PM10F'], results['DufCon'])
```

## Testing

Unit and golden tests are in `tests/`. Test data and batch input/output are provided for science validation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.