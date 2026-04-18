# pyfofem

`pyfofem` is a Python library for modeling first-order fire effects, porting key FOFEM science and workflows to Python. It provides vectorized APIs for tree mortality, fuel consumption, smoke emissions, burnup, and soil heating.

## Directory Structure

```text
pyfofem/
|-- src/pyfofem/                        # Python library
|   |-- __init__.py                     # Public API re-exports
|   |-- pyfofem.py                      # Core orchestrators
|   `-- components/                     # Specialized computation modules
|-- tests/                              # Unit, golden, and parity tests
|   |-- test_equations_golden.py
|   |-- test_burnup_golden.py
|   |-- test_compare_cpp_python.py
|   |-- test_cpp_comparison.py
|   |-- test_soil_cpp_parity.py
|   |-- compare_cpp_python.py
|   |-- compare_cpp_python_soil.py
|   `-- test_data/                      # Input CSVs and expected outputs
|-- reference/fofem_cpp/                # C++ FOFEM reference source
|-- docs/reference/                     # Literature and reference docs
|-- docs/CODEBASE.md                    # Architecture and model mapping
`-- README.md
```

## Features

- Tree mortality models (`crnsch`, `bolchar`, `crcabe`)
- Fuel consumption for litter, duff, herb, shrub, canopy, mineral soil
- Burnup post-frontal combustion engine (Albini & Reinhardt port)
- Smoke emissions (`legacy`, `default`, `expanded` modes)
- Campbell and Massman soil-heating models
- Integrated soil-heating outputs in `run_fofem_emissions` (`Lay0`, `Lay2`, `Lay4`, `Lay6`, `Lay60d`, `Lay275d`)
- C++ parity scripts/tests for burnup/consumption and soil-heating outputs

## Installation

```bash
git clone https://github.com/gagreene/pyfofem.git
cd pyfofem
pip install numpy pandas scipy tqdm pytest
```

There is no package build file yet (`setup.py` / `pyproject.toml`), so add `src/` to `PYTHONPATH` before importing:

```bash
export PYTHONPATH=/path/to/pyfofem/src:$PYTHONPATH
```

## Usage

```python
from pyfofem import run_fofem_emissions

results = run_fofem_emissions(
    litter=2.5, duff=10.0, duff_depth=3.0, herb=0.5, shrub=0.2,
    crown_foliage=0.1, crown_branch=0.2, pct_crown_burned=50,
    region="InteriorWest", cvr_grp="Ponderosa pine", season="Summer",
    fuel_category="Natural", duff_moist=40, l_moist=10,
    dw10_moist=12, dw1000_moist=20,
    dw1=0.1, dw10=0.2, dw100=0.3, dw1000s=0.4, dw1000r=0.1,
    soil_family="Fine-Silt",      # required when soil_heating is enabled
    soil_moisture=15.0,           # optional mineral-soil moisture (%)
    soil_heating=True,            # bool or dict of advanced overrides
)

print(results["PM10F"])
print(results["DufCon"])
print(results["Lay2"])
```

To match legacy GUI/C++ emissions behavior, pass `em_mode="legacy"`.
In this mode, smoldering NOx (`NOXS`) is expected to be `0` by design.
In `expanded` mode, default smolder group 7 (`CWDRSC`) also has `NOx as NO = 0`,
so `NOXS` mainly comes from the duff group unless you change factor groups.

## Examples

Example scripts live in `examples/`:

- `examples/emissions_batch.py`

Example input datasets are in `examples/example_data/`:

- `fofem_emissions_batch_test.csv`

### Soil-heating options in `run_fofem_emissions`

- `soil_heating`: `False` (default), `True`, or `dict`
- `soil_family`: required when soil heating is enabled; accepts GUI-style names (for example `Fine-Silt`) or internal family names
- `soil_moisture`: optional top-level mineral-soil moisture override (%)
- `soil_heating` dict overrides:
  - `soil_moisture`
  - `start_temp`
  - `efficiency_wl`
  - `efficiency_hs`
  - `efficiency_duff`
  - `depth_layers_cm` (13 depths expected)
  - `timestep_s`

Soil moisture precedence during soil-heating runs:
1. Top-level `soil_moisture`
2. `soil_heating["soil_moisture"]`
3. `moisture_regime` soil value
4. Clipped `duff_moist` fallback (`0..25%`)

## Testing

Run the full test suite:

```bash
pytest tests/
```

Run the unified publish-oriented suite (recommended for CI/package checks):

```bash
# Fast publish-safe suite
python tests/run_unified_tests.py --suite core

# Extended suite with parity/comparison tests
python tests/run_unified_tests.py --suite full
```

For package-validation workflows where you want to ensure tests are running
against the installed package (not local `src/`), use:

```bash
python tests/run_unified_tests.py --suite core --installed-only
```

### Packaging pipeline usage

PyPI wheel/sdist check:

```bash
python -m pip install .
python tests/run_unified_tests.py --suite core --installed-only
```

Conda recipe `test:commands` example:

```yaml
test:
  commands:
    - python tests/run_unified_tests.py --suite core --installed-only
```

Key parity checks:

- `tests/test_compare_cpp_python.py` and `tests/compare_cpp_python.py` compare Python outputs against C++ multi-case CSV harness results.
- `tests/test_cpp_comparison.py` compares Python against `reference/fofem_cpp/load.txt` and `emis.txt`.
- `tests/test_soil_cpp_parity.py` and `tests/compare_cpp_python_soil.py` compare `Lay*` soil-heating outputs against C++ `reference/fofem_cpp/soil.tmp`.

## License

MIT. See [LICENSE](LICENSE) when added to this repository.
