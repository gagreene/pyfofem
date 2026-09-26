# pyfofem

`pyfofem` is a Python library for modelling first-order fire effects. Its calculations are validated against the pinned FOFEM C++ reference implementation. It provides vectorized APIs for tree mortality, fuel consumption, smoke emissions, burnup, and soil heating.

Start with the quick start guide: [docs/QUICK_START.md](docs/QUICK_START.md)

## Directory Structure

```text
pyfofem/
|-- src/pyfofem/                        # Python library
|   |-- __init__.py                     # Public API re-exports
|   |-- pyfofem.py                      # Core orchestrators
|   `-- components/                     # Specialized computation modules
|-- tests/                              # Unit, golden, and parity tests (pytest package)
|   |-- __init__.py                     # Makes tests/ a package for qualified imports
|   |-- _support.py                     # Shared path constants (no src/ sys.path insert)
|   |-- conftest.py                     # Fixtures, markers, installed-only session check
|   |-- run_unified_tests.py            # `--suite core|full`, `--installed-only` runner
|   |-- prepare_cpp_reference.py        # Regenerates reference fixtures
|   |-- compare_cpp_python_soil_heating.py  # Lay* driver; run as a tests package module
|   |-- unit/                           # Golden-CSV + reference-independent unit tests
|   |-- integration/                    # Full-pipeline (`run_fofem_emissions`) tests
|   |-- regression/                     # Behavior regression tests
|   |-- cpp_parity_live/                # Tests requiring the compiled reference
|   `-- test_data/                      # Input CSVs and expected outputs
|-- examples/                           # Batch/array usage driver + example data
|-- development/burnup_array/           # Experimental prototype, outside the test gate
|-- reference/fofem_cpp/                # Pinned FOFEM reference source
|-- docs/reference/                     # Literature and reference docs
|-- docs/CODEBASE.md                    # Architecture and model mapping
`-- README.md
```

## Features

- Tree mortality models (`crnsch`, `bolchar`, `crcabe`)
- Fuel consumption for litter, duff, herb, shrub, canopy, mineral soil
- Burnup post-frontal combustion engine based on Albini & Reinhardt
- Smoke emissions (`legacy`, `default`, `expanded` modes)
- Campbell soil-heating model (Massman HMV is in development and unavailable)
- Integrated soil-heating outputs in `run_fofem_emissions` (`Lay0`, `Lay2`, `Lay4`, `Lay6`, `Lay60d`, `Lay275d`)
- Reference-validation scripts/tests for burnup, consumption, and soil-heating outputs

## Python support

PyFOFEM supports Python 3.11, 3.12, 3.13, and 3.14. GitHub Actions runs
smoke tests on Ubuntu for every supported version and on macOS and Windows for
Python 3.12 on pull requests. It runs the core suite on Ubuntu for every
supported version on updates to `master`.
## Installation

Install the published distribution with pip or uv:

```bash
python -m pip install pyfofem-fire-effects
# or
uv pip install pyfofem-fire-effects
`$([Environment]::NewLine)
The distribution name differs from the Python import package. Use
`import pyfofem` in Python code.

Install [uv](https://docs.astral.sh/uv/) and create the project environment:

```bash
git clone https://github.com/gagreene/pyfofem.git
cd pyfofem
uv sync
```

For development and the supported test suite, install the test extra and the
repository's `dev` dependency group, then use the locked environment:

```bash
uv sync --all-extras --group dev
uv run python tests/run_unified_tests.py --suite core
```

`uv.lock` records the reproducible development and CI resolution. Users who
prefer pip can install the package with `python -m pip install .`; contributors
using pip should add the test extra with `python -m pip install -e .[test]`.
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

To match original FOFEM legacy emissions behavior, pass `em_mode="legacy"`.
In this mode, smoldering NOx (`NOXS`) is expected to be `0` by design.
In `expanded` mode, default smolder group 7 (`CWDRSC`) also has `NOx as NO = 0`,
so `NOXS` mainly comes from the duff group unless you change factor groups.

## Output variables

Scalar calls return scalar values. When any modelled input is an array,
corresponding outputs are NumPy arrays with one value per input case. Fuel-load
and consumption outputs use the selected unit system: T/ac for `Imperial` and
kg/m² for `SI`. Emissions use lb/acre for `Imperial` and g/m² for `SI`.

### Mortality outputs

`run_fofem_mortality()` returns the mortality probability directly; it does
not return a dictionary. The result is dimensionless and ranges from 0 (tree
survives) to 1 (tree dies). Unsupported species/model combinations can return
`NaN`. Scalar inputs produce a `float`, while array inputs produce a NumPy
array.

| Model | Output | Description |
|---|---|---|
| `bolchar` | Mortality probability | Probability of post-fire mortality from the bole-char model. |
| `crnsch` | Mortality probability | Probability of post-fire mortality from the crown-scorch model. |
| `crcabe` | Mortality probability | Probability of post-fire mortality from the cambium-kill model. |

### Consumption, emissions, and soil-heating outputs

`run_fofem_emissions()` returns a dictionary. The following table includes
every key that can appear. The seven `*_Duff` keys depend on `em_mode`, and the
six `Lay*` keys are present only when soil heating is enabled.

| Variable | Units<br>(SI, Imperial) | Description |
|---|---|---|
| `LitPre`, `LitCon`, `LitPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire litter load. |
| `DW1Pre`, `DW1Con`, `DW1Pos` | kg/m², T/ac | Pre-fire, consumed, and post-fire 1-hour down woody fuel. |
| `DW10Pre`, `DW10Con`, `DW10Pos` | kg/m², T/ac | Pre-fire, consumed, and post-fire 10-hour down woody fuel. |
| `DW100Pre`, `DW100Con`, `DW100Pos` | kg/m², T/ac | Pre-fire, consumed, and post-fire 100-hour down woody fuel. |
| `DW1kSndPre`, `DW1kSndCon`, `DW1kSndPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire sound 1000-hour down woody fuel, summed across diameter classes. |
| `DW1kRotPre`, `DW1kRotCon`, `DW1kRotPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire rotten 1000-hour down woody fuel, summed across diameter classes. |
| `DufPre`, `DufCon`, `DufPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire duff load. |
| `HerPre`, `HerCon`, `HerPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire herbaceous fuel. |
| `ShrPre`, `ShrCon`, `ShrPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire shrub fuel. |
| `FolPre`, `FolCon`, `FolPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire crown foliage. |
| `BraPre`, `BraCon`, `BraPos` | kg/m², T/ac | Pre-fire, consumed, and post-fire crown branch fuel. |
| `MSE` | % | Mineral soil exposure. |
| `DufDepPre`, `DufDepCon`, `DufDepPos` | cm, in | Pre-fire, consumed, and post-fire duff depth. |
| `FlaDur` | s | Duration through the last timestep with flaming consumption above the Burnup reporting threshold. |
| `SmoDur` | s | Duration through the last timestep with smoldering consumption above the Burnup reporting threshold. |
| `FlaCon` | kg/m², T/ac | Total fuel consumed in the flaming phase. |
| `SmoCon` | kg/m², T/ac | Total fuel consumed in the smoldering phase. |
| `Lit-Equ` | Equation ID | Litter-consumption equation selected for the case. |
| `DufCon-Equ` | Equation ID | Duff-consumption equation selected for the case. |
| `DufRed-Equ` | Equation ID | Duff-depth-reduction equation selected for the case. |
| `MSE-Equ` | Equation ID | Mineral-soil-exposure equation selected for the case. |
| `Herb-Equ` | Equation ID | Herbaceous-consumption equation selected for the case. |
| `Shrub-Equ` | Equation ID | Shrub-consumption equation selected for the case. |
| `BurnupLimitAdj` | Code | Recoverable Burnup input-adjustment code. `0` means no adjustment; concatenated digits identify multiple adjustments. See the code list below. |
| `BurnupError` | Code | Burnup outcome code. `0` means success; nonzero values identify a failure. See the code list below. |
| `PM10F`, `PM10S` | g/m², lb/acre | PM10 emissions from flaming and total smoldering combustion. |
| `PM25F`, `PM25S` | g/m², lb/acre | PM2.5 emissions from flaming and total smoldering combustion. |
| `CH4F`, `CH4S` | g/m², lb/acre | Methane emissions from flaming and total smoldering combustion. |
| `COF`, `COS` | g/m², lb/acre | Carbon monoxide emissions from flaming and total smoldering combustion. |
| `CO2F`, `CO2S` | g/m², lb/acre | Carbon dioxide emissions from flaming and total smoldering combustion. |
| `NOXF`, `NOXS` | g/m², lb/acre | Nitrogen oxides, reported as NO, from flaming and total smoldering combustion. |
| `SO2F`, `SO2S` | g/m², lb/acre | Sulfur dioxide emissions from flaming and total smoldering combustion. |
| `PM10S_Duff` | g/m², lb/acre | Duff-only smoldering PM10. Present for `legacy` and `expanded` modes. |
| `PM25S_Duff` | g/m², lb/acre | Duff-only smoldering PM2.5. Present for `legacy` and `expanded` modes. |
| `CH4S_Duff` | g/m², lb/acre | Duff-only smoldering methane. Present for `legacy` and `expanded` modes. |
| `COS_Duff` | g/m², lb/acre | Duff-only smoldering carbon monoxide. Present for `legacy` and `expanded` modes. |
| `CO2S_Duff` | g/m², lb/acre | Duff-only smoldering carbon dioxide. Present for `legacy` and `expanded` modes. |
| `NOXS_Duff` | g/m², lb/acre | Duff-only smoldering nitrogen oxides, reported as NO. Present for `legacy` and `expanded` modes. |
| `SO2S_Duff` | g/m², lb/acre | Duff-only smoldering sulfur dioxide. Present for `legacy` and `expanded` modes. |
| `Lay0` | °C | Maximum modelled mineral-soil surface temperature. Present only when soil heating is enabled. |
| `Lay2` | °C | Maximum modelled mineral-soil temperature at 2 cm depth. Present only when soil heating is enabled. |
| `Lay4` | °C | Maximum modelled mineral-soil temperature at 4 cm depth. Present only when soil heating is enabled. |
| `Lay6` | °C | Maximum modelled mineral-soil temperature at 6 cm depth. Present only when soil heating is enabled. |
| `Lay60d` | Layer index | Deepest requested soil layer whose modelled temperature exceeds 60 °C; `-1` means no layer exceeded the threshold. With the default 1-cm depth grid, the index is also the depth in cm. |
| `Lay275d` | Layer index | Deepest requested soil layer whose modelled temperature exceeds 275 °C; `-1` means no layer exceeded the threshold. With the default 1-cm depth grid, the index is also the depth in cm. |

Units in paired entries are listed in the table header's **SI, Imperial**
order. The `PM10S`, `PM25S`, `CH4S`, `COS`, `CO2S`, `NOXS`, and `SO2S` values include
duff smoldering. Their corresponding `*_Duff` values report the duff-only
portion rather than an additional quantity to add to the total.

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

### Burnup status codes in `run_fofem_emissions`

`run_fofem_emissions()` returns two burnup-status fields:

- `BurnupError`: hard burnup failure code. `0` means burnup ran successfully.
- `BurnupLimitAdj`: clipping/adjustment code for recoverable inputs. `0` means no clipping was applied.

When `BurnupError != 0`, the burnup model does not run for that case and
`pyfofem` falls back to simplified consumption-duration defaults for the
emissions pipeline.

`BurnupError` codes:

- `0`: success
- `10`: `fistart` starting fire intensity, below **40 kW/m²**. The accepted range is
  **40 to 100,000 kW/m²**, inclusive; values above the maximum are clipped
  under adjustment code `1`.
- `11`: `ti` surface fire residence time, below **10 s**. The accepted range
  is **10 to 200 s**, inclusive; values above the maximum are clipped under
  adjustment code `2`.
- `12`: `u` windspeed at the top of the fuel bed, below **0 m/s**. The
  accepted range is **0 to 5 m/s**, inclusive; values above the maximum are
  clipped under adjustment code `3`.
- `13`: `tamb_c` ambient temperature in degrees Celsius, below **-40 °C**.
  The accepted range is **-40 to 40 °C**, inclusive; values above the maximum
  are clipped under adjustment code `5`.
- `14`: `dfm` duff moisture content as a fraction of dry weight, above
  **1.972** (**197.2%** in the public `duff_moist` input). When duff is
  present, the accepted range is **0.1 to 1.972** (**10% to 197.2%**),
  inclusive; values below the minimum are clipped under adjustment code `6`.
  Duff moisture is not range-checked when the duff load is zero.
- `15`: fire cannot dry the fuel. This is a calculated physical failure rather
  than a separate input-range threshold.
- `16`: no fuel ignited within the residence time. This is a calculated
  physical failure rather than a separate input-range threshold.
- `20`: `wdry` oven-dry fuel loading, outside **(1e-8, 1e6) kg/m²**.
- `21`: `ash` mineral ash content as a dry-mass fraction, outside
  **(0.0001, 0.1)**.
- `22`: `htval` low heat of combustion, outside **(1e7, 3e7) J/kg**.
- `23`: `fmois` fuel moisture content as a fraction of dry weight, outside
  **(0.01, 3.0)** (**1% to 300%**).
- `24`: `dendry` oven-dry fuel mass density, outside **(200, 1000) kg/m³**.
- `25`: `sigma` fuel-particle surface-area-to-volume ratio, outside
  **(4, 10,000) m⁻¹**.
- `26`: `cheat` fuel specific heat capacity, outside
  **(1000, 3000) J/(kg·K)**.
- `27`: `condry` oven-dry fuel thermal conductivity, outside
  **(0.025, 0.25) W/(m·K)**.
- `28`: `tpig` piloted-ignition temperature, outside **(200, 400) °C**.
- `29`: `tchar` end-of-pyrolysis char temperature, outside
  **(250, 500) °C**.
- `90`: no fuel particles; every fuel loading is less than or equal to zero.
- `91`: `ntimes` maximum number of simulation timesteps, is less than or
  equal to zero; this is controlled by `burnup_kwargs["max_times"]`.
- `99`: unexpected burnup exception

Square brackets or the word "inclusive" above indicate valid endpoints. The
parenthesized fuel-particle ranges for codes `20`-`29` are strict: values equal
to either endpoint are rejected.

`BurnupLimitAdj` codes are concatenated digits when more than one adjustment is
applied. For example, `13` means codes `1` and `3` both occurred, and `246`
means codes `2`, `4`, and `6` occurred.

- `0`: no clipping applied
- `1`: `fistart` starting fire intensity, above **100,000 kW/m²**; clipped
  to **100,000 kW/m²**.
- `2`: `ti` surface fire residence time, above **200 s**; clipped to
  **200 s**.
- `3`: `u` windspeed at the top of the fuel bed, above **5 m/s**; clipped to
  **5 m/s**.
- `4`: `d` fuel bed depth, below **0.1 m** or above **5 m**; clipped to the
  nearest endpoint of the inclusive **0.1 to 5 m** range.
- `5`: `tamb_c` ambient temperature in degrees Celsius, above **40 °C**;
  clipped to **40 °C**.
- `6`: `dfm` duff moisture content as a fraction of dry weight, below
  **0.1** when duff is present; clipped to **0.1** (**10%** in the public
  `duff_moist` input).

## Testing

Run the full supported suite. `pyproject.toml` sets `testpaths = ["tests"]`,
so plain `pytest` (or `python -m pytest`) collects only the supported
package suite under `tests/` and does not touch the experimental prototype
below:

```bash
python -m pytest
```

Run the unified publish-oriented suite (recommended for CI/package checks):

```bash
# Fastest, representative pull-request checks
python tests/run_unified_tests.py --suite ci-smoke

# Fast publish-safe suite
python tests/run_unified_tests.py --suite core

# Extended suite with parity/comparison tests
python tests/run_unified_tests.py --suite full
```

Run the standalone Lay* soil-heating reference comparison from the repository
root so its ``tests`` package import resolves correctly:

```bash
python -m tests.compare_cpp_python_soil_heating
```

This diagnostic reads the pinned `reference/fofem_cpp/soil.tmp` fixture, exits
nonzero when a comparison exceeds its embedded tolerance, and is not part of
the unified test suites.

`.github/workflows/ci.yml` runs `ci-smoke` on pull requests and `core` on
pushes to `master`. `core` validates Python behavior against committed golden
data and does not build or run the C++ reference. The `full` suite invokes the
live C++ harness and golden generators; it is intentionally excluded from
ordinary CI and release checks. Core reads the pinned C++ Git revision and every
golden manifest, so a changed reference commit fails before it can be treated as
a current golden baseline. Run `full` only when that failure identifies a pinned
upstream C++ change or when a deliberate Python/parity investigation needs new C++
evidence.

### Experimental prototype: `development/burnup_array`

`development/burnup_array` is a non-production, array-based burnup
prototype. It is **outside the default/release test gate** — `testpaths`
does not include it, `run_unified_tests.py` does not run it, and it is not
part of `core` or `full`.

Its explicit, separate diagnostic invocation:

```bash
python -m pytest development/burnup_array/tests -q
```

As of this writing that command **fails during collection**, not just an
individual test: `test_consumption_calcs_array.py` imports
`burnup_array_calcs` as a top-level module, but `burnup_array_calcs.py`
itself uses a relative import (`from .burnup_array_kernel import ...`),
raising `ImportError: attempted relative import with no known parent
package`. This is a known, tracked failure — not a skip, and not silently
part of the supported suite. Maintain it independently of the library's
supported test suites.

For package-validation workflows where you want to ensure tests are running
against the installed package (not local `src/`), use:

```bash
python tests/run_unified_tests.py --suite core --installed-only
```

### Packaging pipeline usage

PyPI wheel/sdist check:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python -m pip install .
python tests/run_unified_tests.py --suite core --installed-only
```

Conda recipe `test:commands` example:

```yaml
test:
  commands:
    - python tests/run_unified_tests.py --suite core --installed-only
```

The Conda recipe lives in `conda-recipe/`. See `conda-recipe/README.md` for
build and test commands.

Reference-validation tooling and deterministic golden-data verification live
under `tests/cpp_parity_live/`. See [CODEBASE.md](docs/CODEBASE.md) for the
current test tiers and maintenance guidance. Release provenance and attribution
for bundled FOFEM-derived data are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

PyFOFEM source is licensed under the [MIT License](LICENSE).
Bundled FOFEM-derived runtime tables and their provenance are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
