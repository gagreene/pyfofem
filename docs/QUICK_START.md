# Quick Start

This guide gets `pyfofem` running with the smallest useful examples.

## Install

Runtime install:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

Development install with tests:

```bash
python -m pip install -r requirements-test.txt
python -m pip install -e .[test]
```

## Minimal Emissions Run

`run_fofem_emissions()` is the main entrypoint for fuel consumption, emissions,
burnup, and optional soil heating.

```python
from pyfofem import run_fofem_emissions

result = run_fofem_emissions(
    litter=2.5,
    duff=10.0,
    duff_depth=3.0,
    herb=0.5,
    shrub=0.2,
    crown_foliage=0.1,
    crown_branch=0.2,
    pct_crown_burned=50.0,
    region="InteriorWest",
    cvr_grp="Ponderosa pine",
    season="Summer",
    fuel_category="Natural",
    duff_moist=40.0,
    l_moist=10.0,
    dw10_moist=12.0,
    dw1000_moist=20.0,
    dw1=0.1,
    dw10=0.2,
    dw100=0.3,
    dw1000s=0.4,
    dw1000r=0.1,
    use_burnup=True,
    units="Imperial",
)

print(result["LitCon"])
print(result["PM25F"])
print(result["BurnupError"])
```

Notes:

- Use `units="Imperial"` for `T/ac` and inches.
- Use `units="SI"` for `kg/m^2` and centimeters.
- `use_burnup=False` skips the slower post-frontal burnup model.

## Soil Heating

To include soil heating, provide `soil_family` and set `soil_heating=True`:

```python
from pyfofem import run_fofem_emissions

result = run_fofem_emissions(
    litter=2.5,
    duff=10.0,
    duff_depth=3.0,
    herb=0.5,
    shrub=0.2,
    crown_foliage=0.1,
    crown_branch=0.2,
    pct_crown_burned=50.0,
    region="InteriorWest",
    cvr_grp="Ponderosa pine",
    season="Summer",
    fuel_category="Natural",
    duff_moist=40.0,
    dw10_moist=12.0,
    dw1000_moist=20.0,
    dw1=0.1,
    dw10=0.2,
    dw100=0.3,
    dw1000s=0.4,
    dw1000r=0.1,
    soil_family="Fine-Silt",
    soil_moisture=15.0,
    soil_heating=True,
    units="Imperial",
)

print(result["Lay2"])
print(result["Lay60d"])
```

Accepted `soil_family` values include GUI-style names such as `Fine-Silt`.

## Batch Arrays

Most numeric inputs can be passed as NumPy arrays. This lets `pyfofem`
process many cells in one call.

```python
import numpy as np
from pyfofem import run_fofem_emissions

n = 3
result = run_fofem_emissions(
    litter=np.full(n, 2.5),
    duff=np.full(n, 10.0),
    duff_depth=np.full(n, 3.0),
    herb=np.full(n, 0.5),
    shrub=np.full(n, 0.2),
    crown_foliage=np.full(n, 0.1),
    crown_branch=np.full(n, 0.2),
    pct_crown_burned=np.full(n, 50.0),
    region=np.array(["InteriorWest"] * n, dtype=object),
    cvr_grp=np.array(["Ponderosa pine"] * n, dtype=object),
    season=np.array(["Summer"] * n, dtype=object),
    fuel_category=np.array(["Natural"] * n, dtype=object),
    duff_moist=np.full(n, 40.0),
    dw10_moist=np.full(n, 12.0),
    dw1000_moist=np.full(n, 20.0),
    dw1=np.full(n, 0.1),
    dw10=np.full(n, 0.2),
    dw100=np.full(n, 0.3),
    dw1000s=np.full(n, 0.4),
    dw1000r=np.full(n, 0.1),
    num_workers=1,
    units="Imperial",
)

print(result["PM25F"])
```

For large batches:

- increase `num_workers` to parallelize burnup
- keep `show_progress=True` if you want progress bars

## Emissions Modes

`em_mode` controls which emission-factor path is used:

- `default`: standard single-group emissions
- `legacy`: C++/GUI-style legacy emissions behavior
- `expanded`: separate flaming/coarse-smolder/duff-smolder factors

Example:

```python
result = run_fofem_emissions(..., em_mode="legacy")
```

## Burnup Controls

You can pass advanced burnup parameters through `burnup_kwargs`:

```python
result = run_fofem_emissions(
    ...,
    burnup_kwargs={
        "timestep": 15.0,
        "max_times": 3000,
        "r0": 1.83,
        "dr": 0.4,
    },
)
```

## Examples And Tests

Examples:

- `examples/emissions_batch.py`
- `examples/example_data/fofem_emissions_batch_test.csv`

Run tests:

```bash
python tests/run_unified_tests.py --suite core
```

Run publish-style installed-package tests:

```bash
python tests/run_unified_tests.py --suite core --installed-only
```

