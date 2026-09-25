# PyFOFEM Codebase Reference

PyFOFEM is a Python implementation of the FOFEM fire-effects workflow. This
document describes the current library architecture, supported interfaces, and
maintenance practices. It intentionally omits the historical test-development
record; use version control and local development records when that history is
needed.

## Repository layout

```text
src/pyfofem/
├── __init__.py                 Public exports and package version
├── pyfofem.py                  High-level facade functions and constants
├── components/
│   ├── burnup.py               Burnup engine
│   ├── burnup_calcs.py         Burnup result calculations
│   ├── consumption_calcs.py    Fuel-consumption equations
│   ├── emission_calcs.py       Emission-factor calculation
│   ├── emission_pipeline.py    Emissions orchestration
│   ├── mortality_calcs.py      Tree-mortality equations
│   ├── soil_heating.py         Soil-heating models
│   └── tree_flame_calcs.py     Tree, crown, flame, and scorch utilities
└── supporting_data/            Runtime CSV tables packaged with the library

tests/
├── unit/                       Component and contract tests
├── integration/                Facade and workflow tests
├── regression/                 Regression tests for corrected behavior
└── cpp_parity_live/            Reference-harness and golden-data tooling

reference/fofem_cpp/            Pinned FOFEM reference source
reference/fofem_cpp_overlay/    Reproducible harness overlay
docs/                           User and developer documentation
examples/                       Runnable examples
```

The runtime tables under `src/pyfofem/supporting_data/` are package data.
Changes to their schema, lookup behavior, or package-data configuration require
an installed-wheel test.

## Public interfaces

The package root exports the supported calculation functions. In most user
workflows, start with one of these facade functions:

- `run_fofem_emissions()` runs consumption, burnup, emissions, and optional
  soil heating for one or more fuelbeds.
- `run_burnup()` runs the post-frontal combustion model directly.
- `run_fofem_mortality()` dispatches to `mort_bolchar`, `mort_crnsch`, or
  `mort_crcabe` using a model name and parameter dictionary.
- `soil_heat_from_consumption()` constructs supported soil-heating forcing
  from consumption outputs.
- `soil_heat_campbell()` runs the supported soil-heating model directly.

The component functions remain public for focused analyses. Their docstrings
define parameter units, array behavior, validation, and return values.

## Main data flow

```text
Fuelbed inputs
    │
    ├── Consumption equations ──> consumption totals and remaining fuels
    │                                 │
    │                                 ├── Burnup ──> flaming/smoldering totals
    │                                 │                 │
    │                                 │                 └── Emissions
    │                                 │
    │                                 └── Soil-heating forcing ──> Campbell model
    │
    └── Tree inputs + fire behavior ──> mortality probability
```

`run_fofem_emissions()` coordinates the first branch. Mortality is an
independent tree-level calculation and can use direct fire-injury inputs or
values derived from the tree/flame utilities.

## Model components

### Consumption and burnup

`consumption_calcs.py` supplies litter, duff, herb, shrub, canopy, and
mineral-soil calculations. Inputs use the units stated by each function; the
facade converts between its public fuelbed representation and component units.

`burnup.py` provides the time-stepped burnup engine. `burnup_calcs.py`
calculates summary quantities, including flaming and smoldering duration.
`emission_pipeline.py` owns the integration between consumption, burnup, and
emissions; callers should normally use the facade rather than reconstructing
that sequence themselves.

### Emissions

`calc_smoke_emissions()` supports three factor-selection modes:

- `default`: a standard single factor group.
- `legacy`: the original FOFEM-style factor path.
- `expanded`: independently selected flaming, coarse-smoldering, and
  duff-smoldering factor groups.

Emission factors ship in `supporting_data/emissions_factors.csv`. Preserve
factor units and column names when updating that table.

### Mortality and tree/flame utilities

`mort_bolchar()` handles broadleaf bole-char mortality, `mort_crnsch()`
handles crown-scorch mortality, and `mort_crcabe()` handles cambium-kill
mortality. They accept scalars or broadcast-compatible arrays and return
continuous mortality probabilities.

`tree_flame_calcs.py` contains bark-thickness, canopy-cover, flame-length,
char-height, crown-injury, and scorch-height helpers. Use explicit direct
injury inputs when they are known; derived geometry remains available when
they are not.

### Soil heating

Campbell soil heating is the supported model. `soil_heat_campbell()` uses one
coupled heat, water, humidity, and vapor solver backend for both duff and
non-duff forcing routes. `soil_heat_from_consumption()` is the preferred way
to create forcing from facade outputs:

- Duff heating requires consumed-duff load, pre-fire duff depth, and duff
  moisture.
- Non-duff heating accepts herb/shrub consumption alone, or a burnup result
  when litter or woody fuel contributes.

Massman heat-moisture-vapor modeling is in development and deliberately
unavailable. `soil_heat_massman()` raises `NotImplementedError`; it is not
part of the public package API or the emissions facade.

## Units and arrays

Public functions document their own units. The common conventions are:

| Quantity | Public convention |
|---|---|
| DBH, tree height, crown depth, flame length, scorch height | cm or m as named by the parameter |
| Fuel loads | metric or tons/acre as named by the parameter |
| Moisture | percent where the parameter says percent |
| Fire intensity | kW/m |
| Mortality | probability from 0 to 1 |

Accept scalar values for one record and NumPy-compatible arrays for batches.
Functions generally return a scalar when their primary inputs are scalar and a
one-dimensional array otherwise. Do not rely on implicit broadcasting for
unrelated input dimensions; normalize batch data before calling a component.

## Reference data and scientific validation

The repository pins a FOFEM reference snapshot as a submodule. It is used to
verify routing, units, and numerical results without making the reference
source part of PyFOFEM's public API.

The parity harness is applied from `reference/fofem_cpp_overlay/`; never edit
the pinned submodule as part of ordinary PyFOFEM work. Reference datasets under
`tests/test_data/test_golden_output/` include manifests recording source
revision, harness inputs, toolchain identity, and applicable tolerances.

Do not adjust a scientific tolerance or regenerate a golden dataset merely to
make a test pass. First identify the relevant equation, unit conversion,
routing decision, or source-data change. Regenerate only when the dataset's
own provenance inputs changed and retain proof that scientific CSV values did
not drift unless a deliberately validated reference update requires it.

## Testing

Run the unified suite from the repository root:

```powershell
python tests/run_unified_tests.py --suite ci-smoke
python tests/run_unified_tests.py --suite core
python tests/run_unified_tests.py --suite full
```

Use `--installed-only` after changing package code or package metadata to
verify the installed artifact rather than the checkout:

```powershell
python tests/run_unified_tests.py --suite core --installed-only
python tests/run_unified_tests.py --suite full --installed-only
```

The full suite includes tests that build and execute the reference harness on
supported Windows toolchains. Where that toolchain is unavailable, the
reference-specific tests report a clean skip; unit and integration tests still
run.

Golden datasets have deterministic verification commands. Run the repository's
five `--verify-only` gates after changing a harness, its overlay, a frozen
generator input, or a tolerance-policy route. The commands and their ownership
are maintained beside the golden tooling.

## CI tiers

Continuous integration uses three progressively broader tiers:

| Tier | Trigger | Purpose |
|---|---|---|
| Smoke | Pull requests | Fast cross-platform feedback |
| Core | Pushes to `master` | Broad library regression coverage |
| Full | Scheduled, tagged, or explicitly requested runs | Reference harness, installed-wheel, and golden verification coverage |

Keep smoke tests a subset of core tests. Full coverage is intentionally less
frequent because it builds the reference harness and validates generated
artifacts.

## Maintenance conventions

- Keep new production functions documented with reStructuredText parameter,
  return, and exception fields.
- Keep private helpers before public functions, and alphabetize each group when
  substantively editing a module.
- Add a focused regression test before correcting a defect.
- Preserve pre-existing local, staged, untracked, and submodule work unless its
  owner explicitly authorizes a change.
- Treat TestPyPI publication, production publication, tags, and pushes as
  separate actions requiring explicit approval.

## Current limitations

- Massman heat-moisture-vapor modeling is not available yet.
- Cover-type lookup from SAF, NVCS, and FCCS classifications is not provided;
  callers supply PyFOFEM region and cover-group codes.
- The 1,000-hour fuel-size distribution helper is not currently exposed.
- The FOFEM reference snapshot and bundled reference material have separate
  provenance and redistribution considerations from PyFOFEM's own source.

For user-oriented examples, start with [Quick Start](QUICK_START.md). For
package-release work, consult the maintained release-readiness plan and the
project's provenance records.
