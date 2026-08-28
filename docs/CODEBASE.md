# pyfofem - Codebase Reference

This document describes the architecture, data flow, and conventions of both the
Python `pyfofem` library and the C++ FOFEM reference it ports.  It serves as the
single source of truth for contributors and reviewers.

---

## Repository Layout

```text
pyfofem/
|-- src/pyfofem/                       # <- Python library (the deliverable)
|   |-- __init__.py                    #    Public API re-exports (from pyfofem.py)
|   |-- pyfofem.py                     #    Core orchestrator module
|   |-- components/
|   |   |-- __init__.py                #    Re-exports every component's public symbols
|   |   |-- _component_helpers.py      #    Shared scalar/array plumbing (cross-cutting)
|   |   |-- burnup.py                  #    Albini & Reinhardt burnup engine
|   |   |-- burnup_calcs.py            #    Burnup adapters / class mapping / per-cell worker
|   |   |-- consumption_calcs.py       #    Consumption equations
|   |   |-- emission_calcs.py          #    Emissions modes
|   |   |-- emission_pipeline.py       #    run_fofem_emissions orchestration helpers
|   |   |-- mortality_calcs.py         #    Mortality equations
|   |   |-- tree_flame_calcs.py        #    Fire behavior + geometry helpers
|   |   `-- soil_heating.py            #    Campbell + Massman HMV soil models
|   `-- supporting_data/
|       |-- species_codes_lut.csv      #    Species <-> FOFEM-code mapping
|       `-- FOFEM6.7/                  #    Bundled FOFEM data files
|
|-- reference/fofem_cpp/               # <- Official C++ FOFEM reference source
|   |-- FOF_UNIX/                      #    Portable core science code
|   |-- FOF_DLL/                       #    Windows DLL + Massman HMV solver
|   |-- FOF_GUI/                       #    Windows .NET GUI
|   `-- SWIG/                          #    Auto-generated C# interop
|
|-- docs/reference/
|   |-- code/burnup/                   #    Standalone burnupw.cpp baseline
|   `-- papers/                        #    Literature references
|
|-- tests/                             #    pytest package (`tests/__init__.py`); testpaths=["tests"]
|   |-- __init__.py                    #    Package marker for `tests.*` qualified imports
|   |-- _support.py                    #    Shared path constants; never inserts src/ onto sys.path
|   |-- conftest.py                    #    Fixtures, marker registration, installed-only session hook
|   |-- run_unified_tests.py           #    `--suite core|full`, `--installed-only` test runner
|   |-- prepare_cpp_reference.py       #    Regenerates C++ reference fixtures
|   |-- compare_cpp_python_soil_heating.py       # Scripted Lay* parity comparison driver
|   |-- unit/                          #    Golden-CSV + non-C++-live unit tests
|   |   |-- test_consumption_golden.py #    Golden-value regression tests for consumption equations
|   |   |                                    (split from the pre-Phase-1 test_equations_golden.py;
|   |   |                                    keeps the CSV-driven parametrized coverage)
|   |   |-- test_burnup_golden.py      #    Golden-value regression tests for burnup()
|   |   |-- test_equation_routing.py   #    Equation-ID output regression tests (was test_emission_equation_ids.py)
|   |   `-- test_run_unified_tests_contract.py  # Phase 1: installed-only parent/child contract
|   |-- integration/                   #    Full-pipeline (`run_fofem_emissions`) tests
|   |   |-- test_run_fofem_emissions.py         # Output-dict key/shape contract tests
|   |   |                                         (was test_run_fofem_emissions_output_keys.py)
|   |   `-- test_soil_heating_pipeline.py       # Invalid soil_family error-handling tests
|   |                                             (was test_soil_heating_invalid_soil_family.py)
|   |-- regression/                    #    Named historical-bug regression tests
|   |   |-- test_equations_golden_fixes.py      # Fix A-D classes (split from the pre-Phase-1
|   |   |                                         test_equations_golden.py)
|   |   `-- test_pr1_review_regressions.py
|   |-- cpp_parity_live/               #    Tests requiring the compiled C++ reference
|   |   |-- test_compare_cpp_python.py #    Python-vs-C++ multi-case parity assertions
|   |   |-- test_cpp_comparison.py     #    Python-vs-C++ parity vs. reference/fofem_cpp/load.txt, emis.txt
|   |   `-- test_soil_heating_cpp_parity.py     # Soil Lay* parity vs C++ soil.tmp
|   `-- test_data/
|       |-- test_inputs/
|       `-- _results/
|
|-- examples/
|   |-- emissions_batch.py             #    Batch/array usage driver, writes CSV output
|   `-- example_data/                  #    fofem_emissions_batch_test.csv
|
|-- docs/CODEBASE.md                   # <- This file
`-- README.md
```

---

### Current parity/testing additions

- `tests/cpp_parity_live/test_cpp_comparison.py` provides direct Python-vs-C++ parity assertions against `reference/fofem_cpp/load.txt` and `emis.txt`.
- `tests/cpp_parity_live/test_compare_cpp_python.py` runs scripted multi-case comparisons against the C++ CSV harness.
- `tests/cpp_parity_live/test_soil_heating_cpp_parity.py` and `tests/compare_cpp_python_soil_heating.py` validate soil `Lay*` parity vs C++ `soil.tmp`.
- `tests/run_unified_tests.py --suite core|full` is the current publish-oriented test runner (see `README.md`).
- `examples/emissions_batch.py` (not under `tests/`) is the current emissions batch/example driver.
- `reference/fofem_cpp/FOF_UNIX/test_harness.cpp` is the parameterized C++ CSV harness (`fofem_test`).

> Note: as of this review, `MISSING_COMPONENTS.md` no longer exists in the repo root, and several test filenames previously documented here (`example_fofem_emissions_batch.py`, `compare_cpp_python.py`, `test_soil_cpp_parity.py`, `compare_cpp_python_soil.py`) have been renamed or moved — the listing above reflects the actual current filenames, verified 2026-08-26.

## Architecture Overview

### Python Library (`src/pyfofem/`)

The library is organised as a top-level orchestrator module (`pyfofem.py`)
plus multiple specialized modules under `components/`. Every public function accepts
both scalar and NumPy array inputs (internally converting to arrays and
converting back via `_is_scalar` / `_maybe_scalar`).

**Two re-export hops:** a component function reaches the package's public
surface via `components/__init__.py` → `pyfofem.py` (re-imported "for
backward compatibility," per its own top-of-file comment) → top-level
`__init__.py`. All three layers must be kept in sync when adding a new
public symbol.

| Layer | Files | Responsibility |
|-------|-------|----------------|
| **Public API** | `__init__.py` | Re-exports all public symbols from `pyfofem.py` and `components/` |
| **Core Orchestrator** | `pyfofem.py` | High-level facades (`run_fofem_mortality`, `run_fofem_emissions`) and pipeline wiring |
| **Emissions Pipeline Helpers** | `components/emission_pipeline.py` | Pure computation stages extracted from `run_fofem_emissions()`: `compute_pre_burnup_consumption()`, `initialize_burnup_outputs()`, `compute_equation_arrays()`, `build_emissions_result()` |
| **Shared Helpers** | `components/_component_helpers.py` | Cross-cutting scalar/array plumbing (`_is_scalar`, `_maybe_scalar`, `_to_str_arr`) used by multiple component modules |
| **Burnup Engine** | `components/burnup.py` | Albini & Reinhardt post-frontal combustion simulation (ported from C++) |
| **Burnup Facade/Adapters** | `components/burnup_calcs.py` | `run_burnup`, cell workers, summary extraction, class ordering/mapping |
| **Consumption Equations** | `components/consumption_calcs.py` | Litter/duff/herb/shrub/canopy/mineral-soil equations and carbon |
| **Emissions** | `components/emission_calcs.py` | `legacy` / `default` / `expanded` emissions modes and EF CSV loading |
| **Mortality** | `components/mortality_calcs.py` | `mort_crnsch`, `mort_bolchar`, `mort_crcabe` |
| **Tree/Flame Utilities** | `components/tree_flame_calcs.py` | Scorch/flame/char/canopy helper calculations |
| **Soil Heating** | `components/soil_heating.py` | Campbell (1D equilibrium) and Massman HMV (non-equilibrium) models using `scipy.integrate.solve_ivp` |
| **Data** | `supporting_data/` | Species lookup CSV, emission factor CSV, bundled FOFEM 6.7 files |

### C++ Reference (`reference/fofem_cpp/`)

The C++ codebase follows a **manager-pattern** with struct-in / struct-out
interfaces.  Each subsystem has:
- An **input struct** (`d_CI`, `d_SI`, `d_MI`) with an `*_Init()` function
- An **output struct** (`d_CO`, `d_SO`, `d_MO`)
- A **manager function** (`CM_Mngr`, `SH_Mngr`, `MRT_CalcMort`)

Key build targets in `CMakeLists.txt`:
- `fofem` - standalone CLI executable (from `FOF_UNIX/`)
- `fofem_debug_c` - shared library
- `FOFEMd` - DLL with SWIG C# bindings
- `fofem_test` - parameterized C++ CSV harness for parity testing

| C++ Module | Key Files | Python Equivalent |
|------------|-----------|-------------------|
| Consume Manager | `fof_cm.cpp` | `run_fofem_emissions()` |
| HSF Manager (herb/shrub/fol/duff/mineral) | `fof_hsf.cpp` | `consm_herb()`, `consm_shrub()`, `consm_canopy()`, `consm_duff()`, `consm_mineral_soil()`, `consm_litter()` |
| Burnup Consumed Manager | `fof_bcm.cpp` | `run_burnup()` + `_extract_burnup_consumption()` |
| Burnup Engine | `bur_brn.cpp` / `burnupw.cpp` | `components/burnup.py -> burnup()` |
| Burn Output Vectors | `bur_bov.cpp` | `_extract_burnup_consumption()` |
| Smoke Emissions | `bur_brn.cpp` (ES_* functions) | `calc_smoke_emissions()` |
| New Emission System | `fof_nes.cpp` | `calc_smoke_emissions(mode='expanded')` |
| Soil Heating (Campbell) | `fof_sh.cpp`, `fof_sha.cpp` | `soil_heat_campbell()` |
| Soil Heating (Massman HMV) | `FOF_DLL/HMV_Model.cpp`, `SolveHMV.cpp`, `CrankNicolson.cpp`, etc. | `soil_heat_massman()` |
| Tree Mortality | `fof_mrt.cpp` | `mort_crnsch()`, `mort_crcabe()`, `mort_bolchar()` |
| Display / I/O | `fof_disp.cpp` | N/A (Python returns dicts/DataFrames) |
| Cover-type Lookup | `CVT_SAF.cpp`, `CVT_NVCS.cpp`, `CVT_FCCS.cpp` |  Not ported |
| Batch Processing | `FOF_GUI/Bat_Mai.cpp`, `BAT_*.cpp` |  Not ported |

---

## Data Flow

### C++ FOFEM Pipeline (official)

```mermaid
flowchart TD
    CI["d_CI  Consume Inputs\n(fuel loads T/ac, moistures %,\nregion, season, cover group,\nburnup params)"]

    CM["CM_Mngr()\nConsume Manager\n(fof_cm.cpp)"]

    HSF["HSF_Mngr()\nHerb, Shrub, Foliage,\nBranch, Duff, Mineral Soil\n(fof_hsf.cpp)"]

    BCM["BCM_Mngr()\nBurnup Consumed Manager\n(fof_bcm.cpp)"]

    BSET["BCM_SetInputs()\nT/ac  kg/m, %  fraction,\nmoisture adjustments,\nBRN_SetFuel per class"]

    BRN["BRN_Run()\nAlbini & Reinhardt\nBurnup Engine\n(bur_brn.cpp)"]

    BOV["BOV functions\nExtract per-class\nconsumption\n(bur_bov.cpp)"]

    ES["ES_* functions\nEmission accumulators\nFlaming / Smoldering / Duff\ng/m  lb/ac"]

    CO["d_CO  Consume Outputs\n(Pre/Con/Pos per class T/ac,\nemissions lb/ac,\nfr_SFI[] intensity time-series)"]

    SI["d_SI  Soil Inputs\n(duff depth pre/post,\nsoil moisture, soil type)"]
    SH["SH_Mngr()\nSoil Heating\n(fof_sh.cpp / fof_sha.cpp)"]
    SO["d_SO  Soil Outputs\n(temp  depth  time)"]

    MI["d_MI  Mortality Inputs\n(species, DBH, flame/scorch,\nheight, crown ratio)"]
    MRT["MRT_CalcMort()\nTree Mortality\n(fof_mrt.cpp)"]
    MO["d_MO  Mortality Outputs\n(P(mort), killed, basal area)"]

    CI --> CM
    CM --> HSF
    HSF -->|"herb/shrub/fol/bra consumed\nduff % consumed"| CM
    CM -->|"f_HSFCon, f_PerDufCon"| BCM
    BCM --> BSET
    BSET --> BRN
    BRN --> BOV
    BRN --> ES
    BOV -->|"per-class consumed T/ac"| CO
    ES -->|"emissions lb/ac,\nFlaCon/SmoCon T/ac,\nFlaDur/SmoDur sec"| CO
    HSF -->|"herb/shrub/fol/bra results"| CO
    CM --> CO

    CO -->|"fr_SFI[] intensity kW/m\nduff depth pre/post"| SI
    SI --> SH
    SH --> SO

    MI --> MRT
    MRT --> MO

    style CI fill:#e1f5fe
    style CO fill:#e8f5e9
    style SO fill:#fff3e0
    style MO fill:#fce4ec
```

### Python pyfofem Pipeline

```mermaid
flowchart TD
    USER["User / DataFrame row(s)\n(scalar or equal-length arrays)"]

    RFE["run_fofem_emissions()\n(pyfofem.py orchestrator)"]

    PBC["compute_pre_burnup_consumption()\n(components/emission_pipeline.py)"]
    CL["consm_litter()"]
    CD["consm_duff()"]
    CH["consm_herb()"]
    CS["consm_shrub()"]
    CC["consm_canopy()"]
    CM["consm_mineral_soil()"]

    IBO["initialize_burnup_outputs()\n(simplified per-cell defaults,\noverwritten if burnup succeeds)"]

    CELLS["Per-cell kwargs (1 dict per row)"]
    POOL{{"num_workers == 1 ?\nsequential loop : ProcessPoolExecutor"}}
    RBC["_run_burnup_cell()\n(components/burnup_calcs.py)"]
    BE["burnup()\n(components/burnup.py)"]
    EX["_extract_burnup_consumption()\n+ _burnup_durations()"]
    ERRC{{"BurnupValidationError\nor other exception?"}}
    ECODE["Translated to numeric\nBurnupError / BurnupLimitAdj code\n(message-substring match)"]
    MERGE["Merge per-cell results back\ninto output arrays;\nBurnupError != 0 rows are\nzeroed and use IBO defaults"]

    SHC["soil_heat_campbell()\n(per cell, 'duff' or 'non_duff' model,\ndriven by burnup fr_SFI-equivalent time series)"]
    SHM["soil_heat_massman()\n(NOT called by run_fofem_emissions\n— separate user call only)"]
    SOUT["Lay0/Lay2/Lay4/Lay6\nLay60d/Lay275d\n(NaN unless soil_heating enabled)"]

    EQA["compute_equation_arrays()\n(components/emission_pipeline.py)"]
    CSE["calc_smoke_emissions()\n(legacy / default / expanded)"]
    BER["build_emissions_result()\n(components/emission_pipeline.py)"]

    OUT["dict with CONSUMPTION_VARS keys\n(Pre/Con/Pos per class,\nemissions, durations, Lay*)"]

    MB["mort_bolchar()"]
    MC["mort_crnsch()"]
    MK["mort_crcabe()"]
    MOUT["float / ndarray\nP(mortality)"]

    USER --> RFE
    RFE --> PBC
    PBC --> CL & CD & CH & CS & CC & CM
    RFE --> IBO
    RFE -->|"when use_burnup=True"| CELLS
    CELLS --> POOL
    POOL --> RBC
    RBC --> BE --> EX
    RBC --> ERRC
    ERRC -->|"yes"| ECODE
    EX -->|"per-class consumed,\nflaming/smoldering,\ndurations"| MERGE
    ECODE --> MERGE
    IBO --> MERGE
    MERGE -->|"when soil_heating enabled\n(non-errored cells only)"| SHC
    PBC --> EQA
    MERGE --> CSE
    SHC --> BER
    EQA --> BER
    CSE --> BER
    MERGE --> BER
    BER --> OUT

    USER -->|"optional separate call"| SHM
    SHC & SHM --> SOUT

    USER -->|"separate call"| MB & MC & MK
    MB & MC & MK --> MOUT

    style RFE fill:#e1f5fe
    style OUT fill:#e8f5e9
    style SOUT fill:#fff3e0
    style MOUT fill:#fce4ec
```

**Diagram notes (verified against `pyfofem.py` and `components/emission_pipeline.py` 2026-08-26):**
- `run_fofem_emissions()` no longer calls the six `consm_*` functions directly — that's done inside `compute_pre_burnup_consumption()`, one of four pipeline-stage helpers extracted from the orchestrator into `emission_pipeline.py`.
- Per-cell burnup dispatch is parallelizable: `num_workers == 1` runs a plain Python loop over `_run_burnup_cell()`; `num_workers > 1` dispatches the same function across a `concurrent.futures.ProcessPoolExecutor`, both wrapped in a `tqdm` progress bar when `show_progress=True`.
- A cell whose burnup run raises `BurnupValidationError` (or any other exception) never reaches `EX` — `_run_burnup_cell()` catches it and returns a numeric `BurnupError` code instead (see Gotcha below). Cells with `BurnupError != 0` have **all** of their per-cell consumption/duration outputs hard-zeroed before final assembly, not just the burnup-derived ones.
- `soil_heat_massman()` is fully implemented but is **not** wired into `run_fofem_emissions()` — only `soil_heat_campbell()` is. The `Lay*` outputs in `OUT` always come from Campbell.

---

## Key C++ Files (`FOF_UNIX/`) and Responsibilities

### Entry Points and Managers

| File | Function(s) | Purpose |
|------|-------------|---------|
| `ansi_mai.cpp` | `main()`, `ConEmiSoi()` | CLI entry point; sample code demonstrating the full pipeline |
| `fof_cm.cpp` | `CM_Mngr()` | **Master orchestrator**  calls `HSF_Mngr` then `BCM_Mngr`, sums totals |
| `fof_hsf.cpp` | `HSF_Mngr()`, `Calc_Herb()`, `Calc_Shrub()`, `Calc_CrownFoliage()`, `Calc_CrownBranch()` | Non-burnup fuel consumption (herb, shrub, foliage, branch, duff, mineral soil) |
| `fof_bcm.cpp` | `BCM_Mngr()`, `BCM_SetInputs()`, `BCM_DW10M_Adj()`, `BCM_DW1k_MoiRot()` | Converts T/ackg/m, applies moisture adjustments, feeds fuel to burnup, extracts results |

### Burnup Engine

| File | Function(s) | Purpose |
|------|-------------|---------|
| `bur_brn.cpp` | `BRN_Init()`, `BRN_SetFuel()`, `BRN_SetFireDat()`, `BRN_Run()`, `BRN_CheckData()` | FOFEM's wrapper around the Albini/Reinhardt burnup simulation; also hosts emission accumulators (`ES_*`) |
| `bur_bov.cpp` | `BOV_Init()`, `BOV_Entry()`, `BOV_Get()`, `BOV_Get3()` | Burn Output Vectors  maps burnup's sorted component indices back to named fuel classes (litter, DW1, DW10, DW100, DW1kSnd, DW1kRot by size) |

### Fuel Consumption Sub-Models

| File | Equations | Purpose |
|------|-----------|---------|
| `fof_duf.cpp` | Eqs 120 | Duff consumption and depth reduction |
| `fof_lem.cpp` | Eqs 997999 | Litter consumption (including SE and Pine Flatwoods) |
| `fof_sd.cpp` | Eq 10+ | Mineral soil exposure |
| `fof_hsf.cpp` | Eqs 22236 | Herb and shrub consumption (region/cover-group dispatch) |

### Emissions

| File | Purpose |
|------|---------|
| `bur_brn.cpp` (ES_* functions) | Default Ward et al. 1993 emission factors; accumulates flaming/smoldering/duff emissions in g/m |
| `fof_nes.cpp` | "New Emission System"  loads `Emission_Factors.csv`, provides per-group factors for 8 vegetation types |
| `fof_co.h` | `d_CO` output struct with `f_PM10F`, `f_PM25S`, etc. in **lb/acre** |

### Soil Heating

| File | Purpose |
|------|---------|
| `fof_sh.cpp`, `fof_sha.cpp` | Campbell 1D equilibrium model; receives `fr_SFI[]` intensity time-series from burnup |
| `FOF_DLL/HMV_Model.cpp`, `SolveHMV.cpp`, `CrankNicolson.cpp`, `cal*.cpp` (~50 files) | Full Massman non-equilibrium heat-moisture-vapor PDE solver. **Only in FOF_DLL**, not FOF_UNIX. |

### Mortality

| File | Purpose |
|------|---------|
| `fof_mrt.cpp` | Species-specific mortality equations; dispatches by species code to crown scorch, bole char, or crown volume models |
| `fof_iss.h` | Internal species struct (bark coefficients, equation codes) |

### Data Structures

| File | Struct | Fields | Purpose |
|------|--------|--------|---------|
| `fof_ci.h` | `d_CI` | ~60 fields | All consume inputs: fuel loads (T/ac), moistures (%), region, season, cover group, burnup parameters, emission factor settings |
| `fof_co.h` | `d_CO` | ~100 fields | All consume outputs: Pre/Con/Pos per class (T/ac), emissions (lb/ac), `fr_SFI[]` intensity array, FlaCon/SmoCon, durations |
| `fof_sh.h` / `fof_sh2.h` | `d_SI` / `d_SO` | ~30 fields | Soil heating input/output |
| `fof_mrt.h` | `d_MI` / `d_MO` | ~25 fields | Mortality input/output |
| `fof_sgv.h` | `d_SGV` | 6 fields | Per-timestep fire intensity record for soil heating |
| `bur_bov.h` | (internal) |  | Burn output vector index mapping |

---

## Unit Conventions

| Context | Loads | Depth | Moisture | Temperature | Emissions | Intensity |
|---------|-------|-------|----------|-------------|-----------|-----------|
| **C++ external API** | T/acre | inches | % (whole) | C | lb/acre | kW/m^2 |
| **C++ burnup internal** | kg/m^2 | meters | fraction | K |  | kW/m^2 |
| **Python `units='Imperial'`** | T/acre | inches | % (whole) | C | lb/acre | kW/m^2 |
| **Python `units='SI'`** | kg/m^2 | cm | % (whole) | C | g/m^2 | kW/m^2 |
| **Python burnup engine** | kg/m^2 | meters | fraction | K (internal) |  | kW/m^2 |

### Key C++ conversion functions (in `fof_util.cpp`)
- `TPA_To_KiSq()`  T/acre -> kg/m^2
- `KgSq_To_TPA()`  kg/m^2 -> T/acre
- `GramSqMt_To_Pounds()`  g/m^2 -> lb/acre

### Python constants (in `pyfofem.py`)
- `_TPAC_TO_KGPM2 = 1/4.4609`  T/acre -> kg/m^2
- `_KGPM2_TO_TPAC = 4.4609`  kg/m^2 -> T/acre
- `_IN_TO_CM = 2.54`  inches -> cm

---

## Implicit Assumptions and Gotchas

### 1. C++ moisture adjustments (historical gap now resolved)

The C++ `BCM_SetInputs()` applies moisture adjustments before feeding burnup:

| Fuel Class | C++ Adjustment | Python `run_fofem_emissions` |
|------------|----------------|------------------------------|
| 1-hr | `DW10_moisture - 0.02` | Uses `dw10_moist / 100 - 0.02` |
| 10-hr | `DW10_moisture` (as-is) | Uses `dw10_moist / 100` |
| 100-hr | `DW10_moisture + 0.02` | Uses `dw10_moist / 100 + 0.02` |
| 1000-hr sound | `DW1000_moisture / 100` | Same |
| 1000-hr rotten | `DW1000_moisture / 100 * 2.5` (capped at 3.0) | Same as C++ |

**RESOLVED:** The rotten moisture multiplier (`e_DW1000hr_AdjRot = 2.5`, capped at 3.0) and the +/-0.02 fine-fuel adjustments from `BCM_DW10M_Adj()` are now implemented in `run_fofem_emissions()`.

Current Python behavior now matches C++ for burnup-input moisture adjustments:
- 1-hr uses `dw10_moist/100 - 0.02`
- 10-hr uses `dw10_moist/100`
- 100-hr uses `dw10_moist/100 + 0.02`
- rotten 1000-hr uses `min((dw1000_moist/100) * 2.5, 3.0)`

### 2. C++ ensures at least one burnable fuel particle is present

`BCM_SetInputs()` injects `f_Load = 0.0000001` into 1-hr wood when needed so burnup has at least one fuel particle (notably duff-only scenarios). **RESOLVED:** Python mirrors this with `1e-7` kg/m^2 DW1 injection for duff-only/no-wood cases.

### 3. C++ litter handling: burnup always processes litter for emissions

Even when SouthEast or Pine Flatwoods equations compute litter consumption separately, the C++ still sends the consumed amount into burnup so it can calculate fire intensity and emissions from it (Note-2/3 in `BCM_Mngr`). The Python `run_fofem_emissions` sends the full pre-fire litter load into burnup and lets burnup consume it, then optionally overrides with the regional equation result.

### 4. `run_burnup()` returns a 3-tuple, not 2

Changed from `(results, summary)` to `(results, summary, class_order)` to
support mapping burnup's sorted component indices back to named fuel classes.
**External callers must unpack all three.**

### 5. Burnup sorts fuel classes internally

The burnup engine sorts particles by decreasing SAV (increasing size), then
moisture, then density.  The `BurnSummaryRow` list and `BurnResult.comp_flaming`
/ `comp_smoldering` arrays follow this **sorted** order, not the input order.
`class_order` (returned by `run_burnup`) provides the mapping.

### 6. Rotten wood: C++ uses `BRN_SetFuel("ROT", ...)` for lower density

In the C++, `BRN_SetFuel` with the `"ROT"` flag applies `dendry = 300 kg/m^3`
(vs. 513 for `"SND"`).  The Python replicates this via `_DENSITY_ROTTEN = 300`
and `_DENSITY_SOUND = 513` in both `run_fofem_emissions` and `run_burnup`
when rotten classes are provided.

### 7. Duff moisture validation prevents burnup from running

The burnup engine validates duff moisture in the range 0.1-1.972 (10-197.2%).
High duff moisture (common in spring burns) causes `BurnupValidationError`,
at which point `run_fofem_emissions` falls back to simplified percentage
defaults with `warnings.warn()`.

### 8. `hfi` units ambiguity

The `run_fofem_emissions` docstring describes `hfi` as "Head fire intensity
(kW/m)" (Byram's fireline intensity, energy per metre of fire front), but the
burnup engine expects `fi` as "fire intensity (kW/m^2)" (area-based).  The C++
`d_CI.f_INTENSITY` comment says "kW/m2 sq m".  **The Python passes the value
through without conversion.**

### 9. `comp_flaming` / `comp_smoldering` are rates, not masses

`BurnResult.comp_flaming[i]` stores the mass-loss **rate** (kg/m^2/s)
accumulated during that recording interval.  To get consumed mass, multiply
by the timestep `dt`.  `_extract_burnup_consumption()` handles this.

### 10. Emissions mode selection matters for parity

In C++, emissions can be calculated with:
- legacy/original `ES_Calc` (combustion-efficiency factors, selected when `f_CriInt < 0`)
- expanded `ES_Calc_NEW` (separate flaming/coarse-smolder/duff EF groups).

Python now exposes this explicitly via `calc_smoke_emissions(mode=...)` and
`run_fofem_emissions(em_mode=...)`:
- `legacy` for C++ GUI/original parity
- `default` for single-group EF CSV mode
- `expanded` for split-group EF CSV mode

### 11. C++ emissions are in g/m, converted to lb/acre at output

All `ES_*` functions return g/m.  `BCM_Mngr` converts to lb/acre via
`GramSqMt_To_Pounds()`.  The Python `calc_smoke_emissions` can output
either lb/acre (`units='Imperial'`) or g/m (`units='SI'`).

### 12. Season strings are normalized to canonical labels in Python

C++ defines `"Summer"`, `"Spring"`, `"Winter"`, `"Fall"`. Python normalizes
input season strings to canonical title-case labels before equation routing.

### 13. Soil heating integration in Python

The C++ pipes `fr_SFI[]` (burnup intensity time-series) directly from `d_CO`
into `SH_Mngr`. Python now mirrors this path inside `run_fofem_emissions`
when `soil_heating` is enabled:

- `soil_family` is required (GUI/C++ aliases are normalized internally).
- Soil moisture is resolved from `soil_moisture`, `soil_heating['soil_moisture']`,
  `moisture_regime`, or a clipped `duff_moist` fallback.
- Duff vs non-duff routing follows the C++-style branch, and `Lay*` outputs are
  populated in the returned dict.

When `soil_heating=False`, `Lay*` outputs remain `NaN`.

### 14. C++ `cheat` upper limit is 3000, Python now matches

The C++ `bur_brn.h` changed the limit from 2000 to 3000.  ** RESOLVED: Python's `_FUEL_BOUNDS` now uses 3000 (matching C++).**

### 15. FlaDur/SmoDur units now aligned

C++ `d_CO.f_FlaDur` / `f_SmoDur` are in **seconds**.  ** RESOLVED: Python's `_burnup_durations()` and `run_fofem_emissions()` now return durations in seconds.**

### 16. A burnup error zeroes *all* per-cell consumption outputs, not just burnup-derived ones

When a cell's `BurnupError != 0`, `run_fofem_emissions()`'s step 5b sets an
explicit list of ~30 per-cell arrays to `0.0` — including litter, herb,
shrub, foliage, branch, duff, and mineral-soil-exposure outputs that were
computed independently of burnup in step 4. This is broader than "burnup
didn't run, so skip burnup outputs": a duff-consumption result that
`consm_duff()` computed successfully is still discarded for that row if
burnup separately failed. Confirm this is the intended contract before
relying on non-burnup outputs from a row with a nonzero `BurnupError`.

### 17. `_run_burnup_cell()` maps exceptions to numeric codes by matching substrings in the exception message

`components/burnup_calcs.py`'s per-cell worker catches `BurnupValidationError`
and assigns a `BurnupError` code by checking whether specific substrings
(`'cannot dry fuel'`, `'no fuel ignited'`, `'duff moisture'`, `'sav'`, etc.)
appear in `str(exc).lower()`, with a hardcoded `_FUEL_ATTR_TO_CODE` dict for
the fuel-property checks. There is no structural link (e.g. an error-code
attribute on `BurnupValidationError`) between the message text raised in
`burnup.py`/`_check_fuel()`/`_check_fire()` and this matching table — editing
a raised message string in one place without updating the other can silently
misclassify (or fail to classify) a failure as `BurnupError=99` ("unexpected
burnup exception").

### 18. `soil_heat_massman()` is not integrated into `run_fofem_emissions()`

Only `soil_heat_campbell()` is called from the orchestrator's per-cell
soil-heating branch (`model='duff'` or `model='non_duff'`, selected by
whether pre-fire duff depth is positive). `soil_heat_massman()` — the full
non-equilibrium heat-moisture-vapor PDE solver documented as a "Done"
feature — is only reachable via a direct, separate call. The `Lay*` keys in
`run_fofem_emissions()`'s output dict always originate from Campbell,
regardless of which model a caller might assume from the README's mention of
both models.

### 19. `_to_str_arr()` duplication — Fixed 2026-08-26

**RESOLVED** (PR #1 Copilot review): `components/consumption_calcs.py`
imported the shared `_to_str_arr()` from `_component_helpers.py` but then
redefined it locally, shadowing the import — the local copy was
byte-for-byte identical logic, so the import was dead. Removed the local
duplicate; the module now uses the shared helper it already imported.

### 20. `calc_smoke_emissions()` return type annotation — Fixed 2026-08-26

**RESOLVED** (PR #1 Copilot review): Was annotated
`-> Dict[str, float]`, but every mode (legacy/default/expanded) always
returns `np.ndarray` values — the internal `_total()` helper coerces every
input through `np.asarray()`, so even scalar calls produce 0-d/1-element
arrays, never plain Python floats. Corrected to `Dict[str, np.ndarray]`.

### 21. `np.atleast_1d` doesn't flatten 2D+ input — Fixed 2026-08-26

**RESOLVED** (PR #1 Copilot review + deeper sweep): `np.atleast_1d()`
leaves already-≥1D input unchanged, including 2D+ arrays — unlike
`np.ravel()`, which always flattens to 1D. All three `mort_*` functions in
`mortality_calcs.py`, all of `consumption_calcs.py`'s input coercion, and
`pyfofem.py`'s `run_fofem_emissions()` broadcast step used
`np.atleast_1d(np.asarray(...))` (54 occurrences total), which meant a 2D
input produced a 2D boolean mask indexed against a 1D output array —
reproduced directly as `IndexError: too many indices for array: array is
1-dimensional, but 2 were indexed`. Swapped every occurrence to
`np.ravel()`, which is behavior-identical for scalar/1D input (the only
shapes any test or documented usage exercises) and only changes the
previously-broken 2D+ case. Regression coverage:
`tests/regression/test_pr1_review_regressions.py`.

### 22. `_FIRE_BOUNDS['fistart']` minimum didn't match C++ — Fixed 2026-08-26

**RESOLVED** (PR #1 Copilot review): Was `10.0`, contradicting its own
inline comment, the `_check_fire()` docstring's C++ bounds table, and
`_BURNUP_LIMIT_ERROR[10]`'s description, all three of which already said
`40.0`. Verified directly against the compiled C++ source
(`reference/fofem_cpp/FOF_UNIX/bur_brn.cpp:1144`,
`const double fir1 = 40.0`) rather than trusting Python's own internal
docs, since all three could in principle have inherited the same original
mistake. Fixed to `40.0`. No test used a value in the 10–40 kW/m² range
that this affects. Regression coverage:
`tests/regression/test_pr1_review_regressions.py::test_fistart_min_matches_cpp_reference`.

### 23. `_check_fire()` is dead code — three different, inconsistent bounds-handling paths exist

Found while fixing #22. `burnup()`'s `validate=True` path only calls
`_check_fuel()` (fuel-particle bounds); `_check_fire()` (fire-environment
bounds: `fistart`, `ti`, `u`, `d`, `tamb_c`, `dfm`), which has clean
raise-with-message semantics for every bound, is fully defined but never
invoked anywhere in production code or tests. Instead, two *different*,
ad hoc implementations exist, neither of which calls `_check_fire()`:

- `_run_burnup_cell()` (the actual worker `run_fofem_emissions()` uses,
  via `ProcessPoolExecutor`) — asymmetric per bound: values exceeding the
  *upper* limit are clipped (recording a `burnup_limit_adjust` code,
  1-6), values below the *lower* limit are rejected outright (returned as
  a numeric `burnup_error` code, 10-14) rather than clipped or raised as
  an exception — except `dfm` (min/max inverted: clipped low, rejected
  high) and `d`/fuel-bed-depth (clipped on *both* sides, no rejection
  path at all).
- `gen_burnup_in_file()` (a separate, standalone `.brn`-file-writing
  utility, not used by `run_fofem_emissions()`) — clips *both* sides
  unconditionally for every bound (`max(lo, min(x, hi))`), no error
  codes, no rejection path for anything.

So the same conceptual "is this fire-environment input valid" question
currently has three different, disagreeing answers depending on which of
the three code paths is asked. Left open pending explicit decision on
whether/how to consolidate these — potentially wiring `_check_fire()` in
as the single source of truth is a real behavior change for at least
`_run_burnup_cell()`'s current lower-bound-rejection cases (previously
returned a `burnup_error` code, would instead raise
`BurnupValidationError`).

### 24. No `_FIRE_BOUNDS` entry for C++'s duff-loading bounds

Found while fixing #22. C++'s `BRN_CheckData()` also validates duff
dry-weight loading (`wdf`) against `e_wdf1 = 0.022`, `e_wdf2 = 80.0`
kg/m² (`bur_brn.h`), but Python's `_FIRE_BOUNDS` has no `wdf` entry at
all — `_check_fire()`'s `wdf_load` parameter is only used to gate the
`dfm` (duff moisture) check, never validated against its own magnitude.
Left open pending explicit decision, and coupled to #23 since
`_check_fire()` isn't currently called regardless.

---

## Mapping: Python CONSUMPTION_VARS  C++ d_CO Fields

| Python Key | C++ `d_CO` Field | Units (Imperial) |
|------------|------------------|------------------|
| `LitPre` / `LitCon` / `LitPos` | `f_LitPre` / `f_LitCon` / `f_LitPos` | T/acre |
| `DW1Pre` / `DW1Con` / `DW1Pos` | `f_DW1Pre` / `f_DW1Con` / `f_DW1Pos` | T/acre |
| `DW10Pre` / `DW10Con` / `DW10Pos` | `f_DW10Pre` / `f_DW10Con` / `f_DW10Pos` | T/acre |
| `DW100Pre` / `DW100Con` / `DW100Pos` | `f_DW100Pre` / `f_DW100Con` / `f_DW100Pos` | T/acre |
| `DW1kSndPre` / `DW1kSndCon` / `DW1kSndPos` | `f_Snd_DW1kPre` / `f_Snd_DW1kCon` / `f_Snd_DW1kPos` | T/acre |
| `DW1kRotPre` / `DW1kRotCon` / `DW1kRotPos` | `f_Rot_DW1kPre` / `f_Rot_DW1kCon` / `f_Rot_DW1kPos` | T/acre |
| `DufPre` / `DufCon` / `DufPos` | `f_DufPre` / `f_DufCon` / `f_DufPos` | T/acre |
| `DufDepPre` / `DufDepCon` / `DufDepPos` | `f_DufDepPre` / `f_DufDepCon` / `f_DufDepPos` | inches |
| `HerPre` / `HerCon` / `HerPos` | `f_HerPre` / `f_HerCon` / `f_HerPos` | T/acre |
| `ShrPre` / `ShrCon` / `ShrPos` | `f_ShrPre` / `f_ShrCon` / `f_ShrPos` | T/acre |
| `FolPre` / `FolCon` / `FolPos` | `f_FolPre` / `f_FolCon` / `f_FolPos` | T/acre |
| `BraPre` / `BraCon` / `BraPos` | `f_BraPre` / `f_BraCon` / `f_BraPos` | T/acre |
| `MSE` | `f_MSEPer` | % |
| `PM10F` / `PM10S` /  | `f_PM10F` / `f_PM10S` /  | lb/acre |
| `FlaDur` / `SmoDur` | `f_FlaDur` / `f_SmoDur` | sec |
| `FlaCon` / `SmoCon` | `f_FlaCon` / `f_SmoCon` | T/acre |

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Bark thickness |  Done | `calc_bark_thickness` |
| Scorch height / flame length / char height |  Done | `calc_scorch_ht`, `calc_flame_length`, `calc_char_ht` |
| Crown length / volume scorched |  Done | `calc_crown_length_vol_scorched` |
| Canopy cover |  Done | `calc_canopy_cover` |
| Carbon calculation |  Done | `calc_carbon` |
| Moisture regime lookup |  Done | `get_moisture_regime` |
| Litter consumption |  Done | `consm_litter` |
| Duff consumption |  Done | `consm_duff` |
| Herbaceous consumption |  Done | `consm_herb` |
| Shrub consumption |  Done | `consm_shrub` |
| Canopy consumption |  Done | `consm_canopy` |
| Mineral soil exposure |  Done | `consm_mineral_soil` |
| Burnup engine |  Done | `components/burnup.py`  verified against C++ `burnupw.cpp` |
| Burnup facade |  Done | `run_burnup` |
| Smoke emissions (legacy) |  Done | `calc_smoke_emissions(mode='legacy')` (C++ ES_Calc parity) |
| Smoke emissions (default) |  Done | `calc_smoke_emissions(mode='default')` |
| Smoke emissions (expanded) |  Done | `calc_smoke_emissions(mode='expanded')` |
| Master orchestrator |  Done | `run_fofem_emissions` integrates burnup for woody/duff and optional soil heating (`Lay*`) |
| Crown scorch mortality |  Done | `mort_crnsch` |
| Crown volume + cambium mortality |  Done | `mort_crcabe` |
| Bole char mortality |  Done | `mort_bolchar` |
| Soil heating  Campbell |  Done | `soil_heat_campbell` — the only model wired into `run_fofem_emissions()` |
| Soil heating  Massman HMV |  Done (standalone) | `soil_heat_massman` — implemented but not called by `run_fofem_emissions()`; see Gotcha #18 |
| Moisture adjustments (0.02, 2.5 rotten) |  Done | See `run_fofem_emissions()`  Gotcha #1 resolved |
| Zero-load guard (`1e-7` kg/m^2 in DW1) |  Done | See `run_fofem_emissions()`  Gotcha #2 resolved |
| Batch processing driver/example |  Done (example) | `examples/emissions_batch.py` performs array/batch runs and writes CSV outputs |
| C++ soil-heating parity checks |  Done | `tests/cpp_parity_live/test_soil_heating_cpp_parity.py` + `tests/compare_cpp_python_soil_heating.py` |
| Cover-type auto-lookup (SAF/NVCS/FCC) |  Not started | C++: `CVT_*.cpp` / `fof_fccs.csv` |
| Weight distribution (1000-hr  size classes) |  Not started | C++: `cr_WD` in `d_CI` |
| Duration units reconciliation (sec vs min) |  Done | `_burnup_durations()` and `run_fofem_emissions()` now return seconds  Gotcha #15 resolved |






