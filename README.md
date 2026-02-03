# pyfofem

**pyfofem** is a Python library for modeling fire effects on forest vegetation, based on the FOFEM (First Order Fire Effects Model) framework. It provides functions for estimating tree mortality, scorch height, flame length, bark thickness, and other fire-related metrics using published models and species-specific parameters.

## Directory Structure

- `src/pyfofem/pyfofem.py`  
  Core library code with fire effects modeling functions.
- `src/pyfofem/supporting_data/species_codes_lut.csv`  
  Lookup table for species codes.
- `src/pyfofem/reference_docs/`  
  Reference papers and documentation supporting the models.

## Features

- Vectorized calculations for fire intensity, scorch height, flame length, bark thickness, and crown scorch mortality.
- Species-specific mortality models based on FOFEM equations.
- Integration with lookup tables for species codes and parameters.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gagreene/pyfofem.git
cd pyfofem
pip install -r requirements.txt
```

## Usage
```python
from pyfofem.pyfofem import mort_crnsch, calc_scorch_ht, calc_flame_length

# Example: Estimate crown scorch mortality
Pm = mort_crnsch(
    spp=['PIPO', 'PSME'],
    dbh=[30, 25],
    ht=[20, 18],
    crown_depth=[8, 7],
    fire_intensity=[5000, 4000]
)
print(Pm)
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.