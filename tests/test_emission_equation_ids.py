import numpy as np

from pyfofem.components.emission_pipeline import compute_equation_arrays


def test_compute_equation_arrays_matches_cpp_routing():
    reg = np.array(
        [
            "SouthEast",
            "SouthEast",
            "InteriorWest",
            "NorthEast",
            "SouthEast",
            "PacificWest",
        ],
        dtype=object,
    )
    cvr = np.array(
        [
            "Pine Flatwoods",
            "Pocosin",
            "GrassGroup",
            "Sagebrush",
            "Shrub",
            "Ponderosa pine",
        ],
        dtype=object,
    )
    season = np.array(["Summer", "Fall", "Summer", "Fall", "Winter", "Spring"], dtype=object)
    fuel_type = np.array(["Natural", "Natural", "Slash", "Natural", "Natural", "Natural"], dtype=object)

    lit_eq, duf_con_eq, duf_red_eq, herb_eq, shrub_eq, mse_eq = compute_equation_arrays(
        reg,
        cvr,
        season,
        fuel_type,
    )

    assert np.array_equal(lit_eq, np.array([997, 998, 999, 999, 998, 999]))
    assert np.array_equal(duf_con_eq, np.array([16, 20, 2, 2, 16, 2]))
    assert np.array_equal(duf_red_eq, np.array([16, 20, 6, 6, 16, 6]))
    assert np.array_equal(herb_eq, np.array([223, 222, 22, 22, 222, 22]))
    assert np.array_equal(shrub_eq, np.array([236, 235, 23, 233, 231, 23]))
    assert np.array_equal(mse_eq, np.array([14, 202, 10, 10, 14, 10]))


def test_compute_equation_arrays_northeast_special_cases():
    reg = np.array(["NorthEast", "NorthEast"], dtype=object)
    cvr = np.array(["RedJacPin", "BalBRWSpr"], dtype=object)
    season = np.array(["Spring", "Summer"], dtype=object)
    fuel_type = np.array(["Natural", "Natural"], dtype=object)

    _, duf_con_eq, duf_red_eq, _, _, mse_eq = compute_equation_arrays(reg, cvr, season, fuel_type)

    assert np.array_equal(duf_con_eq, np.array([15, 15]))
    assert np.array_equal(duf_red_eq, np.array([15, 15]))
    assert np.array_equal(mse_eq, np.array([14, 14]))
