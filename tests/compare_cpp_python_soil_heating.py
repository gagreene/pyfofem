#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare pyfofem soil-heating outputs against C++ soil.tmp output.

This script uses the ansi_mai.cpp-style single-case inputs and compares:
  Lay0, Lay2, Lay4, Lay6, Lay60d, Lay275d

against values derived from `reference/fofem_cpp/soil.tmp`.
"""
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from pyfofem import run_fofem_emissions


SOIL_TMP = os.path.join(_REPO, "reference", "fofem_cpp", "soil.tmp")


def _parse_soil_tmp(path: str):
    rows = []
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            line = ln.strip()
            if not line:
                continue
            if not re.match(r"^\d", line):
                continue
            vals = [float(x) for x in line.split()]
            # expected: time + 14 temperatures (Surface + 1..13 cm)
            if len(vals) < 15:
                continue
            rows.append(vals[:15])
    if not rows:
        raise RuntimeError(f"No numeric soil rows found in: {path}")
    return rows


def _cpp_lay_values_from_soil_tmp(rows):
    # row format: [time_min, surf, 1cm, 2cm, ..., 13cm]
    max_layer = [0.0] * 14
    lay60d = -1
    lay275d = -1
    for row in rows:
        temps = row[1:15]
        for i, t in enumerate(temps):
            if t > max_layer[i]:
                max_layer[i] = t
            if t > 60.0:
                lay60d = i
            if t > 275.0:
                lay275d = i

    # C++ stores max temps in int ir_Temp[], so truncate to int
    ir_temp = [int(v) for v in max_layer]
    return {
        "Lay0": float(ir_temp[0]),
        "Lay2": float(ir_temp[2]),
        "Lay4": float(ir_temp[4]),
        "Lay6": float(ir_temp[6]),
        "Lay60d": float(lay60d),
        "Lay275d": float(lay275d),
    }


def _run_python_case():
    # ansi_mai-style case + soil-heating args
    return run_fofem_emissions(
        litter=1.0,
        duff=1.0,
        duff_depth=1.0,
        herb=1.0,
        shrub=1.0,
        crown_foliage=1.0,
        crown_branch=1.0,
        pct_crown_burned=50.0,
        region="InteriorWest",
        season="Summer",
        fuel_category="Natural",
        duff_moist=10.0,
        dw10_moist=20.0,
        dw1000_moist=20.0,
        dw1=1.0,
        dw10=1.0,
        dw100=1.0,
        dw3_6s=0.125,
        dw6_9s=0.125,
        dw9_20s=0.125,
        dw20s=0.125,
        dw3_6r=0.125,
        dw6_9r=0.125,
        dw9_20r=0.125,
        dw20r=0.125,
        hfi=50.0,
        flame_res_time=60.0,
        fuel_bed_depth=0.3,
        ambient_temp=27.0,
        windspeed=0.0,
        use_burnup=True,
        units="Imperial",
        soil_family="Fine-Silt",
        soil_heating={
            "soil_moisture": 15.0,
            "start_temp": 21.0,
            "efficiency_wl": 0.15,
            "efficiency_hs": 0.10,
            "efficiency_duff": 1.0,
        },
    )


def main() -> int:
    cpp_rows = _parse_soil_tmp(SOIL_TMP)
    cpp_vals = _cpp_lay_values_from_soil_tmp(cpp_rows)
    py = _run_python_case()

    keys = ["Lay0", "Lay2", "Lay4", "Lay6", "Lay60d", "Lay275d"]
    tols = {"Lay0": 5.0, "Lay2": 5.0, "Lay4": 5.0, "Lay6": 5.0, "Lay60d": 1.0, "Lay275d": 1.0}

    fails = []
    for k in keys:
        pv = float(py.get(k, float("nan")))
        cv = float(cpp_vals[k])
        d = abs(pv - cv)
        if d > tols[k]:
            fails.append((k, pv, cv, d, tols[k]))

    print("\nC++ vs Python soil Lay* comparison")
    for k in keys:
        print(f"{k:7s}  py={float(py.get(k, float('nan'))):8.3f}  cpp={cpp_vals[k]:8.3f}")

    if fails:
        print("\nFAILURES:")
        for k, pv, cv, d, tol in fails:
            print(f"{k}: py={pv:.3f} cpp={cv:.3f} diff={d:.3f} tol={tol:.3f}")
        return 1

    print("\nAll soil Lay* checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

