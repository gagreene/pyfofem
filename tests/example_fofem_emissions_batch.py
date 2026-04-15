#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_emissions.py – Batch-process the full fofem_emissions_results.csv using
the array-aware run_fofem_emissions interface.

All fuel-loading, moisture, and fire-environment columns are passed as
numpy arrays in a single call so that the non-burnup sub-models (litter,
duff, herb, shrub, canopy, mineral-soil) are evaluated in one vectorised
pass.  The burnup model, which must run cell-by-cell, is dispatched in
parallel with os.cpu_count() - 1 workers.
"""
import os
import sys
import time

# Resolve paths relative to the project root (one level up from this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

import numpy as np
import pandas as pd

from pyfofem import run_fofem_emissions

def main():
    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(
        _SCRIPT_DIR, 'test_data', 'test_inputs', 'fofem_emissions_batch_test.csv'
    )
    df = pd.read_csv(csv_path, comment=None)#[:1000]
    df.columns = [c.lstrip('#') for c in df.columns]
    print(f'Loaded {len(df):,} rows from {csv_path}')

    # ------------------------------------------------------------------
    # Pre-process categorical columns
    # ------------------------------------------------------------------
    # Strip surrounding quotes / whitespace from CoverGroup
    df['CoverGroup'] = (
        df['CoverGroup']
        .astype(str)
        .str.strip('"')
        .str.strip()
    )

    # Use the 'Season' column (capital S); fall back to 'Summer' if missing/NaN
    season_col = 'Season' if 'Season' in df.columns else 'season'
    df[season_col] = df[season_col].fillna('Summer').astype(str).str.strip()

    # Replace any windspeed NaN with 0
    df['instand_ws_mps'] = pd.to_numeric(df['instand_ws_mps'], errors='coerce').fillna(0.0)

    # ------------------------------------------------------------------
    # Build numpy arrays for every input parameter
    # ------------------------------------------------------------------
    def _arr(col):
        return df[col].to_numpy(dtype=float)

    litter          = _arr('Litter')
    duff            = _arr('Duff')
    duff_moist      = _arr('DuffMoisture')
    duff_depth      = _arr('DuffDepth')
    herb            = _arr('Herbaceous')
    shrub           = _arr('Shrub')
    crown_foliage   = _arr('CrownFoliage')
    crown_branch    = _arr('CrownBranch')
    pct_crown_burned= _arr('PercentCrownBurned')
    dw10_moist      = _arr('wfl10HourMoisture')
    dw1000_moist    = _arr('wfl1000HourMoisture')
    dw1             = _arr('wfl1Hour')
    dw10            = _arr('wfl10Hour')
    dw100           = _arr('wfl100Hour')
    dw3_6s          = _arr('wfl3_6S')
    dw6_9s          = _arr('wfl6_9S')
    dw9_20s         = _arr('wfl9_20S')
    dw20s           = _arr('wfl20plusS')
    dw3_6r          = _arr('wfl3_6R')
    dw6_9r          = _arr('wfl6_9R')
    dw9_20r         = _arr('wfl9_20R')
    dw20r           = _arr('wfl20plusR')
    hfi             = _arr('hfi')
    flame_res_time  = _arr('flame_res_time_sec')
    fuel_bed_depth  = _arr('fuel_bed_depth_m')
    ambient_temp    = _arr('temp')
    windspeed       = _arr('instand_ws_mps')

    region          = df['Region'].to_numpy(dtype=object)
    cvr_grp         = df['CoverGroup'].to_numpy(dtype=object)
    season          = df[season_col].to_numpy(dtype=object)
    fuel_category   = df['FuelCategory'].to_numpy(dtype=object)

    # ------------------------------------------------------------------
    # Determine number of parallel workers
    # ------------------------------------------------------------------
    cpu_count = os.cpu_count() or 4
    num_workers = max(1, cpu_count - 1)
    print(f'Using {num_workers} worker(s) (cpu_count={cpu_count})')

    # ------------------------------------------------------------------
    # Single array call to run_fofem_emissions
    # ------------------------------------------------------------------
    print(f'Running run_fofem_emissions on {len(df):,} rows…')
    t0 = time.perf_counter()

    results = run_fofem_emissions(
        litter=litter,
        duff=duff,
        duff_depth=duff_depth,
        herb=herb,
        shrub=shrub,
        crown_foliage=crown_foliage,
        crown_branch=crown_branch,
        pct_crown_burned=pct_crown_burned,
        region=region,
        cvr_grp=cvr_grp,
        season=season,
        fuel_category=fuel_category,
        duff_moist=duff_moist,
        dw10_moist=dw10_moist,
        dw1000_moist=dw1000_moist,
        dw1=dw1,
        dw10=dw10,
        dw100=dw100,
        dw3_6s=dw3_6s,
        dw6_9s=dw6_9s,
        dw9_20s=dw9_20s,
        dw20s=dw20s,
        dw3_6r=dw3_6r,
        dw6_9r=dw6_9r,
        dw9_20r=dw9_20r,
        dw20r=dw20r,
        hfi=hfi,
        flame_res_time=flame_res_time,
        fuel_bed_depth=fuel_bed_depth,
        ambient_temp=ambient_temp,
        windspeed=windspeed,
        use_burnup=True,
        em_mode='default',
        units='Imperial',
        num_workers=num_workers,
        show_progress=True,
    )

    elapsed = time.perf_counter() - t0
    print(f'Completed in {elapsed:.1f}s ({elapsed / len(df) * 1000:.2f} ms/row)')

    # ------------------------------------------------------------------
    # Write output columns back into the dataframe
    # ------------------------------------------------------------------
    _INT_COLS = {'Lit-Equ', 'DufCon-Equ', 'DufRed-Equ', 'MSE-Equ',
                 'Herb-Equ', 'Shurb-Equ', 'BurnupLimitAdj', 'BurnupError'}
    for col, val in results.items():
        if col in _INT_COLS:
            df[col] = np.asarray(val, dtype=int)
        else:
            df[col] = np.asarray(val, dtype=float)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(_SCRIPT_DIR, 'test_data', '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fofem_emissions_batch_test_output.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df):,} rows to {out_path}')

    # ------------------------------------------------------------------
    # Quick summary stats
    # ------------------------------------------------------------------
    print('\n--- Summary (mean across all rows) ---')
    for col in ('LitCon', 'DW10Con', 'DW1kSndCon', 'DW1kRotCon', 'DufCon',
                'FlaCon', 'SmoCon', 'PM25F', 'PM25S'):
        if col in results:
            vals = np.asarray(results[col], dtype=float)
            finite = vals[np.isfinite(vals)]
            print(f'  {col:14s}: mean={finite.mean():.4f}  min={finite.min():.4f}'
                  f'  max={finite.max():.4f}  n_valid={len(finite):,}')

    # BurnupLimitAdj summary
    if 'BurnupLimitAdj' in results:
        adj = np.asarray(results['BurnupLimitAdj'], dtype=int)
        n_adj = int(np.count_nonzero(adj))
        print(f'\n  BurnupLimitAdj: {n_adj:,} of {len(adj):,} rows had burnup '
              f'input clipping ({100 * n_adj / len(adj):.1f}%)')
        if n_adj > 0:
            unique, counts = np.unique(adj[adj != 0], return_counts=True)
            for u, c in zip(unique, counts):
                print(f'    code {u}: {c:,} rows')

    # BurnupError summary
    if 'BurnupError' in results:
        err = np.asarray(results['BurnupError'], dtype=int)
        n_err = int(np.count_nonzero(err))
        print(f'\n  BurnupError: {n_err:,} of {len(err):,} rows had burnup '
              f'errors ({100 * n_err / len(err):.1f}%)')
        if n_err > 0:
            unique, counts = np.unique(err[err != 0], return_counts=True)
            for u, c in zip(unique, counts):
                print(f'    code {u}: {c:,} rows')

    print('\nDone.')


if __name__ == '__main__':
    main()

