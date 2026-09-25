# Third-Party Notices and Provenance

PyFOFEM's original Python source code is licensed under the MIT License in
[`LICENSE`](LICENSE).

## FOFEM reference implementation and runtime data

PyFOFEM was validated against the FOFEM C++ source snapshot pinned as Git
commit `78f97f093ee7d1c77b3cd2622b2bd7248036c1e4` in the
`reference/fofem_cpp` submodule. That reference source, its local build
overlay, and its build products are development assets; they are not included
in PyPI distributions.

The following FOFEM-derived runtime data are included in the package because
the public API requires them:

| Package resource | Purpose |
| --- | --- |
| `supporting_data/emissions_factors.csv` | Emission-factor lookup table. |
| `supporting_data/fofem_bark_thickness.csv` | Bark-thickness extraction used by mortality calculations. |
| `supporting_data/fofem_crnsch_eq1_bark.csv` | Equation-1 bark-thickness extraction. |
| `supporting_data/species_codes_lut.csv` | Species-code lookup table. |
| `supporting_data/FOFEM6.7/FOF_SPP.CSV` | Species table used for canopy-equation lookup. |

The project records the provenance and exact-byte expectations for these
resources in `tests/unit/test_runtime_data_resources.py`. The distribution
intentionally excludes the remainder of the historical FOFEM 6.7 application
directory, including executables, DLLs, help files, sample projects, and
reference documents.

FOFEM was developed by the U.S. Department of Agriculture Forest Service. The
project's distribution decision is that the bundled FOFEM material identified
above is work of the United States federal government and is not subject to
copyright protection under 17 U.S.C. § 105. This notice preserves attribution
and identifies the boundary of that decision; it does not claim that
third-party material outside the listed package resources is included or
licensed by PyFOFEM.

For model use and scientific context, consult the FOFEM documentation and
publications listed under `docs/reference/papers/fofem/` in the source
repository. Those documents are not included in release artifacts.
