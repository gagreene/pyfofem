# Conda Recipe
This directory contains the Conda recipe for building and testing `pyfofem`.

## Build

From the repository root:

```bash
conda build conda-recipe
```

If you use `boa`/`mambabuild`:

```bash
mamba mambabuild conda-recipe
```

## Test

The recipe runs the publish-safe test suite against the installed package:

```bash
python tests/run_unified_tests.py --suite core --installed-only
```

## Notes

- Runtime dependencies are defined in both `pyproject.toml` and `conda-recipe/meta.yaml`.
- `pyproject.toml` is the source of truth for Python packaging metadata.
- Update the version in both files together when cutting a release.
