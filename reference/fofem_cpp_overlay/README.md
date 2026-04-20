# FOFEM C++ Overlay

This directory separates local additions in `reference/fofem_cpp` from the
clean upstream repository checkout.

## Compared revisions

- Remote `fofem_wuinity.git` `HEAD`: `78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`
- Local `reference/fofem_cpp` `HEAD`: `7504e0ccf190015a1946c66d984261b7456d1cfe`

The local checkout is one commit ahead of `origin/master`.

## Comparison summary

- `530` files are identical between the clean remote clone and the local tree.
- `1` tracked file differs: `CMakeLists.txt`
- `39` files are local-only.

## Preserved source additions in this overlay

- `source/CMakeLists.txt`
- `source/FOF_UNIX/test_harness.cpp`
- `source/compile_test.bat`
- `source/FOFEM_CPP_CODEBASE.md`
- `patches/CMakeLists.remote_to_local.patch`

## Local-only generated artifacts not preserved here

These should be regenerated, not versioned as source:

- `build2/`
- `*.obj`
- `fofem_test.exe`
- `compile_log.txt`

## Suggested update workflow

1. Refresh upstream code into `reference/fofem_cpp` from `fofem_wuinity.git`.
2. Reapply the local overlay:
   - restore `FOF_UNIX/test_harness.cpp`
   - restore `compile_test.bat`
   - optionally restore `FOFEM_CPP_CODEBASE.md`
   - apply the `CMakeLists.txt` patch or copy `source/CMakeLists.txt`
3. Rebuild `fofem_test.exe` and any other generated outputs.

## Rebuild notes

The local `CMakeLists.txt` change adds a `fofem_test` executable target based
on `FOF_UNIX/test_harness.cpp`. The batch file `compile_test.bat` provides an
MSVC command-line build path for the harness as an alternative to CMake.
