# FOFEM C++ Overlay

This directory separates local additions in `reference/fofem_cpp` from the
clean upstream repository checkout.

## Compared revisions

- Pinned upstream `fofem_wuinity.git` commit: `78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`
- Local `reference/fofem_cpp` `HEAD`: `78f97f093ee7d1c77b3cd2622b2bd7248036c1e4`

The local checkout's `HEAD` is an exact match for the pinned upstream commit
(verified 2026-08-28; corrects an earlier note here that claimed the local
checkout was one commit ahead of `origin/master` — that claim was stale and
did not reflect the actual checked-out SHA. See `development/plans/gate0/`
provenance findings P-1/P-2.). The overlay is applied on top of this exact
commit as an **uncommitted working-tree change** — `CMakeLists.txt` is
modified in place, and three files are added untracked. Nothing is
committed inside `reference/fofem_cpp`, and the checkout's pinned commit is
never advanced by applying the overlay.

## Comparison summary (working tree vs. pinned commit, 2026-08-28)

- `1` tracked file differs from the pinned commit: `CMakeLists.txt`
  (adds the `fofem_test` executable target; see the patch below).
- `3` files are local-only (untracked, not part of the pinned commit):
  `FOF_UNIX/test_harness.cpp`, `compile_test.bat`, `FOFEM_CPP_CODEBASE.md`.

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

**Run `tests/prepare_cpp_reference.py` whenever `reference/fofem_cpp` is
updated from the upstream C++ source.** It automates all three steps below
— `python tests/prepare_cpp_reference.py --refresh --build` refreshes the
checkout, reapplies this overlay, and rebuilds the harness in one call. Run
it (at minimum without `--refresh`, to reapply the overlay) any time the
checkout has been touched by something other than this workflow, since
nothing else keeps the overlay files in sync automatically.

1. Refresh upstream code into `reference/fofem_cpp` from `fofem_wuinity.git`
   (`--refresh`; skip this step to keep the pinned commit exactly as-is,
   which is the normal case).
2. Reapply the local overlay (`_apply_overlay()`, the default action):
   - restore `FOF_UNIX/test_harness.cpp`
   - restore `compile_test.bat`
   - optionally restore `FOFEM_CPP_CODEBASE.md`
   - copy `source/CMakeLists.txt` over the checkout's `CMakeLists.txt`
3. Rebuild `fofem_test.exe` and any other generated outputs (`--build`).

The `CMakeLists.remote_to_local.patch` file under `patches/` is a
**provenance record**, not something applied by the workflow above (the
workflow copies `source/CMakeLists.txt` directly). Regenerate it after any
`CMakeLists.txt` overlay change with **git's own `--output` flag**, not
shell redirection (`>`):

```
git -C reference/fofem_cpp diff --output=<absolute-path-to>/reference/fofem_cpp_overlay/patches/CMakeLists.remote_to_local.patch -- CMakeLists.txt
```

`--output=<path>` makes git write the file itself, in plain UTF-8, with no
BOM — this is encoding-safe and reproducible from **any shell**
(PowerShell, cmd.exe, bash), because the shell never performs the
redirection or touches the file's bytes. **Do not** use plain `>` shell
redirection to (re)create this file: an earlier version of it was saved as
UTF-16LE (PowerShell's `>` redirects to UTF-16LE with a BOM by default
unless `-Encoding` is passed explicitly) and was therefore unusable as a
plain-text patch/diff artifact. A POSIX-only `> file` from bash happens to
produce UTF-8 correctly, but is not a cross-platform-safe instruction to
hand to a Windows-primary workflow — `git diff --output=` is portable and
removes the ambiguity entirely.

**Verify after regenerating** (both checks must pass before treating the
patch as valid provenance):

1. No UTF-16 BOM / plain UTF-8: `file <path>` should report `ASCII text` or
   `UTF-8 text` (never `UTF-16`), or equivalently the first two bytes must
   not be `FF FE` / `FE FF`.
2. Applies cleanly against a fresh checkout of the pinned commit:
   ```
   git show <pinned-SHA>:CMakeLists.txt > /tmp/CMakeLists.txt   # or an
                                                                  # equivalent
                                                                  # clean copy
   git apply --check reference/fofem_cpp_overlay/patches/CMakeLists.remote_to_local.patch
   ```
   (run from a throwaway directory containing that clean `CMakeLists.txt`
   under git so `git apply --check` has something to apply against). Exit
   code `0` confirms the stored patch is valid against the pinned revision.

## Rebuild notes

The local `CMakeLists.txt` change adds a `fofem_test` executable target that
compiles `FOF_UNIX/test_harness.cpp` together with every other `FOF_UNIX/*.cpp`
file except `ansi_mai.cpp` (which defines `main()` for the unrelated `fofem`
target). Build only the `fofem_test` target
(`cmake --build <builddir> --target fofem_test`); the pre-existing `fofem`,
`fofem_debug_c`, and `FOFEMd` targets are not exercised by Phase 2 and are
not guaranteed to build cleanly in this checkout (see
`development/plans/gate0/` for scope). The batch file `compile_test.bat`
provides an MSVC command-line build path for the harness as an alternative
to CMake.

`CMAKE_CXX_STANDARD` is left at `11` (the overlay does not raise it): the
Phase 2 harness (`test_harness.cpp`) was written to compile cleanly under
C++11, so raising the standard for the whole file (which would also affect
the pinned `FOF_UNIX/*.cpp` sources compiled into the untouched `fofem`/
`fofem_debug_c`/`FOFEMd` targets) was unnecessary and deliberately avoided.
