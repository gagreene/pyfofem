cmake_minimum_required(VERSION 3.5)

project(FOFEM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# test_harness.cpp is Phase 2 overlay-only test tooling, not upstream FOFEM
# source. It must never enter fofem/fofem_debug_c/FOFEMd (it has its own
# main() and would duplicate ansi_mai.cpp's), so every upstream glob below
# excludes it explicitly. fof_soi_instr.cpp (F-70 diagnostic-observer pass,
# see below) and bur_brn_instr.cpp (F-62 completion/acceptance-recovery
# pass, see further below) are likewise excluded from all three: each
# duplicates a real pinned file (soiltemp_step()/tcond()/etc., or
# FireIntensity()/_CompDump()/etc.) these targets already include, and is
# used ONLY by its own separate fofem_test_*diag target.

# Find all CPP files in the src directory
file(GLOB CPP_FILES FOF_UNIX/*.cpp)
list(FILTER CPP_FILES EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER CPP_FILES EXCLUDE REGEX "fof_soi_instr\\.cpp$")
list(FILTER CPP_FILES EXCLUDE REGEX "bur_brn_instr\\.cpp$")

# Add the CPP files to the executable
add_executable(fofem ${CPP_FILES})

#copy of FOF_UNIX with all functions marked for export
file(GLOB CPP_LIB_FILES FOF_UNIX/*.cpp)
list(FILTER CPP_LIB_FILES EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER CPP_LIB_FILES EXCLUDE REGEX "fof_soi_instr\\.cpp$")
list(FILTER CPP_LIB_FILES EXCLUDE REGEX "bur_brn_instr\\.cpp$")

add_library(fofem_debug_c SHARED ${CPP_LIB_FILES})

file(GLOB SOURCES FOF_UNIX/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "fof_soi_instr\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "bur_brn_instr\\.cpp$")
file(GLOB HEADERS FOF_UNIX/*.h)
add_library(FOFEMd
	SHARED
	${SOURCES}
	SWIG/fofem_csharp_wrap.cxx
	${HEADERS})
TARGET_LINK_LIBRARIES(FOFEMd)

# Test harness: reads CSV, drives the six Phase 2 modes, writes CSV output.
# TEST_CPP_FILES excludes both ansi_mai.cpp (upstream's own main()) and
# test_harness.cpp (added back exactly once, explicitly, below) so the
# harness's own translation unit is never listed twice in one target.
# fof_soi_instr.cpp (F-70 diagnostic-observer pass) and bur_brn_instr.cpp
# (F-62 completion/acceptance-recovery pass) are also excluded: this is
# the NORMAL target, which must keep linking only the real, unmodified,
# pinned fof_soi.cpp/bur_brn.cpp.
file(GLOB TEST_CPP_FILES FOF_UNIX/*.cpp)
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "ansi_mai\\.cpp$")
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "fof_soi_instr\\.cpp$")
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "bur_brn_instr\\.cpp$")
add_executable(fofem_test
	FOF_UNIX/test_harness.cpp
	${TEST_CPP_FILES})

# F-70 diagnostic-observer target (2026-09-18 pass): identical to
# fofem_test above EXCEPT the real, pinned fof_soi.cpp is swapped for the
# overlay's own fof_soi_instr.cpp -- a byte-for-byte copy of the pinned
# file plus exactly one diagnostic hook call (see that file's own header
# comment for the full provenance/diff proof). fof_soi_instr.cpp is also
# excluded from the NORMAL fofem_test's own glob above (it lives in the
# same FOF_UNIX/ directory as every other source) so fofem_test never
# links two definitions of soiltemp_step()/tcond()/etc. Never used to
# generate an accepted golden -- diagnostics-only, exactly like the
# separate ASan build_diag/ target this project already established.
file(GLOB TEST_CPP_FILES_SOIDIAG FOF_UNIX/*.cpp)
list(FILTER TEST_CPP_FILES_SOIDIAG EXCLUDE REGEX "ansi_mai\\.cpp$")
list(FILTER TEST_CPP_FILES_SOIDIAG EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER TEST_CPP_FILES_SOIDIAG EXCLUDE REGEX "fof_soi\\.cpp$")
list(FILTER TEST_CPP_FILES_SOIDIAG EXCLUDE REGEX "fof_soi_instr\\.cpp$")
list(FILTER TEST_CPP_FILES_SOIDIAG EXCLUDE REGEX "bur_brn_instr\\.cpp$")
add_executable(fofem_test_soidiag
	FOF_UNIX/test_harness.cpp
	FOF_UNIX/fof_soi_instr.cpp
	${TEST_CPP_FILES_SOIDIAG})

# F-62 completion/acceptance-recovery pass (2026-09-21) diagnostic-observer
# target: identical to fofem_test above EXCEPT the real, pinned
# bur_brn.cpp is swapped for the overlay's own bur_brn_instr.cpp -- a
# byte-for-byte copy of the pinned file plus exactly one gated-return
# change unlocking a PRE-EXISTING, already-authored-by-the-original-
# FOFEM-developers per-timestep/per-fuel-class flame-vs-smolder CSV dump
# (_CompDump(), previously permanently disabled by its own unconditional
# 'return;') behind the FOFEM_TEST_BURN_DIAG environment variable (see
# that file's own header comment for the full provenance/diff proof).
# bur_brn_instr.cpp is also excluded from the NORMAL fofem_test's own
# glob above so fofem_test never links two definitions of
# FireIntensity()/_CompDump()/etc. Never used to generate an accepted
# golden -- diagnostics-only, exactly like fofem_test_soidiag above.
file(GLOB TEST_CPP_FILES_BURNDIAG FOF_UNIX/*.cpp)
list(FILTER TEST_CPP_FILES_BURNDIAG EXCLUDE REGEX "ansi_mai\\.cpp$")
list(FILTER TEST_CPP_FILES_BURNDIAG EXCLUDE REGEX "test_harness\\.cpp$")
list(FILTER TEST_CPP_FILES_BURNDIAG EXCLUDE REGEX "bur_brn\\.cpp$")
list(FILTER TEST_CPP_FILES_BURNDIAG EXCLUDE REGEX "bur_brn_instr\\.cpp$")
list(FILTER TEST_CPP_FILES_BURNDIAG EXCLUDE REGEX "fof_soi_instr\\.cpp$")
add_executable(fofem_test_burndiag
	FOF_UNIX/test_harness.cpp
	FOF_UNIX/bur_brn_instr.cpp
	${TEST_CPP_FILES_BURNDIAG})
