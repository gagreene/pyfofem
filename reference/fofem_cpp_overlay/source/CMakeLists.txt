cmake_minimum_required(VERSION 3.5)

project(FOFEM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# test_harness.cpp is Phase 2 overlay-only test tooling, not upstream FOFEM
# source. It must never enter fofem/fofem_debug_c/FOFEMd (it has its own
# main() and would duplicate ansi_mai.cpp's), so every upstream glob below
# excludes it explicitly.

# Find all CPP files in the src directory
file(GLOB CPP_FILES FOF_UNIX/*.cpp)
list(FILTER CPP_FILES EXCLUDE REGEX "test_harness\\.cpp$")

# Add the CPP files to the executable
add_executable(fofem ${CPP_FILES})

#copy of FOF_UNIX with all functions marked for export
file(GLOB CPP_LIB_FILES FOF_UNIX/*.cpp)
list(FILTER CPP_LIB_FILES EXCLUDE REGEX "test_harness\\.cpp$")

add_library(fofem_debug_c SHARED ${CPP_LIB_FILES})

file(GLOB SOURCES FOF_UNIX/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "test_harness\\.cpp$")
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
file(GLOB TEST_CPP_FILES FOF_UNIX/*.cpp)
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "ansi_mai\\.cpp$")
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "test_harness\\.cpp$")
add_executable(fofem_test
	FOF_UNIX/test_harness.cpp
	${TEST_CPP_FILES})
