cmake_minimum_required(VERSION 3.5)

project(FOFEM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


# Find all CPP files in the src directory
file(GLOB CPP_FILES FOF_UNIX/*.cpp)

# Add the CPP files to the executable
add_executable(fofem ${CPP_FILES})

#copy of FOF_UNIX with all functions marked for export
file(GLOB CPP_LIB_FILES FOF_UNIX/*.cpp)

add_library(fofem_debug_c SHARED ${CPP_LIB_FILES})

file(GLOB SOURCES FOF_UNIX/*.cpp)
file(GLOB HEADERS FOF_UNIX/*.h)
add_library(FOFEMd
	SHARED
	${SOURCES}
	SWIG/fofem_csharp_wrap.cxx
	${HEADERS})
TARGET_LINK_LIBRARIES(FOFEMd)

# Test harness: reads CSV, runs CM_Mngr, writes CSV output
file(GLOB TEST_CPP_FILES FOF_UNIX/*.cpp)
list(FILTER TEST_CPP_FILES EXCLUDE REGEX "ansi_mai\\.cpp$")
add_executable(fofem_test
	FOF_UNIX/test_harness.cpp
	${TEST_CPP_FILES})
