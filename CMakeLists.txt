# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.20)

# Project name, version and language used
project(FastMath VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(BENCHMARK_ENABLE_TESTING OFF)

enable_testing()

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# Add subdirectories
add_subdirectory(lib/benchmark)
add_subdirectory(lib/googletest)
add_subdirectory(src/algorithms)
add_subdirectory(src/experiments/experiment_1)
add_subdirectory(src/utils)
add_subdirectory(tests/algorithm_tests)
add_subdirectory(tests/matrix_tests)