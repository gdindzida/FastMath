add_library(alg_lib gemm.cpp)

target_include_directories(alg_lib 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
# PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(alg_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(alg_lib PUBLIC _OPENMP)
endif()

# Optionally set compile options (warnings, optimizations)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(alg_lib PRIVATE -Wall -Wextra -pedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(alg_lib PRIVATE /W4)
endif()