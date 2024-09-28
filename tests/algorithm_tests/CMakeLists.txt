add_executable(gemm_tests gemm_tests.cpp)

target_link_libraries(gemm_tests PRIVATE gtest gtest_main alg_lib)

target_include_directories(gemm_tests 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)

add_test(NAME GemmTests COMMAND gemm_tests)