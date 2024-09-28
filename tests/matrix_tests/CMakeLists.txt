add_executable(matrix_tests matrix_tests.cpp)

target_link_libraries(matrix_tests PRIVATE gtest gtest_main)

target_include_directories(matrix_tests 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)

add_test(NAME MatrixTests COMMAND matrix_tests)