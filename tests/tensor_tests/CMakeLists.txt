add_executable(tensor_tests tensor_tests.cpp)

target_link_libraries(tensor_tests PRIVATE gtest gtest_main)

target_include_directories(tensor_tests 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)

add_test(NAME TensorTests COMMAND tensor_tests)