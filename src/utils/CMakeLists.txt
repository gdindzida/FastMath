add_library(utils_lib utils.cpp)

target_include_directories(utils_lib 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
# PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Optionally set compile options (warnings, optimizations)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(utils_lib PRIVATE -Wall -Wextra -pedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(utils_lib PRIVATE /W4)
endif()