add_executable(experiment_1 main.cpp)

target_link_libraries(experiment_1 PRIVATE alg_lib benchmark::benchmark)

if(OpenMP_CXX_FOUND)
    target_link_libraries(experiment_1 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(experiment_1 PUBLIC _OPENMP)
endif()

target_include_directories(experiment_1 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)