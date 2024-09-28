add_executable(experiment_1 main.cpp)

target_link_libraries(experiment_1 PRIVATE alg_lib benchmark::benchmark)

target_include_directories(experiment_1 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)