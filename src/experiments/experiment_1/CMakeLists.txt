add_executable(experiment_1 main.cpp)

target_link_libraries(experiment_1 PRIVATE utils_lib alg_lib)

target_include_directories(experiment_1 
PUBLIC "${PROJECT_SOURCE_DIR}/include"
)