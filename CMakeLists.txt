cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES native)
add_executable(GPU_BFS main.cu)

if (EXISTS "${CMAKE_SOURCE_DIR}/local_config.cmake")
    include(local_config.cmake)
endif ()

project(GPU_BFS CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(../fast_matrix_market/include)

set_target_properties(GPU_BFS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
