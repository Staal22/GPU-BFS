cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES native)
project(GPU_BFS CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(GPU_BFS main.cu)

set_target_properties(GPU_BFS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
