cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES native)
project(GPU_BFS CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(NVSHMEM_LIB_DIR "/usr/lib/x86_64-linux-gnu/nvshmem/12")
set(NVSHMEM_INCLUDE_DIR "/usr/include/nvshmem_12")

find_library(
        NVSHMEM_LIBRARY
        NAMES nvshmem
        PATHS ${NVSHMEM_LIB_DIR}
        REQUIRED
)

include_directories(${NVSHMEM_INCLUDE_DIR}
        ../fast_matrix_market/include)

add_executable(GPU_BFS main.cu)

set_target_properties(GPU_BFS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        BUILD_RPATH ${NVSHMEM_LIB_DIR}
        INSTALL_RPATH ${NVSHMEM_LIB_DIR})

target_link_libraries(GPU_BFS PRIVATE ${NVSHMEM_LIBRARY})
