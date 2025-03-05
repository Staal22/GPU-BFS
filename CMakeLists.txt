cmake_minimum_required(VERSION 3.28)

if (EXISTS "${CMAKE_SOURCE_DIR}/local_config.cmake")
    include(local_config.cmake)
endif ()

project(GPU_BFS CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

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
