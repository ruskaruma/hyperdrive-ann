cmake_minimum_required(VERSION 3.20)
project(hyperdrive-ann LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find CUB
find_path(CUB_INCLUDE_DIR cub/cub.cuh)
if(NOT CUB_INCLUDE_DIR)
    message(FATAL_ERROR "CUB not found. Please install CUB library.")
endif()
include_directories(${CUB_INCLUDE_DIR})

# Set CUDA properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(ANN_SOURCES
    src/ann.cu
    src/kernels.cu
    src/ann.hpp
)

# Create static library
add_library(${PROJECT_NAME}_lib STATIC ${ANN_SOURCES})
target_link_libraries(${PROJECT_NAME}_lib CUDA::cudart)

# CLI executable
add_executable(${PROJECT_NAME} src/cli.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib CUDA::cudart)

# Data generator executable
add_executable(gen_data bench/gen_data.cpp)

# Create directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bench)
