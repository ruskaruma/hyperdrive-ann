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

# Find pybind11 (optional)
find_package(pybind11 QUIET)

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

# Set CUDA properties on the library target
set_property(TARGET ${PROJECT_NAME}_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Enable CUDA device code linking
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CLI executable
add_executable(${PROJECT_NAME} src/cli.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib CUDA::cudart)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Data generator executable
add_executable(gen_data bench/gen_data.cpp)

# Test executables
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic ${PROJECT_NAME}_lib CUDA::cudart)
set_target_properties(test_basic PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Python bindings (if pybind11 is available)
if(pybind11_FOUND)
    pybind11_add_module(hyperdrive_ann src/python_bindings.cpp)
    target_link_libraries(hyperdrive_ann PRIVATE ${PROJECT_NAME}_lib CUDA::cudart)
    set_target_properties(hyperdrive_ann PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Create directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bench)
