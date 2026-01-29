cmake_minimum_required(VERSION 3.18...3.30)
project(pointerflow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(pointerflow_native STATIC
    native/pointerflow.cpp
    native/kernels/strided_patch.cu
)

target_include_directories(pointerflow_native PUBLIC 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/native
)

pybind11_add_module(_core native/bindings.cpp)

target_link_libraries(_core PRIVATE 
    pointerflow_native 
    CUDA::cudart
)

install(TARGETS _core DESTINATION pointerflow)