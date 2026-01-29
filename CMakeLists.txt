add_library(pointerflow_native STATIC
    native/pointerflow.cpp
    native/kernels/strided_patch.cu
)

pybind11_add_module(_core
    native/bindings.cpp
)

target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native)