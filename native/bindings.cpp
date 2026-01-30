#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pointerflow.h"

#ifdef _WIN32
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#else
    #include <sys/types.h>
#endif

namespace py = pybind11;

extern "C" void launch_dummy(float* d_ptr, int n);

PYBIND11_MODULE(_core, m) {
    py::class_<PointerFlowBuffer>(m, "PointerFlowBuffer")
        .def(py::init<size_t>())
        .def("host_ptr", &PointerFlowBuffer::host_ptr)
        .def("device_ptr", &PointerFlowBuffer::device_ptr)
        .def("byte_size", &PointerFlowBuffer::byte_size)
        .def("print_first_floats", &PointerFlowBuffer::print_first_floats)
        .def("launch_dummy", [](PointerFlowBuffer &self, int n) {
            launch_dummy(reinterpret_cast<float*>(self.device_ptr()), n);
        });

    m.def("as_numpy_float32", [](uintptr_t host_ptr, ssize_t num_elements) {
        float* ptr = reinterpret_cast<float*>(host_ptr);
        
        py::capsule base(ptr, [](void*) {}); 

        return py::array_t<float>(
            { num_elements }, 
            { sizeof(float) }, 
            ptr, 
            base
        );
    });
}