#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pointerflow.h"

namespace py = pybind11;

extern "C" void launch_dummy(float* d_ptr, int n, void* stream_ptr);

PYBIND11_MODULE(_core, m) {
    py::class_<PointerFlowBuffer>(m, "PointerFlowBuffer")
        .def(py::init<size_t>(), py::arg("byte_size"))
        .def("host_ptr", &PointerFlowBuffer::host_ptr)
        .def("device_ptr", &PointerFlowBuffer::device_ptr)
        .def("byte_size", &PointerFlowBuffer::byte_size)
        .def("print_first_floats", &PointerFlowBuffer::print_first_floats, py::arg("count"))
        .def("synchronize", &PointerFlowBuffer::synchronize)
        .def("launch_dummy", [](PointerFlowBuffer& self, int n_elements) {
            ::launch_dummy(
                reinterpret_cast<float*>(self.device_ptr()),
                n_elements,
                static_cast<void*>(self.stream())
            );
        }, py::arg("n_elements"));

    m.def("as_numpy_float32", [](uintptr_t host_ptr, size_t num_elements) {
        float* ptr = reinterpret_cast<float*>(host_ptr);
        py::capsule base(ptr, [](void*) { /* no delete */ });

        return py::array_t<float>(
            {num_elements},
            {sizeof(float)},
            ptr,
            base
        );
    }, py::arg("host_ptr"), py::arg("num_elements"));
}