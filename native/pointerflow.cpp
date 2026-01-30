#include "pointerflow.h"
#include <iostream>
#include <algorithm>

PointerFlowBuffer::PointerFlowBuffer(size_t byte_size) : size_(byte_size) {
    CUDA_CHECK(cudaHostAlloc(&host_ptr_, byte_size, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0));
}

PointerFlowBuffer::~PointerFlowBuffer() {
    if (host_ptr_) cudaFreeHost(host_ptr_);
}

uintptr_t PointerFlowBuffer::host_ptr() const { return reinterpret_cast<uintptr_t>(host_ptr_); }
uintptr_t PointerFlowBuffer::device_ptr() const { return reinterpret_cast<uintptr_t>(device_ptr_); }

void PointerFlowBuffer::print_first_floats(int count) const {
    if (!host_ptr_) return;
    const float* data = static_cast<const float*>(host_ptr_);
    int max_print = std::min(count, static_cast<int>(size_ / sizeof(float)));
    std::cout << "First " << max_print << " floats from C++: ";
    for (int i = 0; i < max_print; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;
}