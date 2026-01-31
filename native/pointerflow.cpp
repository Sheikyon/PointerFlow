#include "pointerflow.h"
#include <iostream>
#include <algorithm>

extern "C" void launch_dummy(float* d_ptr, int n, void* stream_ptr);

PointerFlowBuffer::PointerFlowBuffer(size_t byte_size) : size_(byte_size) {
    CUDA_CHECK(cudaHostAlloc(&host_ptr_, byte_size, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

PointerFlowBuffer::~PointerFlowBuffer() {
    if (stream_) cudaStreamDestroy(stream_);
    if (host_ptr_) cudaFreeHost(host_ptr_);
}

uintptr_t PointerFlowBuffer::host_ptr() const { return reinterpret_cast<uintptr_t>(host_ptr_); }
uintptr_t PointerFlowBuffer::device_ptr() const { return reinterpret_cast<uintptr_t>(device_ptr_); }

void PointerFlowBuffer::synchronize() const {
    if (stream_) CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void PointerFlowBuffer::print_first_floats(int count) const {
    if (!host_ptr_) return;
    
    synchronize();

    const float* data = static_cast<const float*>(host_ptr_);
    int max_print = std::min(count, static_cast<int>(size_ / sizeof(float)));
    std::cout << "First " << max_print << " floats from C++: ";
    for (int i = 0; i < max_print; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;
}

void PointerFlowBuffer::launch_dummy(int n_elements) const {
    if (!device_ptr_ || n_elements <= 0) return;
    if (static_cast<size_t>(n_elements) * sizeof(float) > size_) {
        throw std::runtime_error("Buffer too small for dummy kernel");
    }
    ::launch_dummy(static_cast<float*>(device_ptr_), n_elements, stream_);
}