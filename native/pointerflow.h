#ifndef POINTERFLOW_H
#define POINTERFLOW_H

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

class PointerFlowBuffer {
public:
    PointerFlowBuffer(size_t byte_size);
    ~PointerFlowBuffer();

    uintptr_t host_ptr() const;
    uintptr_t device_ptr() const;
    size_t byte_size() const { return size_; }

    cudaStream_t stream() const { return stream_; }

    void synchronize() const;

    void print_first_floats(int count) const;

    void launch_dummy(int n_elements) const;

private:
    void* host_ptr_ = nullptr;
    void* device_ptr_ = nullptr;
    size_t size_ = 0;
    cudaStream_t stream_ = nullptr;
};

#endif