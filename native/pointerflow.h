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

    void print_first_floats(int count) const;

private:
    void* host_ptr_ = nullptr;
    void* device_ptr_ = nullptr;
    size_t size_ = 0;
};

#endif