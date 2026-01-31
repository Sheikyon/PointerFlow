#include <cuda_runtime.h>

__global__ void placeholder_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

extern "C" void launch_placeholder(float* d_ptr, int n, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    placeholder_kernel<<<blocks, threads, 0, stream>>>(d_ptr, n);
}