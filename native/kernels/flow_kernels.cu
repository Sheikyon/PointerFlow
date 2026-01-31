#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void dummy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

extern "C" void launch_dummy(float* d_ptr, int n, void* stream_ptr) {
    if (n <= 0) return;

    cudaStream_t stream = stream_ptr ? static_cast<cudaStream_t>(stream_ptr) : 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    dummy_kernel<<<blocks, threads, 0, stream>>>(d_ptr, n);
}