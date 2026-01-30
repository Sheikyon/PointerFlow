#include <cuda_runtime.h>

__global__ void dummy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

extern "C" void launch_dummy(float* d_ptr, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dummy_kernel<<<blocks, threads>>>(d_ptr, n);
    cudaDeviceSynchronize();
}