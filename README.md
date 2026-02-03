# PointerFlow

**PointerFlow** is a high-performance library for Python designed to eliminate the data transfer bottleneck between CPU and GPU. By using **zero-copy memory mapping**, it allows processors to share a common physical memory space completely asynchronously. It is powered by C++ and custom-made CUDA kernels.

## The Edge of Low Latency

In applications where milliseconds also count, PointerFlow is faster than conventional frameworks like PyTorch in real-time data flow management applications.

- **Round-Trip Latency:** Reduced from **0.86 ms** (industry standard) to **0.14 ms**.
- **Dispatch Overhead:** Return of control to Python in **< 0.1 ms** using *non-blocking CUDA Streams*.
- **True Zero-Copy:** Complete elimination of `cudaMemcpy`. Data written to a NumPy view is instantly available to the GPU via the PCIe bus.

## Technical Architecture

PointerFlow combines the efficiency of **C++20** with the flexibility of **Python** through a low-level integration.

1. **Memory Management (RAII):** Automatic lifecycle management of *Pinned Memory* and *CUDA Streams* to prevent memory leaks in high-frequency executions.
2. **NumPy Integration:** Creating memory views using `py::capsule`, allowing Python to manipulate data in RAM that is read directly by the GPU.
3. **Asynchronous Execution:** Native support for CUDA kernels fired in independent streams, allowing real overlap of computation and business logic.