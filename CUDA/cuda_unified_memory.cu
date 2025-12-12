#include <iostream>
#include <cuda_runtime.h>

__global__ void incrementKernel(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] += 1;
}

int main() {
    const int N = 1 << 20; // 1 million elements
    const size_t size = N * sizeof(int);

    int *um_data;

    // -------------------------------
    // 1. Allocate Unified Memory
    // -------------------------------
    cudaMallocManaged(&um_data, size);
    std::cout << "Unified memory allocated.\n";

    // Initialize memory on CPU
    for (int i = 0; i < N; i++) um_data[i] = i;

    // -------------------------------
    // 2. Provide memory usage advice
    // -------------------------------
    // Advise that GPU 0 will mostly use this memory
    cudaMemAdvise(um_data, size, cudaMemAdviseSetPreferredLocation, 0);
    std::cout << "Memory advice set: preferred GPU location.\n";

    // -------------------------------
    // 3. Create a stream and prefetch memory
    // -------------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemPrefetchAsync(um_data, size, 0, stream); // Prefetch to GPU 0
    std::cout << "Memory prefetched to GPU.\n";

    // -------------------------------
    // 4. Launch kernel in stream
    // -------------------------------
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    incrementKernel<<<gridSize, blockSize, 0, stream>>>(um_data, N);

    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
    std::cout << "Kernel completed.\n";

    // -------------------------------
    // 5. Prefetch memory back to CPU
    // -------------------------------
    cudaMemPrefetchAsync(um_data, size, cudaCpuDeviceId, stream);
    cudaStreamSynchronize(stream);
    std::cout << "Memory prefetched back to CPU.\n";

    // Verify
    std::cout << "um_data[0] = " << um_data[0] 
              << ", um_data[N-1] = " << um_data[N-1] << "\n";

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(um_data);

    return 0;
}
