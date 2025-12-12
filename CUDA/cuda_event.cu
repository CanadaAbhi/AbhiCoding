#include <iostream>
#include <cuda_runtime.h>

// Simple kernel that does some work
__global__ void dummyKernel(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float x = data[idx];
        for (int i = 0; i < 1000; i++) {
            x = x * 1.000001f + 0.1f;
        }
        data[idx] = x;
    }
}

int main() {
    const int N = 1 << 20;      // 1 million elements
    const size_t size = N * sizeof(float);

    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    // -------------------------------
    // 1. Create Events
    -------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------
    // 2. Record Start Event
    // -------------------------------
    cudaEventRecord(start, 0);  // 0 = default stream

    // Launch kernel
    dummyKernel<<<256, 256>>>(d_data, N);

    // -------------------------------
    // 3. Record Stop Event
    // -------------------------------
    cudaEventRecord(stop, 0);

    // -------------------------------
    // 4. Wait for stop event to finish
    // -------------------------------
    cudaEventSynchronize(stop);

    // -------------------------------
    // 5. Measure elapsed time in ms
    // -------------------------------
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " 
              << milliseconds << " ms\n";

    // -------------------------------
    // 6. Destroy Events
    // -------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    return 0;
}
