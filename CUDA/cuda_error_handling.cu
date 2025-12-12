#include <iostream>
#include <cuda_runtime.h>

// A kernel with intentional out-of-bounds access
__global__ void faultyKernel(int *d_data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Intentional out-of-bounds access to generate error
    d_data[idx + 100] = idx;
}

int main() {
    const int N = 10;
    int *d_data;

    // Allocate device memory
    cudaMalloc(&d_data, N * sizeof(int));

    // Launch kernel with error
    faultyKernel<<<1, 10>>>(d_data);

    // -------------------------------
    // 1. Peek at last error (does not reset)
    // -------------------------------
    cudaError_t errPeek = cudaPeekAtLastError();
    if (errPeek != cudaSuccess) {
        std::cout << "[Peek] CUDA error: " 
                  << cudaGetErrorString(errPeek) << "\n";
    } else {
        std::cout << "[Peek] No error detected.\n";
    }

    // -------------------------------
    // 2. Get last error (clears it)
    // -------------------------------
    cudaError_t errGet = cudaGetLastError();
    if (errGet != cudaSuccess) {
        std::cout << "[Get] CUDA error: " 
                  << cudaGetErrorString(errGet) << "\n";
    } else {
        std::cout << "[Get] No error detected.\n";
    }

    // -------------------------------
    // 3. Synchronize to catch async errors
    // -------------------------------
    cudaDeviceSynchronize();
    cudaError_t errSync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        std::cout << "[Sync] CUDA error after device sync: "
                  << cudaGetErrorString(errSync) << "\n";
    }

    // Free device memory
    cudaFree(d_data);

    return 0;
}
