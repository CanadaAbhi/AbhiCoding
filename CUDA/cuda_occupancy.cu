#include <iostream>
#include <cuda_runtime.h>

// Simple kernel
__global__ void simpleKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}

int main() {
    int N = 1024;
    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));

    // --------------------------------
    // 1. Set cache configuration for kernel
    // --------------------------------
    cudaFuncSetCacheConfig(simpleKernel, cudaFuncCachePreferShared);
    std::cout << "Cache config set to prefer shared memory.\n";

    // --------------------------------
    // 2. Compute optimal block size for maximum occupancy
    // --------------------------------
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, simpleKernel, 0, 0);
    std::cout << "Recommended block size: " << blockSize 
              << ", min grid size: " << minGrid << "\n";

    // Launch kernel with recommended configuration
    int grid = (N + blockSize - 1) / blockSize;
    simpleKernel<<<grid, blockSize>>>(d_data);

    // --------------------------------
    // 3. Query device limits
    // --------------------------------
    size_t stackLimit;
    cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize);
    std::cout << "Current device stack size: " << stackLimit << " bytes\n";

    // --------------------------------
    // 4. Set device limit
    // --------------------------------
    size_t newStackLimit = 1024 * 16; // 16 KB
    cudaDeviceSetLimit(cudaLimitStackSize, newStackLimit);

    size_t updatedLimit;
    cudaDeviceGetLimit(&updatedLimit, cudaLimitStackSize);
    std::cout << "Updated device stack size: " << updatedLimit << " bytes\n";

    cudaDeviceSynchronize();
    cudaFree(d_data);

    return 0;
}
