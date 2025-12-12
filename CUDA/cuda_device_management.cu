#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    // -----------------------------------------
    // 1. Get number of CUDA-capable GPUs
    // -----------------------------------------
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Number of CUDA-capable GPUs: " << deviceCount << "\n\n";

    if (deviceCount == 0) {
        std::cout << "No GPU available.\n";
        return 0;
    }

    // -----------------------------------------
    // 2. Get current active GPU
    // -----------------------------------------
    int currentDevice = -1;
    cudaGetDevice(&currentDevice);
    std::cout << "Current active device (before set): " 
              << currentDevice << "\n";

    // -----------------------------------------
    // 3. Select a GPU to run on
    // -----------------------------------------
    int deviceToUse = 0;  // choose device 0 for demo
    cudaSetDevice(deviceToUse);

    cudaGetDevice(&currentDevice);
    std::cout << "Active device (after set): " 
              << currentDevice << "\n\n";

    // -----------------------------------------
    // 4. Get GPU properties
    // -----------------------------------------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, currentDevice);

    std::cout << "Device Properties:\n";
    std::cout << "  Name:                   " << prop.name << "\n";
    std::cout << "  Compute Capability:     " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total Global Memory:    " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Multi Processors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << "  Max Threads per Block:  " << prop.maxThreadsPerBlock << "\n\n";

    // -----------------------------------------
    // 5. cudaDeviceSynchronize
    // (This program has no async kernels,
    //  but this API waits for GPU to finish work)
    // -----------------------------------------
    cudaDeviceSynchronize();
    std::cout << "GPU synchronized.\n";

    // -----------------------------------------
    // 6. Reset the device (frees all GPU allocations)
    // -----------------------------------------
    cudaDeviceReset();
    std::cout << "Device reset completed.\n";

    return 0;
}
