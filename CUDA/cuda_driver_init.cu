#include <iostream>
#include <cuda.h>   // Driver API

int main() {
    CUresult res;

    // -------------------------------
    // 1. Initialize the CUDA driver
    // -------------------------------
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << res << "\n";
        return -1;
    }
    std::cout << "CUDA Driver initialized.\n";

    // -------------------------------
    // 2. Get number of CUDA devices
    // -------------------------------
    int deviceCount = 0;
    res = cuDeviceGetCount(&deviceCount);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGetCount failed: " << res << "\n";
        return -1;
    }
    std::cout << "Number of CUDA devices: " << deviceCount << "\n";

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return 0;
    }

    // -------------------------------
    // 3. Get a specific device (device 0)
    // -------------------------------
    CUdevice device;
    res = cuDeviceGet(&device, 0); // get device 0
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed: " << res << "\n";
        return -1;
    }

    char name[256];
    cuDeviceGetName(name, 256, device);
    std::cout << "Device 0 name: " << name << "\n";

    return 0;
}
