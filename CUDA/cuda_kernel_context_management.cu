#include <iostream>
#include <cuda.h>

int main() {
    CUresult res;

    // -------------------------------
    // 1. Initialize CUDA Driver
    // -------------------------------
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << res << "\n";
        return -1;
    }

    // -------------------------------
    // 2. Get device 0
    // -------------------------------
    CUdevice device;
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed: " << res << "\n";
        return -1;
    }

    char name[256];
    cuDeviceGetName(name, 256, device);
    std::cout << "Using device: " << name << "\n";

    // -------------------------------
    // 3. Create a CUDA context
    // -------------------------------
    CUcontext context;
    res = cuCtxCreate(&context, 0, device); // flags = 0
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxCreate failed: " << res << "\n";
        return -1;
    }
    std::cout << "CUDA context created.\n";

    // -------------------------------
    // 4. Set context as current
    // -------------------------------
    res = cuCtxSetCurrent(context);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxSetCurrent failed: " << res << "\n";
        return -1;
    }
    std::cout << "Context set as current.\n";

    // -------------------------------
    // 5. Perform some work (optional)
    // For demo, just query free memory
    // -------------------------------
    size_t freeMem, totalMem;
    cuMemGetInfo(&freeMem, &totalMem);
    std::cout << "Device memory: " << freeMem/1024/1024 << " MB free of " 
              << totalMem/1024/1024 << " MB total\n";

    // -------------------------------
    // 6. Destroy context
    // -------------------------------
    res = cuCtxDestroy(context);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxDestroy failed: " << res << "\n";
        return -1;
    }
    std::cout << "CUDA context destroyed.\n";

    return 0;
}
