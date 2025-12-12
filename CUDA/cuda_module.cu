#include <iostream>
#include <cuda.h>

int main() {
    CUresult res;

    // -------------------------------
    // 1. Initialize CUDA Driver
    // -------------------------------
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed\n";
        return -1;
    }

    // -------------------------------
    // 2. Get device 0 and create context
    // -------------------------------
    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, 0, device);

    // -------------------------------
    // 3. Load PTX module
    // -------------------------------
    CUmodule module;
    res = cuModuleLoad(&module, "kernel.ptx");
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuModuleLoad failed\n";
        return -1;
    }
    std::cout << "Module loaded.\n";

    // -------------------------------
    // 4. Get kernel function handle
    // -------------------------------
    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "addKernel");
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuModuleGetFunction failed\n";
        return -1;
    }
    std::cout << "Kernel function handle obtained.\n";

    // -------------------------------
    // 5. Allocate device memory
    // -------------------------------
    const int N = 10;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(int));

    int h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = i;

    cuMemcpyHtoD(d_data, h_data, N * sizeof(int));

    // -------------------------------
    // 6. Launch kernel
    // -------------------------------
    void *args[] = { &d_data, (void*)&N };

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    res = cuLaunchKernel(kernel,
                         gridSize, 1, 1,    // gridDim
                         blockSize, 1, 1,   // blockDim
                         0, nullptr,        // sharedMemBytes, stream
                         args, 0);          // kernel params

    if (res != CUDA_SUCCESS) {
        std::cerr << "cuLaunchKernel failed\n";
        return -1;
    }

    cuCtxSynchronize();
    std::cout << "Kernel executed.\n";

    // -------------------------------
    // 7. Copy back results
    // -------------------------------
    cuMemcpyDtoH(h_data, d_data, N * sizeof(int));

    std::cout << "Output: ";
    for (int i = 0; i < N; i++) std::cout << h_data[i] << " ";
    std::cout << "\n";

    // -------------------------------
    // 8. Cleanup
    // -------------------------------
    cuMemFree(d_data);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
