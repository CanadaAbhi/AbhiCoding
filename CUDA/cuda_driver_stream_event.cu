#include <iostream>
#include <cuda.h>

// Simple kernel to do some work
__global__ void simpleKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}

int main() {
    CUresult res;

    // -------------------------------
    // 1. Initialize driver & context
    // -------------------------------
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    // -------------------------------
    // 2. Allocate device memory
    // -------------------------------
    const int N = 1024;
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, N * sizeof(int));
    cuMemsetD32(d_data, 0, N);

    // -------------------------------
    // 3. Create stream
    // -------------------------------
    CUstream stream;
    cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    std::cout << "Stream created.\n";

    // -------------------------------
    // 4. Create events
    // -------------------------------
    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop, CU_EVENT_DEFAULT);
    std::cout << "Events created.\n";

    // -------------------------------
    // 5. Record start event
    // -------------------------------
    cuEventRecord(start, stream);

    // -------------------------------
    // 6. Launch kernel on stream
    // -------------------------------
    void *args[] = { &d_data, (void*)&N };
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    CUmodule module;
    cuModuleLoad(&module, "kernel.ptx");                 // Load PTX containing simpleKernel
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "simpleKernel");

    cuLaunchKernel(kernel,
                   gridSize, 1, 1,
                   blockSize, 1, 1,
                   0, stream,
                   args, 0);

    // -------------------------------
    // 7. Record stop event
    // -------------------------------
    cuEventRecord(stop, stream);

    // -------------------------------
    // 8. Synchronize stream
    // -------------------------------
    cuStreamSynchronize(stream);
    std::cout << "Stream synchronized.\n";

    // -------------------------------
    // 9. Synchronize events & measure elapsed time
    // -------------------------------
    cuEventSynchronize(stop);

    float ms = 0;
    cuEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";

    // -------------------------------
    // 10. Cleanup
    // -------------------------------
    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuStreamDestroy(stream);
    cuMemFree(d_data);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
