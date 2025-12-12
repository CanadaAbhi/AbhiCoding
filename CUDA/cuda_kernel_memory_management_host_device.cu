#include <iostream>
#include <cuda.h>

int main() {
    CUresult res;

    // -------------------------------
    // 1. Initialize driver
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

    // -------------------------------
    // 3. Create context
    // -------------------------------
    CUcontext context;
    res = cuCtxCreate(&context, 0, device);

    // -------------------------------
    // 4. Allocate device memory
    // -------------------------------
    CUdeviceptr d_ptr1, d_ptr2;
    size_t N = 10;
    res = cuMemAlloc(&d_ptr1, N * sizeof(int));
    res = cuMemAlloc(&d_ptr2, N * sizeof(int));
    std::cout << "Allocated device memory.\n";

    // -------------------------------
    // 5. Host data
    // -------------------------------
    int h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = i + 1;

    // -------------------------------
    // 6. Copy Host → Device
    // -------------------------------
    cuMemcpyHtoD(d_ptr1, h_data, N * sizeof(int));
    std::cout << "Copied Host → Device.\n";

    // -------------------------------
    // 7. Copy Device → Device
    // -------------------------------
    cuMemcpyDtoD(d_ptr2, d_ptr1, N * sizeof(int));
    std::cout << "Copied Device → Device.\n";

    // -------------------------------
    // 8. Copy Device → Host
    // -------------------------------
    int h_output[N];
    cuMemcpyDtoH(h_output, d_ptr2, N * sizeof(int));
    std::cout << "Copied Device → Host.\n";

    std::cout << "Output: ";
    for (int i = 0; i < N; i++) std::cout << h_output[i] << " ";
    std::cout << "\n";

    // -------------------------------
    // 9. Free device memory
    // -------------------------------
    cuMemFree(d_ptr1);
    cuMemFree(d_ptr2);
    std::cout << "Freed device memory.\n";

    // -------------------------------
    // 10. Destroy context
    // -------------------------------
    cuCtxDestroy(context);

    return 0;
}
