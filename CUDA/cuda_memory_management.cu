#include <iostream>
#include <cuda_runtime.h>

int main() {
    // -----------------------------------------
    // 1. cudaMalloc - Allocate memory on GPU
    // -----------------------------------------
    int *d_data = nullptr;
    size_t size = 10 * sizeof(int);

    cudaMalloc((void**)&d_data, size);
    std::cout << "Allocated " << size << " bytes on GPU using cudaMalloc.\n";

    // Set GPU memory to zero
    cudaMemset(d_data, 0, size);
    std::cout << "Initialized GPU memory to zero using cudaMemset.\n";

    // -----------------------------------------
    // 2. Host memory to copy from
    // -----------------------------------------
    int h_data[10];
    for (int i = 0; i < 10; i++) h_data[i] = i;

    // Copy Host -> Device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    std::cout << "Copied Host → Device using cudaMemcpy.\n";

    // Copy back Device → Host
    int h_output[10];
    cudaMemcpy(h_output, d_data, size, cudaMemcpyDeviceToHost);
    std::cout << "Copied Device → Host using cudaMemcpy.\n";

    // Print copied output
    std::cout << "Device→Host copied values: ";
    for (int i = 0; i < 10; i++) std::cout << h_output[i] << " ";
    std::cout << "\n\n";

    // -----------------------------------------
    // 3. cudaMallocManaged - Unified Memory
    // -----------------------------------------
    int *managedPtr = nullptr;
    cudaMallocManaged(&managedPtr, size);
    std::cout << "Allocated Unified Memory using cudaMallocManaged.\n";

    // Use memory directly on CPU
    for (int i = 0; i < 10; i++) managedPtr[i] = i * 10;

    // Synchronize before GPU access
    cudaDeviceSynchronize();

    std::cout << "Unified Memory values: ";
    for (int i = 0; i < 10; i++) std::cout << managedPtr[i] << " ";
    std::cout << "\n\n";

    // -----------------------------------------
    // 4. cudaHostAlloc - Pinned (page-locked) host memory
    // -----------------------------------------
    int *pinnedPtr = nullptr;
    cudaHostAlloc((void**)&pinnedPtr, size, cudaHostAllocDefault);
    std::cout << "Allocated pinned host memory using cudaHostAlloc.\n";

    // Fill pinned memory
    for (int i = 0; i < 10; i++) pinnedPtr[i] = i * 100;

    // Fast transfer: Host → Device
    cudaMemcpy(d_data, pinnedPtr, size, cudaMemcpyHostToDevice);
    std::cout << "Copied pinned Host → Device using cudaMemcpy.\n";

    std::cout << "Pinned Host Memory values: ";
    for (int i = 0; i < 10; i++) std::cout << pinnedPtr[i] << " ";
    std::cout << "\n\n";

    // -----------------------------------------
    // 5. cudaHostRegister - Pin existing host memory
    // -----------------------------------------
    int normal_host_array[10];
    for (int i = 0; i < 10; i++) normal_host_array[i] = i + 500;

    // Pin the existing host memory
    cudaHostRegister(normal_host_array, size, cudaHostRegisterDefault);
    std::cout << "Pinned existing host array using cudaHostRegister.\n";

    // Copy from pinned array to device
    cudaMemcpy(d_data, normal_host_array, size, cudaMemcpyHostToDevice);

    // Unregister host memory
    cudaHostUnregister(normal_host_array);
    std::cout << "Unregistered host memory.\n\n";

    // -----------------------------------------
    // 6. Free all allocated memory
    // -----------------------------------------
    cudaFree(d_data);
    std::cout << "Freed GPU memory.\n";

    cudaFree(managedPtr);
    std::cout << "Freed Unified Memory.\n";

    cudaFreeHost(pinnedPtr);
    std::cout << "Freed pinned host memory.\n";

    return 0;
}
