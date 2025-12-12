#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Size of array
    const int N = 10;
    const size_t size = N * sizeof(int);

    // -------------------------------
    // Allocate device memory
    // -------------------------------
    int *d_data = nullptr;
    cudaMalloc(&d_data, size);

    // -------------------------------
    // Host memory
    // -------------------------------
    int h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    // -------------------------------
    // 1. Create a CUDA stream
    // -------------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "Stream created.\n";

    // -------------------------------
    // 2. Async Memset (Device)
    // -------------------------------
    cudaMemsetAsync(d_data, 0, size, stream);
    std::cout << "cudaMemsetAsync issued.\n";

    // -------------------------------
    // 3. Async copy Host → Device
    // -------------------------------
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    std::cout << "cudaMemcpyAsync Host→Device issued.\n";

    // -------------------------------
    // Some CPU work in parallel
    // -------------------------------
    std::cout << "\nCPU is free to work while GPU is copying...\n";
    std::cout << "Doing some CPU work...\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "CPU working " << i << "\n";
    }
    std::cout << "\n";

    // -------------------------------
    // 4. Wait for stream to complete
    // -------------------------------
    cudaStreamSynchronize(stream);
    std::cout << "Stream synchronized (all async GPU tasks completed).\n";

    // -------------------------------
    // 5. Copy back Device → Host (sync copy for simplicity)
    // -------------------------------
    int h_output[N];
    cudaMemcpy(h_output, d_data, size, cudaMemcpyDeviceToHost);

    std::cout << "Device→Host copied values: ";
    for (int i = 0; i < N; i++) std::cout << h_output[i] << " ";
    std::cout << "\n";

    // -------------------------------
    // 6. Destroy stream
    // -------------------------------
    cudaStreamDestroy(stream);
    std::cout << "Stream destroyed.\n";

    // Free memory
    cudaFree(d_data);

    return 0;
}
