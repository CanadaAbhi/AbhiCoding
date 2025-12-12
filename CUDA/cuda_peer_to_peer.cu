#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cout << "Need at least 2 GPUs for P2P example.\n";
        return 0;
    }

    int dev0 = 0, dev1 = 1;
    int canAccess = 0;

    // -------------------------------
    // 1. Check P2P capability
    // -------------------------------
    cudaDeviceCanAccessPeer(&canAccess, dev0, dev1);
    if (!canAccess) {
        std::cout << "GPU 0 cannot access GPU 1 memory directly.\n";
    } else {
        std::cout << "GPU 0 can access GPU 1 memory directly.\n";
    }

    cudaDeviceCanAccessPeer(&canAccess, dev1, dev0);
    if (!canAccess) {
        std::cout << "GPU 1 cannot access GPU 0 memory directly.\n";
    } else {
        std::cout << "GPU 1 can access GPU 0 memory directly.\n";
    }

    // -------------------------------
    // 2. Enable P2P access
    // -------------------------------
    cudaSetDevice(dev0);
    cudaDeviceEnablePeerAccess(dev1, 0);

    cudaSetDevice(dev1);
    cudaDeviceEnablePeerAccess(dev0, 0);

    std::cout << "P2P access enabled between GPU 0 and GPU 1.\n";

    // -------------------------------
    // 3. Allocate memory on both GPUs
    // -------------------------------
    cudaSetDevice(dev0);
    int *d0;
    cudaMalloc(&d0, sizeof(int) * 10);
    cudaMemset(d0, 42, sizeof(int) * 10);

    cudaSetDevice(dev1);
    int *d1;
    cudaMalloc(&d1, sizeof(int) * 10);
    cudaMemset(d1, 0, sizeof(int) * 10);

    // -------------------------------
    // 4. Copy data GPU0 -> GPU1 using P2P
    // -------------------------------
    cudaMemcpyPeer(d1, dev1, d0, dev0, sizeof(int) * 10);
    std::cout << "Data copied GPU0 -> GPU1 using cudaMemcpyPeer.\n";

    // -------------------------------
    // 5. Copy data back to host for verification
    // -------------------------------
    int h_data[10];
    cudaMemcpy(h_data, d1, sizeof(int) * 10, cudaMemcpyDeviceToHost);
    std::cout << "GPU1 data: ";
    for (int i = 0; i < 10; i++) std::cout << h_data[i] << " ";
    std::cout << "\n";

    // -------------------------------
    // 6. Cleanup
    // -------------------------------
    cudaFree(d0);
    cudaFree(d1);

    return 0;
}
