// ============================================================
//  03_device_synchronize.cpp
//  Demonstrates: hipDeviceSynchronize  hipGetDeviceProperties
//                hipSetDevice  hipGetDevice  hipGetDeviceCount
//
//  Compile:
//    hipcc -O2 03_device_synchronize.cpp -o 03_demo
//  Run:
//    ./03_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── A deliberately slow kernel (many iterations) ─────────────
__global__ void slowKernel(float* out, int n, int iters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float val = static_cast<float>(gid);
    for (int i = 0; i < iters; ++i)
        val = sqrtf(val * val + 1.0f);
    out[gid] = val;
}

int main()
{
    // ── 1. Query all devices ──────────────────────────────────
    printf("=== 1. Device enumeration ===\n");
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("  hipGetDeviceCount → %d GPU(s) found\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, dev));
        printf("\n  --- GPU %d ---\n", dev);
        printf("  Name              : %s\n",        prop.name);
        printf("  Compute Units     : %d\n",        prop.multiProcessorCount);
        printf("  Max threads/block : %d\n",        prop.maxThreadsPerBlock);
        printf("  Warp/wavefront sz : %d\n",        prop.warpSize);   // 64 on AMD
        printf("  VRAM (global)     : %.2f GB\n",
               static_cast<double>(prop.totalGlobalMem) / (1<<30));
        printf("  Shared mem/block  : %zu KB\n",    prop.sharedMemPerBlock / 1024);
        printf("  Clock rate        : %.0f MHz\n",
               static_cast<double>(prop.clockRate) / 1000.0);
        printf("  Memory clock      : %.0f MHz\n",
               static_cast<double>(prop.memoryClockRate) / 1000.0);
        printf("  Memory bus width  : %d bits\n",   prop.memoryBusWidth);
        printf("  L2 cache size     : %d KB\n",     prop.l2CacheSize / 1024);
        printf("  Max grid dim      : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  PCIe bus ID       : %04x:%02x:%02x\n",
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }

    // ── 2. hipSetDevice / hipGetDevice ───────────────────────
    printf("\n=== 2. hipSetDevice / hipGetDevice ===\n");
    HIP_CHECK(hipSetDevice(0));      // always select GPU 0 first
    int currentDev = -1;
    HIP_CHECK(hipGetDevice(&currentDev));
    printf("  Currently selected device: %d\n", currentDev);

    // ── 3. hipDeviceSynchronize — wait for ALL work on device ─
    printf("\n=== 3. hipDeviceSynchronize ===\n");
    printf("  Purpose: blocks CPU until every pending GPU operation completes.\n");
    printf("  Without it: timing, reads, or frees may race with in-flight kernels.\n\n");

    const int N = 1 << 18;   // 256 K elements
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));

    // --- WITHOUT sync: unsafe pattern (shown for illustration) ---
    int threads = 256, blocks = (N + threads - 1) / threads;
    slowKernel<<<blocks, threads>>>(d_out, N, 200);
    // If we called hipMemcpy here without sync, results may be incomplete.
    // hipMemcpy IS synchronous, so it internally syncs — but explicit sync
    // is clearer and necessary before things like hipFree or timing.

    // --- WITH sync: safe pattern ---
    HIP_CHECK(hipDeviceSynchronize());   // ← CPU waits here until GPU done
    printf("  slowKernel completed (hipDeviceSynchronize returned)\n");

    float sample[4] = {};
    HIP_CHECK(hipMemcpy(sample, d_out, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  sample out[0..3]: %.4f  %.4f  %.4f  %.4f\n",
           sample[0], sample[1], sample[2], sample[3]);

    // ── 4. Timing pattern with hipDeviceSynchronize ───────────
    printf("\n=== 4. CPU-side timing with hipDeviceSynchronize ===\n");
    // For GPU timing see 04_streams_events.cpp (hipEvent_t is more accurate).
    // But a quick CPU wall-clock approach:
    HIP_CHECK(hipDeviceSynchronize());         // flush any prior work
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    slowKernel<<<blocks, threads>>>(d_out, N, 500);
    HIP_CHECK(hipDeviceSynchronize());         // wait for kernel

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    printf("  slowKernel (iters=500) wall time: %.3f ms\n", ms);

    // ── 5. hipDeviceReset — nuke all state (use with caution) ─
    printf("\n=== 5. hipDeviceReset (clears all allocations on device) ===\n");
    // hipFree first before reset in production code
    HIP_CHECK(hipFree(d_out));
    // hipDeviceReset destroys the HIP context — all allocations become invalid
    HIP_CHECK(hipDeviceReset());
    printf("  hipDeviceReset complete\n");

    printf("\nAll hipDeviceSynchronize demos PASSED\n");
    return 0;
}
