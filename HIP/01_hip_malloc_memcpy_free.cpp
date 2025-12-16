// ============================================================
//  01_hip_malloc_memcpy_free.cpp
//  Demonstrates: hipMalloc  hipMemcpy  hipFree
//
//  Compile:
//    hipcc -O2 01_hip_malloc_memcpy_free.cpp -o 01_demo
//  Run:
//    ./01_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// ── helper macro: abort on any HIP error ─────────────────────
#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── trivial GPU kernel: square every element ─────────────────
__global__ void squareKernel(float* d, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = d[idx] * d[idx];
}

int main()
{
    const int N     = 1 << 20;          // 1 M floats
    const size_t SZ = N * sizeof(float);

    // ── 1. Allocate HOST memory (ordinary malloc) ─────────────
    float* h_in  = static_cast<float*>(malloc(SZ));
    float* h_out = static_cast<float*>(malloc(SZ));
    assert(h_in && h_out);

    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // ── 2. hipMalloc – allocate DEVICE (GPU VRAM) memory ─────
    //    Signature: hipError_t hipMalloc(void** ptr, size_t size)
    float* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, SZ));
    printf("[hipMalloc]  Allocated %.2f MB on GPU\n",
           static_cast<double>(SZ) / (1024.0 * 1024.0));

    // ── 3. hipMemcpy Host → Device ────────────────────────────
    //    Flags: hipMemcpyHostToDevice  hipMemcpyDeviceToHost
    //           hipMemcpyDeviceToDevice  hipMemcpyDefault
    HIP_CHECK(hipMemcpy(d_data, h_in, SZ, hipMemcpyHostToDevice));
    printf("[hipMemcpy]  H->D  %d floats copied\n", N);

    // ── 4. Launch a kernel to modify the device data ──────────
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    squareKernel<<<BLOCKS, THREADS>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // ── 5. hipMemcpy Device → Host ────────────────────────────
    HIP_CHECK(hipMemcpy(h_out, d_data, SZ, hipMemcpyDeviceToHost));
    printf("[hipMemcpy]  D->H  %d floats copied\n", N);

    // ── 6. Verify results ─────────────────────────────────────
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * static_cast<float>(i);
        if (h_out[i] != expected) { ok = false; break; }
    }
    printf("[Verify]     %s\n", ok ? "PASS – all values correct" : "FAIL");

    // ── 7. hipFree – release device memory ───────────────────
    HIP_CHECK(hipFree(d_data));
    printf("[hipFree]    Device memory released\n");

    // ── 8. hipMemcpy: Device-to-Device copy demo ─────────────
    float* d_src = nullptr;
    float* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, SZ));
    HIP_CHECK(hipMalloc(&d_dst, SZ));
    HIP_CHECK(hipMemcpy(d_src, h_in, SZ, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst, d_src, SZ, hipMemcpyDeviceToDevice));  // D2D
    printf("[hipMemcpy]  D->D copy complete (no CPU involvement)\n");
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    free(h_in);
    free(h_out);

    printf("\nAll hipMalloc / hipMemcpy / hipFree demos PASSED\n");
    return 0;
}