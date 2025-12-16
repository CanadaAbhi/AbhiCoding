// ================================================================
//  06_memset_meminfo_attributes.cpp
//  HIP Memory Management API — hipMemset, hipMemGetInfo,
//                               hipPointerGetAttributes
//
//  COVERS:
//    hipMemset(ptr, val, bytes)           ← fill device memory
//    hipMemsetAsync(ptr, val, bytes, s)   ← async fill
//    hipMemset2D / hipMemset3D            ← 2-D / 3-D fill
//    hipMemGetInfo(&free, &total)         ← VRAM query
//    hipPointerGetAttributes()            ← pointer metadata
//    Memory type constants explained
//
//  COMPILE:
//    hipcc -O2 06_memset_meminfo_attributes.cpp -o 06_memset_info
//  RUN:
//    ./06_memset_info
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#define HIP_CHECK(call)                                               \
    do {                                                              \
        hipError_t _e = (call);                                       \
        if (_e != hipSuccess) {                                       \
            fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",            \
                    hipGetErrorString(_e), __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void checkValue(const float* d, float expected, int* result, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n && d[g] != expected)
        atomicAdd(result, 1);   // count mismatches
}

// ================================================================
//  DEMO 1 — hipMemset basics
// ================================================================
static void demo_memset_basic()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 1: hipMemset — fill device memory              ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMemset fills each BYTE of the device buffer with val (0..255).
    //  It does NOT set each int/float to val — it sets each byte.
    //  To zero a buffer: hipMemset(ptr, 0, bytes)   → all floats = 0.0f
    //  To fill 0xFF:     hipMemset(ptr, 0xFF, bytes) → all ints = -1 (0xFFFFFFFF)
    //
    //  Signature:
    //    hipError_t hipMemset(void* devPtr, int value, size_t count)

    const int   N     = 1 << 20;
    const size_t bytes = N * sizeof(float);
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    // ── Set to zero ───────────────────────────────────────────
    HIP_CHECK(hipMemset(d_buf, 0, bytes));
    float h0[4] = {9,9,9,9};
    HIP_CHECK(hipMemcpy(h0, d_buf, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  hipMemset(0):   d_buf[0..3] = %.0f %.0f %.0f %.0f  (expect 0)\n",
           h0[0], h0[1], h0[2], h0[3]);

    // ── Set all bytes to 0xFF → each 4-byte int = 0xFFFFFFFF = -1 ─
    HIP_CHECK(hipMemset(d_buf, 0xFF, bytes));
    int h_int[4] = {};
    HIP_CHECK(hipMemcpy(h_int, d_buf, 4 * sizeof(int), hipMemcpyDeviceToHost));
    printf("  hipMemset(0xFF): as int[0..3] = %d %d %d %d  (expect -1)\n",
           h_int[0], h_int[1], h_int[2], h_int[3]);

    // ── Set all bytes to 0x3F → float = 0x3F3F3F3F ≈ 0.748 ──
    HIP_CHECK(hipMemset(d_buf, 0x3F, bytes));
    HIP_CHECK(hipMemcpy(h0, d_buf, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  hipMemset(0x3F): as float[0] = %.6f  (0x3F3F3F3F)\n", h0[0]);

    // ── Zero again (most common use case) ─────────────────────
    HIP_CHECK(hipMemset(d_buf, 0, bytes));
    printf("  hipMemset(0) again: ready for use\n");

    HIP_CHECK(hipFree(d_buf));
    printf("  [PASS] hipMemset basics\n");
}

// ================================================================
//  DEMO 2 — hipMemsetAsync (non-blocking fill)
// ================================================================
static void demo_memset_async()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 2: hipMemsetAsync — non-blocking fill          ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int   N     = 1 << 22;
    const size_t bytes = N * sizeof(float);
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Enqueue async zero-fill then an async copy in the same stream
    HIP_CHECK(hipMemsetAsync(d_buf, 0, bytes, stream));

    // Do some CPU work while GPU fills memory
    volatile int cpu_sum = 0;
    for (int i = 0; i < 10000; ++i) cpu_sum += i;

    // Sync
    HIP_CHECK(hipStreamSynchronize(stream));

    float h4[4] = {9,9,9,9};
    HIP_CHECK(hipMemcpy(h4, d_buf, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  hipMemsetAsync(0): d_buf[0..3] = %.0f %.0f %.0f %.0f  (expect 0)\n",
           h4[0], h4[1], h4[2], h4[3]);
    printf("  CPU sum during async fill: %d\n", cpu_sum);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf));
    printf("  [PASS] hipMemsetAsync\n");
}

// ================================================================
//  DEMO 3 — hipMemset2D (pitched 2-D fill)
// ================================================================
static void demo_memset2d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 3: hipMemset2D — pitched 2-D fill              ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMemset2D fills a width×height rectangle, stepping by pitch bytes.
    //  Useful to initialise a pitched allocation from hipMallocPitch.
    //
    //  Signature:
    //    hipError_t hipMemset2D(void* devPtr, size_t pitch,
    //                           int value,
    //                           size_t width, size_t height)

    const size_t COLS  = 128;
    const size_t ROWS  = 64;
    float*  d_mat = nullptr;
    size_t  pitch = 0;

    HIP_CHECK(hipMallocPitch((void**)&d_mat, &pitch,
                               COLS * sizeof(float), ROWS));
    printf("  hipMallocPitch: %zux%zu floats, pitch=%zu bytes\n",
           COLS, ROWS, pitch);

    // Fill entire pitched matrix with 0xAB
    HIP_CHECK(hipMemset2D(d_mat, pitch, 0xAB, pitch, ROWS));

    // Verify first row
    float h_row[4] = {};
    HIP_CHECK(hipMemcpy(h_row, d_mat, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  hipMemset2D(0xAB): float[0]=%.0f  int[0]=0x%X\n",
           h_row[0], *reinterpret_cast<unsigned*>(&h_row[0]));

    // Zero the logical region only (not the padding)
    HIP_CHECK(hipMemset2D(d_mat, pitch, 0,
                           COLS * sizeof(float),   // width = logical cols only
                           ROWS));
    HIP_CHECK(hipMemcpy(h_row, d_mat, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  hipMemset2D(0) logical cols only: float[0]=%.1f  (expect 0)\n",
           h_row[0]);

    HIP_CHECK(hipFree(d_mat));
    printf("  [PASS] hipMemset2D\n");
}

// ================================================================
//  DEMO 4 — hipMemGetInfo: query VRAM availability
// ================================================================
static void demo_memgetinfo()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 4: hipMemGetInfo — query VRAM                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  Signature:
    //    hipError_t hipMemGetInfo(size_t* free, size_t* total)
    //
    //  Returns VRAM free and total (in bytes) for the CURRENT device.
    //  "free" decreases as more hipMalloc calls succeed.

    size_t free0 = 0, total = 0;
    HIP_CHECK(hipMemGetInfo(&free0, &total));
    printf("  Before allocation:\n");
    printf("    Total VRAM : %.2f GB\n", (double)total / (1 << 30));
    printf("    Free  VRAM : %.2f GB  (%.1f%%)\n",
           (double)free0 / (1 << 30),
           100.0 * free0 / total);

    // Allocate some memory and observe the change
    const size_t ALLOC = 512ULL * 1024 * 1024;   // 512 MB
    float* d_big = nullptr;
    HIP_CHECK(hipMalloc(&d_big, ALLOC));

    size_t free1 = 0, total1 = 0;
    HIP_CHECK(hipMemGetInfo(&free1, &total1));
    printf("  After hipMalloc(512 MB):\n");
    printf("    Free  VRAM : %.2f GB  (%.1f%%)\n",
           (double)free1 / (1 << 30),
           100.0 * free1 / total1);
    printf("    Consumed   : %.2f MB  (%.0f MB allocated)\n",
           (double)(free0 - free1) / (1 << 20),
           (double)ALLOC / (1 << 20));

    HIP_CHECK(hipFree(d_big));

    size_t free2 = 0, total2 = 0;
    HIP_CHECK(hipMemGetInfo(&free2, &total2));
    printf("  After hipFree:\n");
    printf("    Free  VRAM : %.2f GB  (recovered %.2f MB)\n",
           (double)free2 / (1 << 30),
           (double)(free2 - free1) / (1 << 20));

    // ── Practical use: check before large allocation ──────────
    printf("\n  Pattern: allocate as much as safely available\n");
    size_t free_now = 0, dummy = 0;
    HIP_CHECK(hipMemGetInfo(&free_now, &dummy));
    size_t safe_alloc = (size_t)(free_now * 0.85);   // use 85% of free
    printf("  Free = %.2f GB  →  safe to allocate %.2f GB\n",
           (double)free_now  / (1 << 30),
           (double)safe_alloc / (1 << 30));

    printf("  [PASS] hipMemGetInfo\n");
}

// ================================================================
//  DEMO 5 — hipPointerGetAttributes: query pointer metadata
// ================================================================
static void demo_pointer_attributes()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 5: hipPointerGetAttributes — pointer metadata  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipPointerGetAttributes fills a hipPointerAttribute_t struct with:
    //    .memoryType    ← hipMemoryTypeHost / hipMemoryTypeDevice
    //    .type          ← hipMemoryTypeHost / Device / Managed / ...
    //    .device        ← device index where memory lives
    //    .devicePointer ← corresponding device pointer (for host allocs)
    //    .hostPointer   ← corresponding host pointer (for device allocs mapped)
    //    .isManaged     ← 1 if UM, 0 otherwise
    //
    //  Signature:
    //    hipError_t hipPointerGetAttributes(
    //        hipPointerAttribute_t* attributes,
    //        const void*            ptr)

    auto printAttr = [](const char* label, const hipPointerAttribute_t& a) {
        const char* typeStr = "unknown";
        switch (a.type) {
            case hipMemoryTypeHost:    typeStr = "hipMemoryTypeHost";    break;
            case hipMemoryTypeDevice:  typeStr = "hipMemoryTypeDevice";  break;
            case hipMemoryTypeArray:   typeStr = "hipMemoryTypeArray";   break;
            case hipMemoryTypeUnified: typeStr = "hipMemoryTypeUnified"; break;
            default: break;
        }
        printf("  %-20s memoryType=%-28s device=%d isManaged=%d\n",
               label, typeStr, a.device, a.isManaged);
        printf("  %-20s devicePointer=%p  hostPointer=%p\n",
               "", a.devicePointer, a.hostPointer);
    };

    hipPointerAttribute_t attr = {};

    // ── 1. Device pointer (hipMalloc) ─────────────────────────
    float* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, 1024));
    HIP_CHECK(hipPointerGetAttributes(&attr, d_ptr));
    printAttr("hipMalloc ptr:", attr);

    // ── 2. Host pinned pointer (hipHostMalloc) ────────────────
    float* h_pin = nullptr;
    HIP_CHECK(hipHostMalloc(&h_pin, 1024, 0));
    HIP_CHECK(hipPointerGetAttributes(&attr, h_pin));
    printAttr("hipHostMalloc ptr:", attr);

    // ── 3. Managed / Unified Memory pointer ──────────────────
    float* um_ptr = nullptr;
    HIP_CHECK(hipMallocManaged(&um_ptr, 1024));
    HIP_CHECK(hipPointerGetAttributes(&attr, um_ptr));
    printAttr("hipMallocManaged:", attr);

    // ── 4. Regular malloc pointer (NOT registered) ────────────
    float* reg_ptr = static_cast<float*>(malloc(1024));
    hipError_t e = hipPointerGetAttributes(&attr, reg_ptr);
    if (e != hipSuccess) {
        printf("  %-20s → %s  (expected — not HIP-registered)\n",
               "malloc ptr:", hipGetErrorString(e));
        hipGetLastError();   // clear error
    } else {
        printAttr("malloc ptr:", attr);
    }

    // ── Practical use: write a type-agnostic copy function ────
    printf("\n  Practical pattern: detect direction automatically\n");
    auto smartCopy = [](void* dst, const void* src, size_t bytes) {
        hipPointerAttribute_t dA = {}, sA = {};
        bool dstIsDevice = false, srcIsDevice = false;

        if (hipPointerGetAttributes(&dA, dst) == hipSuccess)
            dstIsDevice = (dA.type == hipMemoryTypeDevice);
        if (hipPointerGetAttributes(&sA, src) == hipSuccess)
            srcIsDevice = (sA.type == hipMemoryTypeDevice);

        hipMemcpyKind kind = hipMemcpyDefault;
        if (!srcIsDevice && !dstIsDevice) kind = hipMemcpyHostToHost;
        else if (srcIsDevice && !dstIsDevice) kind = hipMemcpyDeviceToHost;
        else if (!srcIsDevice && dstIsDevice) kind = hipMemcpyHostToDevice;
        else kind = hipMemcpyDeviceToDevice;

        return hipMemcpy(dst, src, bytes, kind);
    };

    float h_val[4] = {1,2,3,4};
    HIP_CHECK(smartCopy(d_ptr, h_val, 4 * sizeof(float)));
    float h_out[4] = {};
    HIP_CHECK(smartCopy(h_out, d_ptr, 4 * sizeof(float)));
    printf("  smartCopy result: %.0f %.0f %.0f %.0f  %s\n",
           h_out[0], h_out[1], h_out[2], h_out[3],
           (h_out[3] == 4.f) ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipHostFree(h_pin));
    HIP_CHECK(hipFree(um_ptr));
    free(reg_ptr);
    printf("  [PASS] hipPointerGetAttributes\n");
}

int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API:  hipMemset / hipMemGetInfo /\n");
    printf("                   hipPointerGetAttributes\n");
    printf("══════════════════════════════════════════════════════════\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  |  VRAM: %.2f GB\n",
           prop.name, (double)prop.totalGlobalMem / (1 << 30));

    demo_memset_basic();
    demo_memset_async();
    demo_memset2d();
    demo_memgetinfo();
    demo_pointer_attributes();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  ALL hipMemset / hipMemGetInfo / hipPointerGetAttributes\n");
    printf("  demos PASSED\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
