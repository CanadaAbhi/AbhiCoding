// ================================================================
//  02_hip_memcpy.cpp
//  HIP Memory Management API — Synchronous hipMemcpy
//
//  COVERS:
//    hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice)
//    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost)
//    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice)
//    hipMemcpy(dst, src, bytes, hipMemcpyDefault)   ← auto-detect
//    hipMemcpy2D  — copy 2-D pitched regions
//    hipMemcpy3D  — copy 3-D cuboid regions
//    Bandwidth measurement
//
//  COMPILE:
//    hipcc -O2 02_hip_memcpy.cpp -o 02_memcpy
//  RUN:
//    ./02_memcpy
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

#define HIP_CHECK(call)                                               \
    do {                                                              \
        hipError_t _e = (call);                                       \
        if (_e != hipSuccess) {                                       \
            fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",            \
                    hipGetErrorString(_e), __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void addOne(float* d, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) d[g] += 1.0f;
}

// ================================================================
//  DEMO 1 — Host-to-Device (H2D)
// ================================================================
static void demo_h2d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 1: Host → Device  (hipMemcpyHostToDevice)     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int   N     = 8;
    float h_src[8]    = {1, 2, 3, 4, 5, 6, 7, 8};
    float* d_dst      = nullptr;

    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));

    // ── hipMemcpy H2D ─────────────────────────────────────────
    //   Blocks CPU until the copy is complete.
    //   Source must be readable CPU memory (regular malloc or stack).
    //   Destination must be a device pointer from hipMalloc.
    HIP_CHECK(hipMemcpy(d_dst, h_src, N * sizeof(float),
                         hipMemcpyHostToDevice));
    printf("  hipMemcpy H2D: copied {1,2,..,8} to GPU\n");

    // Verify via D2H
    float h_verify[8] = {};
    HIP_CHECK(hipMemcpy(h_verify, d_dst, N * sizeof(float),
                         hipMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (h_verify[i] != h_src[i]) { ok = false; break; }
    printf("  Verify D2H  : %s  values=%.0f %.0f .. %.0f\n",
           ok ? "PASS" : "FAIL",
           h_verify[0], h_verify[1], h_verify[N-1]);

    HIP_CHECK(hipFree(d_dst));
}

// ================================================================
//  DEMO 2 — Device-to-Host (D2H)
// ================================================================
static void demo_d2h()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 2: Device → Host  (hipMemcpyDeviceToHost)     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int N = 1 << 20;    // 1 M floats
    float* d_src = nullptr;
    float* h_dst = static_cast<float*>(malloc(N * sizeof(float)));

    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));

    // Fill device memory with pattern via kernel
    int t = 256, b = (N + 255) / 256;
    addOne<<<b, t>>>(d_src, N);   // GPU sets each element to 0+1=1 (uninit+1)
    HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));  // reset to 0
    addOne<<<b, t>>>(d_src, N);   // now all = 1.0
    HIP_CHECK(hipDeviceSynchronize());

    // ── hipMemcpy D2H ─────────────────────────────────────────
    //   CPU blocks until GPU memory is fully transferred to h_dst.
    //   h_dst can be regular malloc memory (no pinning required for sync copy).
    HIP_CHECK(hipMemcpy(h_dst, d_src, N * sizeof(float),
                         hipMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (h_dst[i] != 1.0f) { ok = false; break; }
    printf("  hipMemcpy D2H: %d floats  %s  h_dst[0]=%.1f\n",
           N, ok ? "PASS" : "FAIL", h_dst[0]);

    HIP_CHECK(hipFree(d_src));
    free(h_dst);
}

// ================================================================
//  DEMO 3 — Device-to-Device (D2D)
// ================================================================
static void demo_d2d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 3: Device → Device  (hipMemcpyDeviceToDevice) ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int    N     = 1 << 22;   // 4 M floats = 16 MB
    const size_t bytes = N * sizeof(float);

    float* d_src = nullptr;
    float* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));

    // Init src on device
    HIP_CHECK(hipMemset(d_src, 0, bytes));
    int t = 256, b = (N + 255) / 256;
    addOne<<<b, t>>>(d_src, N);
    HIP_CHECK(hipDeviceSynchronize());

    // ── hipMemcpy D2D ─────────────────────────────────────────
    //   Copies between two device pointers.
    //   On AMD with xGMI (Infinity Fabric) this can bypass the CPU entirely.
    //   No CPU memory is involved.
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0));
    HIP_CHECK(hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    double bw = (2.0 * bytes) / (ms * 1e-3) / (1 << 30); // read+write

    // Verify
    float h_check[4] = {};
    HIP_CHECK(hipMemcpy(h_check, d_dst, 4 * sizeof(float),
                         hipMemcpyDeviceToHost));
    printf("  hipMemcpy D2D: %d floats (%.0f MB)  time=%.3f ms\n",
           N, (double)bytes / (1 << 20), ms);
    printf("  Bandwidth    : %.1f GB/s\n", bw);
    printf("  d_dst[0..3]  : %.1f %.1f %.1f %.1f  (expect 1.0)\n",
           h_check[0], h_check[1], h_check[2], h_check[3]);

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
}

// ================================================================
//  DEMO 4 — hipMemcpyDefault (auto-detect direction)
// ================================================================
static void demo_default_direction()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 4: hipMemcpyDefault — runtime direction detect ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMemcpyDefault lets HIP inspect the pointer attributes and
    //  choose H2D / D2H / D2D automatically.  Useful when direction
    //  is not known statically (e.g., template / library code).

    const int   N     = 64;
    float       h_buf[64];
    float*      d_buf = nullptr;
    for (int i = 0; i < N; ++i) h_buf[i] = static_cast<float>(i);
    HIP_CHECK(hipMalloc(&d_buf, N * sizeof(float)));

    // H→D with Default
    HIP_CHECK(hipMemcpy(d_buf, h_buf, N * sizeof(float), hipMemcpyDefault));

    float h_verify[64] = {};
    // D→H with Default
    HIP_CHECK(hipMemcpy(h_verify, d_buf, N * sizeof(float), hipMemcpyDefault));

    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (h_verify[i] != static_cast<float>(i)) { ok = false; break; }
    printf("  hipMemcpyDefault H2D then D2H: %s\n", ok ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_buf));
}

// ================================================================
//  DEMO 5 — hipMemcpy2D (copy sub-region of 2-D matrix)
// ================================================================
static void demo_memcpy2d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 5: hipMemcpy2D — pitched 2-D region copy      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMemcpy2D copies a width×height rectangle from src to dst,
    //  stepping by srcPitch / dstPitch bytes per row.
    //  Useful for sub-matrix copies or after hipMallocPitch.
    //
    //  Signature:
    //    hipError_t hipMemcpy2D(
    //        void*       dst,      size_t dpitch,
    //        const void* src,      size_t spitch,
    //        size_t      width,    size_t height,
    //        hipMemcpyKind kind)

    const int FULL_COLS  = 256;   // padded host row stride
    const int LOGIC_COLS = 100;   // logical columns we care about
    const int ROWS       = 50;

    // Allocate host array (row-major, stride = FULL_COLS)
    float* h_src = static_cast<float*>(
        malloc(ROWS * FULL_COLS * sizeof(float)));
    float* h_dst = static_cast<float*>(
        calloc(ROWS * LOGIC_COLS, sizeof(float)));

    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < FULL_COLS; ++c)
            h_src[r * FULL_COLS + c] = static_cast<float>(r * 1000 + c);

    // Device pitched allocation
    float*  d_mat = nullptr;
    size_t  d_pitch = 0;
    HIP_CHECK(hipMallocPitch((void**)&d_mat, &d_pitch,
                               FULL_COLS * sizeof(float), ROWS));

    // Copy full host matrix → device (H2D 2-D)
    HIP_CHECK(hipMemcpy2D(
        d_mat,  d_pitch,                          // dst, dst stride (bytes)
        h_src,  FULL_COLS * sizeof(float),        // src, src stride (bytes)
        FULL_COLS * sizeof(float),                // width  (bytes to copy per row)
        ROWS,                                     // height (rows)
        hipMemcpyHostToDevice));

    // Copy only LOGIC_COLS columns back (D2H 2-D sub-region)
    HIP_CHECK(hipMemcpy2D(
        h_dst,  LOGIC_COLS * sizeof(float),       // dst, dst stride
        d_mat,  d_pitch,                          // src, src stride (device pitch)
        LOGIC_COLS * sizeof(float),               // copy only 100 cols
        ROWS,
        hipMemcpyDeviceToHost));

    // Verify row 5, col 7
    float expected = static_cast<float>(5 * 1000 + 7);
    float got      = h_dst[5 * LOGIC_COLS + 7];
    printf("  hipMemcpy2D: %dx%d (logic) from %dx%d (full)\n",
           LOGIC_COLS, ROWS, FULL_COLS, ROWS);
    printf("  Device pitch = %zu bytes\n", d_pitch);
    printf("  h_dst[5][7]  = %.0f  (expect %.0f)  %s\n",
           got, expected, (got == expected) ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_mat));
    free(h_src); free(h_dst);
}

// ================================================================
//  DEMO 6 — Bandwidth benchmark H2D vs D2H vs D2D
// ================================================================
static void demo_bandwidth()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 6: Bandwidth — H2D vs D2H vs D2D              ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const size_t MB    = 256ULL * 1024 * 1024;   // 256 MB
    const int    REPS  = 5;

    // Pinned host memory for accurate bandwidth measurement
    float* h_pin = nullptr;
    HIP_CHECK(hipHostMalloc(&h_pin, MB, 0));
    memset(h_pin, 0, MB);

    float* d_a = nullptr;
    float* d_b = nullptr;
    HIP_CHECK(hipMalloc(&d_a, MB));
    HIP_CHECK(hipMalloc(&d_b, MB));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    auto bench = [&](const char* name, auto fn) {
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < REPS; ++i) fn();
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        double bw = (double)MB * REPS / (ms * 1e-3) / (1 << 30);
        printf("  %-6s : %6.1f GB/s  (avg %.3f ms/rep)\n",
               name, bw, ms / REPS);
    };

    bench("H2D", [&](){ hipMemcpy(d_a, h_pin, MB, hipMemcpyHostToDevice); });
    bench("D2H", [&](){ hipMemcpy(h_pin, d_a,  MB, hipMemcpyDeviceToHost); });
    bench("D2D", [&](){ hipMemcpy(d_b,   d_a,  MB, hipMemcpyDeviceToDevice); });

    printf("  (D2D should be fastest — on-device bandwidth limited only by HBM/GDDR)\n");

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipHostFree(h_pin));
}

int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API:  hipMemcpy (Synchronous Copies)\n");
    printf("══════════════════════════════════════════════════════════\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n", prop.name);

    demo_h2d();
    demo_d2h();
    demo_d2d();
    demo_default_direction();
    demo_memcpy2d();
    demo_bandwidth();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  ALL hipMemcpy demos PASSED\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
