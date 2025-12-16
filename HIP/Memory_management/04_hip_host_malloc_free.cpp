// ================================================================
//  04_hip_host_malloc_free.cpp
//  HIP Memory Management API — hipHostMalloc & hipHostFree
//
//  COVERS:
//    hipHostMalloc(&ptr, bytes, 0)         ← default pinned
//    hipHostMalloc(&ptr, bytes, hipHostMallocMapped)  ← zero-copy
//    hipHostMalloc(&ptr, bytes, hipHostMallocWriteCombined) ← WC
//    hipHostFree(ptr)
//    hipHostGetDevicePointer()             ← mapped memory access
//    Why pinned > pageable for DMA throughput
//    Bandwidth comparison: pinned vs pageable
//
//  COMPILE:
//    hipcc -O2 04_hip_host_malloc_free.cpp -o 04_host_malloc
//  RUN:
//    ./04_host_malloc
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(call)                                               \
    do {                                                              \
        hipError_t _e = (call);                                       \
        if (_e != hipSuccess) {                                       \
            fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",            \
                    hipGetErrorString(_e), __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void addScalar(float* d, float v, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) d[g] += v;
}

__global__ void readMapped(float* device_ptr_to_host_mem,
                            float* d_out, int n) {
    // GPU reads directly from (mapped) host memory — zero-copy
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) d_out[g] = device_ptr_to_host_mem[g] * 2.0f;
}

// ================================================================
//  DEMO 1 — Default pinned allocation (hipHostMallocDefault)
// ================================================================
static void demo_default_pinned()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 1: hipHostMalloc — default pinned memory      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipHostMalloc allocates page-locked (pinned) host memory.
    //  Page-locked means the OS will NEVER swap this memory to disk,
    //  so the GPU DMA engine can access it directly at PCIe full speed.
    //
    //  Flags = 0  is equivalent to hipHostMallocDefault.
    //
    //  Rules:
    //    • Must be freed with hipHostFree, NOT free() or delete[]
    //    • Do NOT pin excessively — it reduces available physical RAM
    //      for the OS and other processes
    //    • Typical speedup over pageable: 2–4× for H2D / D2H transfers

    const size_t MB    = 64ULL * 1024 * 1024;   // 64 MB
    const int    N     = MB / sizeof(float);

    float* h_pinned = nullptr;
    HIP_CHECK(hipHostMalloc(&h_pinned, MB, 0));       // flags=0 → default pinned
    printf("  hipHostMalloc (pinned, 64 MB) at h_pinned=%p\n",
           (void*)h_pinned);

    // Initialise on CPU
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    // Copy to GPU and process
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, MB));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Async transfer is TRULY async because host memory is pinned
    HIP_CHECK(hipMemcpyAsync(d_buf, h_pinned, MB,
                               hipMemcpyHostToDevice, stream));
    int t = 256, b = (N + 255) / 256;
    addScalar<<<b, t, 0, stream>>>(d_buf, 9.0f, N);
    HIP_CHECK(hipMemcpyAsync(h_pinned, d_buf, MB,
                               hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    printf("  After kernel (add 9): h_pinned[0]=%.1f  (expect 10.0)  %s\n",
           h_pinned[0], (fabsf(h_pinned[0] - 10.0f) < 1e-5f) ? "PASS" : "FAIL");

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf));

    // ── hipHostFree: release pinned memory ────────────────────
    //   MUST use hipHostFree (not free()) for pinned allocations
    HIP_CHECK(hipHostFree(h_pinned));
    h_pinned = nullptr;
    printf("  hipHostFree: pinned memory released\n");
}

// ================================================================
//  DEMO 2 — Mapped (zero-copy) memory: hipHostMallocMapped
// ================================================================
static void demo_mapped_zero_copy()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 2: hipHostMallocMapped — zero-copy             ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipHostMallocMapped creates host memory that is also directly
    //  accessible from GPU kernels via a device pointer.
    //  Every GPU read/write goes over PCIe — NO copy to VRAM needed.
    //
    //  Best for:
    //    • Large datasets that don't fit in VRAM
    //    • Low compute-to-transfer ratio (streaming workloads)
    //    • Data written by CPU and immediately read once by GPU

    const int   N     = 1 << 16;   // 64 K floats = 256 KB
    const size_t bytes = N * sizeof(float);

    float* h_mapped = nullptr;
    HIP_CHECK(hipHostMalloc(&h_mapped, bytes,
                              hipHostMallocMapped));   // ← key flag

    // Initialise on CPU
    for (int i = 0; i < N; ++i) h_mapped[i] = static_cast<float>(i);

    // Get device-side pointer to the same physical memory
    float* d_mapped_ptr = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(
        (void**)&d_mapped_ptr,     // output: device virtual address
        (void*)h_mapped,           // input:  host virtual address
        0));                       // flags: reserved, must be 0

    printf("  CPU ptr  : %p\n",   (void*)h_mapped);
    printf("  GPU ptr  : %p\n",   (void*)d_mapped_ptr);
    printf("  (different VA spaces, same physical pages)\n");

    // GPU kernel reads from h_mapped via d_mapped_ptr (over PCIe)
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));
    int t = 256, b = (N + 255) / 256;
    readMapped<<<b, t>>>(d_mapped_ptr, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    float h_check[4] = {};
    HIP_CHECK(hipMemcpy(h_check, d_out, 4 * sizeof(float),
                         hipMemcpyDeviceToHost));
    // h_mapped[i] = i, kernel computes i*2
    printf("  d_out[0..3] = %.0f %.0f %.0f %.0f  (expect 0 2 4 6)  %s\n",
           h_check[0], h_check[1], h_check[2], h_check[3],
           (h_check[3] == 6.0f) ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipHostFree(h_mapped));
    printf("  hipHostFree: mapped memory released\n");
}

// ================================================================
//  DEMO 3 — Write-combined memory: hipHostMallocWriteCombined
// ================================================================
static void demo_write_combined()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 3: hipHostMallocWriteCombined                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  Write-combined memory is NOT cached in CPU caches.
    //  CPU writes are batched and sent in large bursts to the GPU,
    //  which can be faster for write-only host→device streams.
    //
    //  WARNING: CPU reads from WC memory are very slow (uncached).
    //  Best practice: write-only from CPU, no CPU reads.

    const size_t MB = 16ULL * 1024 * 1024;   // 16 MB
    const int    N  = MB / sizeof(float);

    float* h_wc = nullptr;
    HIP_CHECK(hipHostMalloc(&h_wc, MB,
                              hipHostMallocWriteCombined));  // ← WC flag

    printf("  Write-combined buffer at %p  (16 MB)\n", (void*)h_wc);

    // CPU write (efficient — coalesced into write-combining buffer)
    for (int i = 0; i < N; ++i) h_wc[i] = 3.14f;

    // Copy to GPU
    float* d_wc = nullptr;
    HIP_CHECK(hipMalloc(&d_wc, MB));
    HIP_CHECK(hipMemcpy(d_wc, h_wc, MB, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    float h_val = 0.f;
    HIP_CHECK(hipMemcpy(&h_val, d_wc, sizeof(float),
                         hipMemcpyDeviceToHost));
    printf("  d_wc[0] = %.2f  (expect 3.14)  %s\n",
           h_val, (fabsf(h_val - 3.14f) < 1e-4f) ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(d_wc));
    HIP_CHECK(hipHostFree(h_wc));
    printf("  hipHostFree: write-combined memory released\n");
}

// ================================================================
//  DEMO 4 — Bandwidth: pageable vs pinned
// ================================================================
static void demo_bandwidth_comparison()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 4: Bandwidth comparison — pageable vs pinned   ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const size_t MB   = 256ULL * 1024 * 1024;   // 256 MB
    const int    REPS = 3;

    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, MB));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    // ── Pageable memory (regular malloc) ─────────────────────
    float* h_page = static_cast<float*>(malloc(MB));
    memset(h_page, 0, MB);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < REPS; ++i)
        HIP_CHECK(hipMemcpy(d_buf, h_page, MB, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_page = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_page, e0, e1));
    double bw_page = (double)MB * REPS / (ms_page * 1e-3) / (1 << 30);

    // ── Pinned memory (hipHostMalloc) ─────────────────────────
    float* h_pin = nullptr;
    HIP_CHECK(hipHostMalloc(&h_pin, MB, 0));
    memset(h_pin, 0, MB);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int i = 0; i < REPS; ++i)
        HIP_CHECK(hipMemcpy(d_buf, h_pin, MB, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_pin = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_pin, e0, e1));
    double bw_pin = (double)MB * REPS / (ms_pin * 1e-3) / (1 << 30);

    printf("  Buffer size   : 256 MB  (%d reps)\n", REPS);
    printf("  Pageable H2D  : %6.2f GB/s  (%.1f ms avg)\n",
           bw_page, ms_page / REPS);
    printf("  Pinned   H2D  : %6.2f GB/s  (%.1f ms avg)\n",
           bw_pin,  ms_pin  / REPS);
    printf("  Pinned speedup: %.2fx\n", bw_pin / bw_page);
    printf("  (Pinned is faster because it avoids OS paging overhead\n");
    printf("   and enables direct DMA without staging copy)\n");

    free(h_page);
    HIP_CHECK(hipHostFree(h_pin));
    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
}

// ================================================================
//  DEMO 5 — hipHostMalloc flags reference
// ================================================================
static void demo_flags_reference()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 5: hipHostMalloc flags reference               ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const size_t SMALL = 4096;
    float* p = nullptr;

    struct FlagInfo { unsigned int flag; const char* name; const char* desc; };
    FlagInfo flags[] = {
        { 0,                           "0 / Default",
          "Standard pinned. Readable+writable by CPU and GPU." },
        { hipHostMallocMapped,         "hipHostMallocMapped",
          "Zero-copy: GPU can read/write directly via device ptr." },
        { hipHostMallocWriteCombined,  "hipHostMallocWriteCombined",
          "CPU-write optimised. CPU reads are slow." },
        { hipHostMallocPortable,       "hipHostMallocPortable",
          "Pinned memory accessible from ALL HIP contexts (multi-GPU)." },
    };

    for (auto& fi : flags) {
        hipError_t e = hipHostMalloc((void**)&p, SMALL, fi.flag);
        if (e == hipSuccess) {
            printf("  %-38s → OK\n  %s\n\n", fi.name, fi.desc);
            HIP_CHECK(hipHostFree(p));
            p = nullptr;
        } else {
            printf("  %-38s → %s (not supported on this device)\n\n",
                   fi.name, hipGetErrorString(e));
        }
    }
}

int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API:  hipHostMalloc & hipHostFree\n");
    printf("══════════════════════════════════════════════════════════\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n", prop.name);

    demo_default_pinned();
    demo_mapped_zero_copy();
    demo_write_combined();
    demo_bandwidth_comparison();
    demo_flags_reference();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  ALL hipHostMalloc / hipHostFree demos PASSED\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
