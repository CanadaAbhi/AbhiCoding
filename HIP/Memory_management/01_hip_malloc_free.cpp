// ================================================================
//  01_hip_malloc_free.cpp
//  HIP Memory Management API — hipMalloc & hipFree
//
//  COVERS:
//    hipMalloc(&ptr, bytes)  — allocate device (VRAM) memory
//    hipFree(ptr)            — release device memory
//    hipMalloc error cases   — zero-byte, over-allocation
//    hipMalloc2D             — pitched 2-D allocation
//    hipFree(nullptr)        — safe no-op
//
//  COMPILE:
//    hipcc -O2 01_hip_malloc_free.cpp -o 01_malloc_free
//  RUN:
//    ./01_malloc_free
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

// ── Error-checking macro ─────────────────────────────────────
#define HIP_CHECK(call)                                               \
    do {                                                              \
        hipError_t _e = (call);                                       \
        if (_e != hipSuccess) {                                       \
            fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",            \
                    hipGetErrorString(_e), __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// ── Simple kernel: fill device buffer with index values ──────
__global__ void fillIndex(float* d, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) d[gid] = static_cast<float>(gid);
}

// ================================================================
//  DEMO 1 — Basic hipMalloc / hipFree lifecycle
// ================================================================
static void demo_basic_lifecycle()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 1: Basic hipMalloc / hipFree lifecycle         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int   N     = 1 << 20;          // 1 M floats = 4 MB
    const size_t bytes = N * sizeof(float);

    // ── hipMalloc: allocate N floats on the GPU ───────────────
    //   Signature: hipError_t hipMalloc(void** devPtr, size_t size)
    //   • devPtr is set to the GPU virtual address
    //   • Memory is NOT initialised (contents undefined)
    //   • Returns hipErrorOutOfMemory if VRAM is exhausted
    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    printf("  hipMalloc  : allocated %.2f MB at d_buf=%p\n",
           (double)bytes / (1 << 20), (void*)d_buf);
    assert(d_buf != nullptr);

    // ── Use the allocation ────────────────────────────────────
    int threads = 256, blocks = (N + 255) / 256;
    fillIndex<<<blocks, threads>>>(d_buf, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Verify first few elements
    float h_check[4] = {};
    HIP_CHECK(hipMemcpy(h_check, d_buf, 4 * sizeof(float),
                         hipMemcpyDeviceToHost));
    printf("  fillIndex  : d_buf[0..3] = %.0f %.0f %.0f %.0f\n",
           h_check[0], h_check[1], h_check[2], h_check[3]);

    // ── hipFree: release the device allocation ────────────────
    //   Signature: hipError_t hipFree(void* devPtr)
    //   • Blocks until all pending GPU operations using devPtr finish
    //   • devPtr becomes invalid after this call
    //   • Passing nullptr is a safe no-op
    HIP_CHECK(hipFree(d_buf));
    d_buf = nullptr;              // good practice: null the pointer after free
    printf("  hipFree    : memory released, ptr nulled\n");

    // ── hipFree(nullptr) is always safe ──────────────────────
    HIP_CHECK(hipFree(nullptr));
    printf("  hipFree(nullptr): no-op, returns hipSuccess\n");

    printf("  [PASS] Basic lifecycle\n");
}

// ================================================================
//  DEMO 2 — Multiple allocations, fragmentation, re-use
// ================================================================
static void demo_multiple_allocs()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 2: Multiple allocations & sequential reuse     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int  NALLOC = 5;
    float*     ptrs[NALLOC];
    size_t     sizes[NALLOC] = {
        1 * (1 << 20),   //  1 MB
        4 * (1 << 20),   //  4 MB
        2 * (1 << 20),   //  2 MB
        8 * (1 << 20),   //  8 MB
        1 * (1 << 20),   //  1 MB
    };

    // Allocate all
    for (int i = 0; i < NALLOC; ++i) {
        HIP_CHECK(hipMalloc(&ptrs[i], sizes[i]));
        printf("  [%d] hipMalloc  %2.0f MB  → %p\n",
               i, (double)sizes[i] / (1 << 20), (void*)ptrs[i]);
    }

    // Free in reverse order (avoids some fragmentation)
    printf("  Freeing in reverse order:\n");
    for (int i = NALLOC - 1; i >= 0; --i) {
        HIP_CHECK(hipFree(ptrs[i]));
        printf("  [%d] hipFree    %2.0f MB  ✓\n",
               i, (double)sizes[i] / (1 << 20));
        ptrs[i] = nullptr;
    }

    printf("  [PASS] Multiple allocations\n");
}

// ================================================================
//  DEMO 3 — hipMalloc error handling
// ================================================================
static void demo_error_handling()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 3: hipMalloc error handling                    ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    // ── Try to over-allocate ──────────────────────────────────
    float* d_huge = nullptr;
    size_t absurd = (size_t)512 * 1024 * 1024 * 1024; // 512 GB
    hipError_t err = hipMalloc(&d_huge, absurd);
    if (err == hipErrorOutOfMemory || err != hipSuccess) {
        printf("  Over-alloc 512 GB → hipError: %s  (expected)\n",
               hipGetErrorString(err));
        // IMPORTANT: clear the error so subsequent HIP calls work
        hipGetLastError();   // clears last error on this thread
    } else {
        printf("  Over-alloc somehow succeeded — freeing\n");
        hipFree(d_huge);
    }
    assert(d_huge == nullptr);   // ptr unchanged on failure

    // ── Zero-byte allocation ──────────────────────────────────
    //   Spec says: result is either nullptr or unique non-null ptr
    //   Either way, must be freed.
    float* d_zero = nullptr;
    hipError_t ze = hipMalloc(&d_zero, 0);
    printf("  hipMalloc(0 bytes) → %s  ptr=%p\n",
           hipGetErrorString(ze), (void*)d_zero);
    if (ze == hipSuccess && d_zero != nullptr)
        hipFree(d_zero);

    printf("  [PASS] Error handling\n");
}

// ================================================================
//  DEMO 4 — Pitched / 2-D allocation (hipMallocPitch)
// ================================================================
static void demo_pitched_2d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 4: hipMallocPitch — pitched 2-D allocation     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMallocPitch pads each row to a hardware-aligned boundary
    //  (typically 128 or 256 bytes), improving coalesced access.
    //
    //  Signature:
    //    hipError_t hipMallocPitch(void** devPtr,
    //                              size_t* pitch,   ← actual row stride (bytes)
    //                              size_t  width,   ← logical row width (bytes)
    //                              size_t  height)  ← number of rows

    const size_t WIDTH  = 100;   // 100 floats per row
    const size_t HEIGHT = 200;   // 200 rows
    float* d_mat = nullptr;
    size_t pitch  = 0;

    HIP_CHECK(hipMallocPitch((void**)&d_mat, &pitch,
                               WIDTH * sizeof(float), HEIGHT));

    printf("  hipMallocPitch: logical=%zux%zu floats\n", WIDTH, HEIGHT);
    printf("  Pitch (row stride) = %zu bytes  (%zu floats per row)\n",
           pitch, pitch / sizeof(float));
    printf("  Padding per row    = %zu bytes\n",
           pitch - WIDTH * sizeof(float));
    printf("  Total device bytes = %zu (%.2f KB)\n",
           pitch * HEIGHT, (double)(pitch * HEIGHT) / 1024.0);

    // Access element [row][col] via: (float*)((char*)d_mat + row*pitch) + col
    HIP_CHECK(hipFree(d_mat));
    printf("  [PASS] Pitched allocation\n");
}

// ================================================================
//  DEMO 5 — hipMalloc3D (cuboid allocation)
// ================================================================
static void demo_malloc3d()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 5: hipMalloc3D — 3-D pitched allocation        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  hipMalloc3D allocates a 3-D region with hardware-optimal pitches.
    //  Returns a hipPitchedPtr describing the layout.
    //
    //  Signature:
    //    hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr,
    //                           hipExtent extent)

    hipExtent extent = make_hipExtent(
        64 * sizeof(float),   // width in bytes (64 floats)
        32,                   // height (rows)
        16                    // depth (slices)
    );

    hipPitchedPtr pitched = {};
    HIP_CHECK(hipMalloc3D(&pitched, extent));

    printf("  hipMalloc3D  : 64×32×16 floats\n");
    printf("  pitched.ptr  : %p\n",    pitched.ptr);
    printf("  pitched.pitch: %zu bytes\n", pitched.pitch);
    printf("  pitched.xsize: %zu bytes\n", pitched.xsize);
    printf("  pitched.ysize: %zu rows\n",  pitched.ysize);

    HIP_CHECK(hipFree(pitched.ptr));
    printf("  [PASS] 3-D allocation\n");
}

int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API:  hipMalloc & hipFree\n");
    printf("══════════════════════════════════════════════════════════\n");

    // Print device info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  |  VRAM: %.2f GB\n",
           prop.name, (double)prop.totalGlobalMem / (1 << 30));

    demo_basic_lifecycle();
    demo_multiple_allocs();
    demo_error_handling();
    demo_pitched_2d();
    demo_malloc3d();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  ALL hipMalloc / hipFree demos PASSED\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
