// ============================================================
//  06_warpsize_atomics.cpp
//  Demonstrates: warpSize (64 on AMD vs 32 NVIDIA)
//                atomicAdd  atomicSub  atomicMin  atomicMax
//                atomicAnd  atomicOr   atomicXor
//                atomicCAS  atomicExch
//                64-bit wavefront ballot (__ballot64)
//                warp-level reduction using shuffle
//
//  Compile:
//    hipcc -O2 06_warpsize_atomics.cpp -o 06_demo
//  Run:
//    ./06_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdint>

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── Kernel 1: atomicAdd — global histogram ────────────────────
__global__ void histogramKernel(const int* data, int* hist, int n, int bins)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int bin = data[gid] % bins;
        // atomicAdd: thread-safe increment — no race condition
        atomicAdd(&hist[bin], 1);
    }
}

// ── Kernel 2: all integer atomics ────────────────────────────
__global__ void allAtomics(int* results)
{
    // Only one thread does all ops to make output deterministic
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // atomicAdd: results[0] += 5  (initial=0 → 5)
    results[0] = 0;
    atomicAdd(&results[0], 5);                 // 5

    // atomicSub: results[1] -= 3  (initial=10 → 7)
    results[1] = 10;
    atomicSub(&results[1], 3);                 // 7

    // atomicMin
    results[2] = 100;
    atomicMin(&results[2], 42);                // 42

    // atomicMax
    results[3] = 0;
    atomicMax(&results[3], 77);                // 77

    // atomicAnd: bitwise AND
    results[4] = 0xFF;
    atomicAnd(&results[4], 0x0F);              // 0x0F = 15

    // atomicOr
    results[5] = 0xF0;
    atomicOr(&results[5], 0x0F);              // 0xFF = 255

    // atomicXor
    results[6] = 0xFF;
    atomicXor(&results[6], 0x55);             // 0xAA = 170

    // atomicExch: swap value, return old
    results[7] = 1234;
    int old7 = atomicExch(&results[7], 5678); // results[7]=5678, old7=1234
    results[8] = old7;                         // store old

    // atomicCAS: compare-and-swap
    // atomicCAS(addr, compare, val) → if *addr==compare: *addr=val; return old
    results[9] = 999;
    int old9 = atomicCAS(&results[9], 999, 42); // matches → sets 42
    results[10] = old9;                           // old9=999

    results[11] = 999;
    int old11 = atomicCAS(&results[11], 0, 42);  // no match (999≠0) → unchanged
    results[12] = old11;                           // old11=999
}

// ── Kernel 3: float atomicAdd ─────────────────────────────────
__global__ void floatAtomicAdd(float* sum, const float* data, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n)
        atomicAdd(sum, data[gid]);   // float atomicAdd supported on all ROCm GPUs
}

// ── Kernel 4: warpSize and wavefront ballot ────────────────────
__global__ void warpSizeDemo(int* out_warpsize, uint64_t* out_ballot,
                              int* out_active)
{
    // warpSize is a built-in — equals 64 on AMD, 32 on NVIDIA
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out_warpsize[0] = warpSize;    // read built-in

    // __ballot64: returns 64-bit mask of which lanes satisfy condition
    // (CUDA uses __ballot_sync with 32-bit mask)
    uint64_t mask = __ballot64(threadIdx.x % 2 == 0);  // even threads
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_ballot[0] = mask;
        out_active[0] = __popcll(mask);   // count of 1-bits
    }
}

// ── Kernel 5: warp-level reduction using shuffle ──────────────
// Performs reduction within a single wavefront (no shared memory needed)
__global__ void warpReduce(const float* in, float* out, int n)
{
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < n) ? in[gid] : 0.f;

    // AMD wavefront = 64 lanes, so we do 6 rounds (2^6 = 64)
    // NVIDIA warp = 32 lanes → only 5 rounds needed
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down(val, offset);

    // Lane 0 of each wavefront holds the wavefront sum
    if (threadIdx.x % warpSize == 0)
        atomicAdd(out, val);
}

// ── Kernel 6: implement spinlock with atomicCAS ───────────────
__global__ void spinlockDemo(int* lock, int* counter, int n)
{
    // Each thread increments counter inside a critical section
    // protected by a GPU spinlock (educational — avoid in real code!)
    for (int i = 0; i < n; ++i) {
        // Acquire: spin until we CAS 0→1
        while (atomicCAS(lock, 0, 1) != 0) { /* spin */ }
        (*counter)++;   // critical section
        atomicExch(lock, 0);  // Release: set lock back to 0
    }
}

int main()
{
    // ── Demo 1: warpSize ──────────────────────────────────────
    printf("=== 1. warpSize built-in (64 on AMD, 32 on NVIDIA) ===\n");

    int* d_warpsize = nullptr;
    uint64_t* d_ballot = nullptr;
    int* d_active = nullptr;
    HIP_CHECK(hipMalloc(&d_warpsize, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_ballot,   sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&d_active,   sizeof(int)));

    warpSizeDemo<<<1, 64>>>(d_warpsize, d_ballot, d_active);
    HIP_CHECK(hipDeviceSynchronize());

    int    h_ws     = 0;
    uint64_t h_bal  = 0;
    int    h_active = 0;
    HIP_CHECK(hipMemcpy(&h_ws,     d_warpsize, sizeof(int),      hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_bal,    d_ballot,   sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_active, d_active,   sizeof(int),      hipMemcpyDeviceToHost));

    printf("  warpSize = %d  (64 = AMD wavefront, 32 = NVIDIA warp)\n", h_ws);
    printf("  __ballot64(threadIdx.x%%2==0) = 0x%016llx\n",
           (unsigned long long)h_bal);
    printf("  Active even lanes = %d  (expect 32 out of 64)\n", h_active);

    // ── Demo 2: all integer atomics ───────────────────────────
    printf("\n=== 2. All integer atomic operations ===\n");
    const int NRES = 16;
    int* d_results = nullptr;
    HIP_CHECK(hipMalloc(&d_results, NRES * sizeof(int)));
    HIP_CHECK(hipMemset(d_results, 0, NRES * sizeof(int)));

    allAtomics<<<1, 64>>>(d_results);
    HIP_CHECK(hipDeviceSynchronize());

    int h_results[NRES] = {};
    HIP_CHECK(hipMemcpy(h_results, d_results, NRES * sizeof(int),
                         hipMemcpyDeviceToHost));

    printf("  atomicAdd(0,  +5)       = %d  (expect 5)\n",   h_results[0]);
    printf("  atomicSub(10, -3)       = %d  (expect 7)\n",   h_results[1]);
    printf("  atomicMin(100, 42)      = %d  (expect 42)\n",  h_results[2]);
    printf("  atomicMax(0,  77)       = %d  (expect 77)\n",  h_results[3]);
    printf("  atomicAnd(0xFF,0x0F)    = %d  (expect 15)\n",  h_results[4]);
    printf("  atomicOr (0xF0,0x0F)    = %d  (expect 255)\n", h_results[5]);
    printf("  atomicXor(0xFF,0x55)    = %d  (expect 170)\n", h_results[6]);
    printf("  atomicExch(1234→5678)   = %d  old=%d\n",
           h_results[7], h_results[8]);
    printf("  atomicCAS match   999→42= %d  old=%d\n",
           h_results[9], h_results[10]);
    printf("  atomicCAS nomatch 999   = %d  old=%d\n",
           h_results[11], h_results[12]);

    // ── Demo 3: histogram with atomicAdd ─────────────────────
    printf("\n=== 3. Histogram using atomicAdd ===\n");
    const int NDATA = 1 << 20;
    const int BINS  = 8;
    int* h_data = new int[NDATA];
    for (int i = 0; i < NDATA; ++i) h_data[i] = i;   // 0,1,...,N-1

    int* d_data = nullptr;
    int* d_hist = nullptr;
    HIP_CHECK(hipMalloc(&d_data, NDATA * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_hist, BINS  * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_data, h_data, NDATA * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_hist, 0, BINS * sizeof(int)));

    int threads = 256;
    int blocks  = (NDATA + threads - 1) / threads;
    histogramKernel<<<blocks, threads>>>(d_data, d_hist, NDATA, BINS);
    HIP_CHECK(hipDeviceSynchronize());

    int h_hist[BINS] = {};
    HIP_CHECK(hipMemcpy(h_hist, d_hist, BINS * sizeof(int), hipMemcpyDeviceToHost));
    int expected_per_bin = NDATA / BINS;
    printf("  %d elements into %d bins (expect ~%d each):\n", NDATA, BINS, expected_per_bin);
    bool hist_ok = true;
    for (int b = 0; b < BINS; ++b) {
        printf("    bin[%d] = %d %s\n", b, h_hist[b],
               h_hist[b] == expected_per_bin ? "✓" : "✗");
        if (h_hist[b] != expected_per_bin) hist_ok = false;
    }
    printf("  Histogram: %s\n", hist_ok ? "PASS" : "FAIL");

    // ── Demo 4: float atomicAdd for parallel sum ──────────────
    printf("\n=== 4. float atomicAdd — parallel dot product ===\n");
    const int NS = 1 << 20;
    float* h_fdata = new float[NS];
    for (int i = 0; i < NS; ++i) h_fdata[i] = 1.0f;

    float* d_fdata = nullptr;
    float* d_sum   = nullptr;
    HIP_CHECK(hipMalloc(&d_fdata, NS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sum,   sizeof(float)));
    HIP_CHECK(hipMemcpy(d_fdata, h_fdata, NS * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_sum, 0, sizeof(float)));

    floatAtomicAdd<<<blocks, threads>>>(d_sum, d_fdata, NS);
    HIP_CHECK(hipDeviceSynchronize());

    float h_sum = 0.f;
    HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost));
    printf("  Sum of %d ones = %.0f  (expect %d)  %s\n",
           NS, h_sum, NS, (fabsf(h_sum - NS) < 1.f) ? "PASS" : "FAIL");

    // ── Demo 5: warp-level shuffle reduction ──────────────────
    printf("\n=== 5. Warp reduction via __shfl_down (no shared mem) ===\n");
    for (int i = 0; i < NS; ++i) h_fdata[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_fdata, h_fdata, NS * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_sum, 0, sizeof(float)));

    warpReduce<<<blocks, threads>>>(d_fdata, d_sum, NS);
    HIP_CHECK(hipDeviceSynchronize());

    h_sum = 0.f;
    HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost));
    printf("  Warp-reduce sum = %.0f  (expect %d)  %s\n",
           h_sum, NS, (fabsf(h_sum - NS) < 2.f) ? "PASS" : "FAIL");

    // ── Cleanup ───────────────────────────────────────────────
    delete[] h_data; delete[] h_fdata;
    HIP_CHECK(hipFree(d_warpsize)); HIP_CHECK(hipFree(d_ballot));
    HIP_CHECK(hipFree(d_active));   HIP_CHECK(hipFree(d_results));
    HIP_CHECK(hipFree(d_data));     HIP_CHECK(hipFree(d_hist));
    HIP_CHECK(hipFree(d_fdata));    HIP_CHECK(hipFree(d_sum));

    printf("\nAll warpSize / atomicXxx demos PASSED\n");
    return 0;
}
