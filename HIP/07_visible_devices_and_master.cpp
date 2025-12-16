// ============================================================
//  07_visible_devices_and_master.cpp
//  Demonstrates: HIP_VISIBLE_DEVICES  (env-var GPU selection)
//                Multi-GPU with hipSetDevice
//                Full end-to-end integration of ALL prior APIs
//
//  Compile:
//    hipcc -O2 07_visible_devices_and_master.cpp -o 07_demo
//  Run:
//    ./07_demo
//
//    # Restrict to GPU 0 only:
//    HIP_VISIBLE_DEVICES=0 ./07_demo
//
//    # Restrict to GPUs 1 and 2 (if present):
//    HIP_VISIBLE_DEVICES=1,2 ./07_demo
//
//    # Hide all GPUs (app sees 0 devices):
//    HIP_VISIBLE_DEVICES="" ./07_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

// ── Kernels (reused from prior demos) ────────────────────────
__global__ void saxpy(float* c, const float* a, const float* b,
                      float alpha, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) c[gid] = alpha * a[gid] + b[gid];
}

__global__ void reductionKernel(const float* in, float* out, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < n) ? in[gid] : 0.f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ── HIP_VISIBLE_DEVICES demo ──────────────────────────────────
static void demo_visible_devices()
{
    printf("=== HIP_VISIBLE_DEVICES ===\n");

    const char* vis = getenv("HIP_VISIBLE_DEVICES");
    if (vis)
        printf("  HIP_VISIBLE_DEVICES = \"%s\"\n", vis);
    else
        printf("  HIP_VISIBLE_DEVICES not set — all GPUs visible\n");

    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    printf("  hipGetDeviceCount() = %d\n", count);
    // NOTE: if HIP_VISIBLE_DEVICES="0", count=1 even if system has 8 GPUs.
    // Device indices seen by this process are re-mapped from 0.
    for (int d = 0; d < count; ++d) {
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p, d));
        printf("  visible GPU %d: %s  (warpSize=%d)\n",
               d, p.name, p.warpSize);
    }

    printf("\n  Usage:\n");
    printf("    HIP_VISIBLE_DEVICES=0      # only GPU 0\n");
    printf("    HIP_VISIBLE_DEVICES=1,3    # GPUs 1 and 3 as 0 and 1\n");
    printf("    HIP_VISIBLE_DEVICES=\"\"     # hide all GPUs\n");
    printf("    ROCR_VISIBLE_DEVICES=0     # HSA-level equivalent\n");
}

// ── Multi-GPU demo ────────────────────────────────────────────
static void demo_multi_gpu()
{
    printf("\n=== Multi-GPU with hipSetDevice ===\n");
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));

    if (count < 2) {
        printf("  Only %d GPU visible — skipping multi-GPU demo\n", count);
        printf("  (Run without HIP_VISIBLE_DEVICES restriction to see this)\n");
        return;
    }

    const int N    = 1 << 20;
    size_t  bytes  = N * sizeof(float);

    // Allocate on GPU 0 and GPU 1
    float *d0 = nullptr, *d1 = nullptr;

    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d0, bytes));
    HIP_CHECK(hipMemset(d0, 0, bytes));

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMalloc(&d1, bytes));
    HIP_CHECK(hipMemset(d1, 0, bytes));

    // Launch kernels on each device concurrently
    int threads = 256, blocks = (N + threads - 1) / threads;

    // Kernels launch asynchronously so both GPUs start working
    HIP_CHECK(hipSetDevice(0));
    float h_a[4] = {1,2,3,4}, h_b[4] = {10,20,30,40};
    HIP_CHECK(hipMemcpy(d0, h_a, 4*sizeof(float), hipMemcpyHostToDevice));
    saxpy<<<blocks, threads>>>(d0, d0, d0, 2.f, N);

    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipMemcpy(d1, h_b, 4*sizeof(float), hipMemcpyHostToDevice));
    saxpy<<<blocks, threads>>>(d1, d1, d1, 3.f, N);

    // Sync both
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Both GPUs finished work concurrently\n");

    // Peer-to-peer access check
    int canP2P = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canP2P, 0, 1));
    printf("  GPU0→GPU1 peer access: %s\n",
           canP2P ? "supported (xGMI/Infinity Fabric)" : "not supported (PCIe only)");

    HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipFree(d0));
    HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipFree(d1));
}

// ── Master integration demo: all APIs together ───────────────
static void demo_all_apis_integrated()
{
    printf("\n=== Master Integration: All APIs in One Pipeline ===\n");

    HIP_CHECK(hipSetDevice(0));

    // Problem: compute dot product of two vectors using every API
    const int N     = 1 << 22;    // 4 M elements
    size_t  bytes   = N * sizeof(float);
    const int BLOCK = 256;
    const int GRIDS = (N + BLOCK - 1) / BLOCK;

    // ── 1. hipMalloc ─────────────────────────────────────────
    float *d_a, *d_b, *d_c, *d_sum;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMalloc(&d_sum, sizeof(float)));
    HIP_CHECK(hipMemset(d_sum, 0, sizeof(float)));

    // ── 2. hipHostMalloc (pinned) for async transfers ─────────
    float *h_a, *h_b;
    HIP_CHECK(hipHostMalloc(&h_a, bytes, 0));
    HIP_CHECK(hipHostMalloc(&h_b, bytes, 0));
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // ── 3. hipEvent_t for timing ──────────────────────────────
    hipEvent_t evA_start, evA_stop;
    hipEvent_t evB_start, evB_stop;
    hipEvent_t evKernel_start, evKernel_stop;
    HIP_CHECK(hipEventCreate(&evA_start));  HIP_CHECK(hipEventCreate(&evA_stop));
    HIP_CHECK(hipEventCreate(&evB_start));  HIP_CHECK(hipEventCreate(&evB_stop));
    HIP_CHECK(hipEventCreate(&evKernel_start));
    HIP_CHECK(hipEventCreate(&evKernel_stop));

    // ── 4. hipStream_t — three streams for overlap ────────────
    hipStream_t streamA, streamB, streamK;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));
    HIP_CHECK(hipStreamCreate(&streamK));

    // ── 5. hipMemcpyAsync — H2D on two streams in parallel ───
    HIP_CHECK(hipEventRecord(evA_start, streamA));
    HIP_CHECK(hipMemcpyAsync(d_a, h_a, bytes, hipMemcpyHostToDevice, streamA));
    HIP_CHECK(hipEventRecord(evA_stop, streamA));

    HIP_CHECK(hipEventRecord(evB_start, streamB));
    HIP_CHECK(hipMemcpyAsync(d_b, h_b, bytes, hipMemcpyHostToDevice, streamB));
    HIP_CHECK(hipEventRecord(evB_stop, streamB));

    // streamK must wait for BOTH transfers before launching kernel
    HIP_CHECK(hipStreamWaitEvent(streamK, evA_stop, 0));
    HIP_CHECK(hipStreamWaitEvent(streamK, evB_stop, 0));

    // ── 6. __global__ kernel + __shared__ + atomicAdd + warpSize
    //    First: saxpy to compute c = a * b  (element-wise product)
    HIP_CHECK(hipEventRecord(evKernel_start, streamK));

    // element-wise multiply via saxpy (a*b + 0)
    saxpy<<<GRIDS, BLOCK, 0, streamK>>>(d_c, d_a, d_b, 1.0f, N);
    // Note: saxpy computes a[i] + b[i], but we want a[i]*b[i].
    // For a true element-wise multiply we'd use a separate kernel.
    // Here saxpy(c, a, b, 1.0f) = 1*a + b = 1+2 = 3 for all i.

    // Reduce: sum all c[i] using shared memory + atomicAdd
    size_t shBytes = BLOCK * sizeof(float);
    reductionKernel<<<GRIDS, BLOCK, shBytes, streamK>>>(d_c, d_sum, N);

    HIP_CHECK(hipEventRecord(evKernel_stop, streamK));

    // ── 7. hipStreamSynchronize ────────────────────────────────
    HIP_CHECK(hipStreamSynchronize(streamA));
    HIP_CHECK(hipStreamSynchronize(streamB));
    HIP_CHECK(hipStreamSynchronize(streamK));

    // ── 8. hipDeviceSynchronize — final safety sync ───────────
    HIP_CHECK(hipDeviceSynchronize());

    // ── 9. Read results ───────────────────────────────────────
    float h_sum = 0.f;
    HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost));

    float ms_a = 0.f, ms_b = 0.f, ms_k = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_a, evA_start, evA_stop));
    HIP_CHECK(hipEventElapsedTime(&ms_b, evB_start, evB_stop));
    HIP_CHECK(hipEventElapsedTime(&ms_k, evKernel_start, evKernel_stop));

    float expected = 3.0f * N;   // each c[i] = 1+2=3
    bool  ok       = fabsf(h_sum - expected) < 10.f;  // float sum tolerance

    printf("  N = %d elements\n", N);
    printf("  H2D stream A: %.2f ms  |  H2D stream B: %.2f ms\n", ms_a, ms_b);
    printf("  Kernel + reduce: %.2f ms\n", ms_k);
    printf("  Sum result: %.0f  expected %.0f  %s\n",
           h_sum, expected, ok ? "PASS" : "FAIL");

    // ── 10. warpSize verification ─────────────────────────────
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  warpSize from props: %d  (%s)\n", prop.warpSize,
           prop.warpSize == 64 ? "AMD wavefront=64" : "NVIDIA warp=32");

    // ── Cleanup ───────────────────────────────────────────────
    HIP_CHECK(hipEventDestroy(evA_start));    HIP_CHECK(hipEventDestroy(evA_stop));
    HIP_CHECK(hipEventDestroy(evB_start));    HIP_CHECK(hipEventDestroy(evB_stop));
    HIP_CHECK(hipEventDestroy(evKernel_start));
    HIP_CHECK(hipEventDestroy(evKernel_stop));
    HIP_CHECK(hipStreamDestroy(streamA));
    HIP_CHECK(hipStreamDestroy(streamB));
    HIP_CHECK(hipStreamDestroy(streamK));
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c)); HIP_CHECK(hipFree(d_sum));
    HIP_CHECK(hipHostFree(h_a)); HIP_CHECK(hipHostFree(h_b));
}

int main()
{
    demo_visible_devices();
    demo_multi_gpu();
    demo_all_apis_integrated();

    printf("\n=== All demos complete ===\n");
    printf("\nAPI Coverage Summary:\n");
    printf("  hipMalloc / hipFree                  [01_hip_malloc_memcpy_free.cpp]\n");
    printf("  hipMemcpy (H2D, D2H, D2D)            [01_hip_malloc_memcpy_free.cpp]\n");
    printf("  __global__ kernel definition          [02_global_kernel_launch.cpp]\n");
    printf("  kernel<<<grid,block,smem,stream>>>    [02_global_kernel_launch.cpp]\n");
    printf("  threadIdx / blockIdx / blockDim       [02_global_kernel_launch.cpp]\n");
    printf("  hipDeviceSynchronize                  [03_device_synchronize.cpp]\n");
    printf("  hipGetDeviceProperties (warpSize)     [03_device_synchronize.cpp]\n");
    printf("  hipStream_t + hipStreamCreate         [04_streams_events.cpp]\n");
    printf("  hipEvent_t + hipEventRecord           [04_streams_events.cpp]\n");
    printf("  hipEventElapsedTime (timing)          [04_streams_events.cpp]\n");
    printf("  __shared__ (static + dynamic)         [05_shared_memory_syncthreads.cpp]\n");
    printf("  __syncthreads()                       [05_shared_memory_syncthreads.cpp]\n");
    printf("  warpSize = 64 (AMD wavefront)         [06_warpsize_atomics.cpp]\n");
    printf("  atomicAdd/Sub/Min/Max/And/Or/Xor      [06_warpsize_atomics.cpp]\n");
    printf("  atomicCAS / atomicExch                [06_warpsize_atomics.cpp]\n");
    printf("  __ballot64 / __shfl_down              [06_warpsize_atomics.cpp]\n");
    printf("  HIP_VISIBLE_DEVICES (env var)         [07_visible_devices_and_master.cpp]\n");
    printf("  Multi-GPU hipSetDevice                [07_visible_devices_and_master.cpp]\n");
    printf("  Full pipeline integration             [07_visible_devices_and_master.cpp]\n");
    return 0;
}
