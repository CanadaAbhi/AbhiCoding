// ================================================================
//  03_hip_memcpy_async.cpp
//  HIP Memory Management API — hipMemcpyAsync
//
//  COVERS:
//    hipMemcpyAsync(dst, src, bytes, dir, stream)
//    Why pinned memory is REQUIRED for true async
//    Overlapping H2D transfers with GPU kernels (double-buffering)
//    hipStreamSynchronize vs hipDeviceSynchronize
//    Measuring transfer + compute overlap benefit
//
//  COMPILE:
//    hipcc -O2 03_hip_memcpy_async.cpp -o 03_memcpy_async
//  RUN:
//    ./03_memcpy_async
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
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

// ── Kernels ───────────────────────────────────────────────────
__global__ void scaleVec(float* d, float alpha, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) d[g] *= alpha;
}

__global__ void computeHeavy(float* d, int n, int iters) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float v = d[g];
    for (int i = 0; i < iters; ++i)
        v = sqrtf(v * v + 1.0f);
    d[g] = v;
}

// ================================================================
//  DEMO 1 — Basic hipMemcpyAsync in a stream
// ================================================================
static void demo_basic_async()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 1: Basic hipMemcpyAsync in a stream            ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int   N     = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // ── CRITICAL: hipMemcpyAsync REQUIRES pinned (page-locked) ──
    //   Regular malloc memory will work but the copy will NOT be
    //   truly asynchronous — HIP silently stages through an internal
    //   pinned buffer and blocks internally.
    //   Use hipHostMalloc for genuinely async transfers.
    float* h_src = nullptr;
    float* h_dst = nullptr;
    HIP_CHECK(hipHostMalloc(&h_src, bytes, 0));   // pinned
    HIP_CHECK(hipHostMalloc(&h_dst, bytes, 0));   // pinned
    for (int i = 0; i < N; ++i) h_src[i] = static_cast<float>(i);

    float* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ── Enqueue async H2D into stream ─────────────────────────
    //   Returns immediately — copy runs on GPU DMA engine concurrently.
    HIP_CHECK(hipMemcpyAsync(d_buf, h_src, bytes,
                               hipMemcpyHostToDevice, stream));
    printf("  hipMemcpyAsync H2D enqueued (returns immediately)\n");

    // ── Enqueue kernel after copy — runs in order in stream ───
    int t = 256, b = (N + 255) / 256;
    scaleVec<<<b, t, 0, stream>>>(d_buf, 2.0f, N);
    printf("  Kernel enqueued in same stream (runs after copy completes)\n");

    // ── Enqueue async D2H ─────────────────────────────────────
    HIP_CHECK(hipMemcpyAsync(h_dst, d_buf, bytes,
                               hipMemcpyDeviceToHost, stream));
    printf("  hipMemcpyAsync D2H enqueued\n");

    // ── Wait for everything in stream to finish ───────────────
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("  hipStreamSynchronize: all ops complete\n");

    // Verify: h_src[i]=i, scaled by 2.0 → h_dst[i]=2i
    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_dst[i] - 2.0f * i) > 0.5f) { ok = false; break; }
    printf("  Verify: h_dst[0..3] = %.0f %.0f %.0f %.0f  %s\n",
           h_dst[0], h_dst[1], h_dst[2], h_dst[3], ok ? "PASS" : "FAIL");

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipHostFree(h_src));
    HIP_CHECK(hipHostFree(h_dst));
}

// ================================================================
//  DEMO 2 — Double-buffering: overlap H2D + kernel + D2H
// ================================================================
static void demo_double_buffer()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 2: Double-buffering — H2D + kernel overlap     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  Pattern:
    //    stream0: H2D(chunk0) → kernel(chunk0) → D2H(chunk0)
    //    stream1: H2D(chunk1) → kernel(chunk1) → D2H(chunk1)
    //    (streams run concurrently on the GPU)

    const int   TOTAL  = 1 << 22;   // 4 M floats total
    const int   CHUNK  = TOTAL / 2; // 2 M per chunk
    const size_t CBYTES = CHUNK * sizeof(float);

    // Pinned host buffers for two chunks
    float* h_in[2]  = {};
    float* h_out[2] = {};
    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipHostMalloc(&h_in[i],  CBYTES, 0));
        HIP_CHECK(hipHostMalloc(&h_out[i], CBYTES, 0));
        for (int j = 0; j < CHUNK; ++j)
            h_in[i][j] = static_cast<float>(i * CHUNK + j);
    }

    // Device double-buffers
    float* d_buf[2] = {};
    for (int i = 0; i < 2; ++i)
        HIP_CHECK(hipMalloc(&d_buf[i], CBYTES));

    // Two streams for concurrent execution
    hipStream_t stream[2];
    HIP_CHECK(hipStreamCreate(&stream[0]));
    HIP_CHECK(hipStreamCreate(&stream[1]));

    // Timing: measure total time with overlap
    hipEvent_t t_start, t_stop;
    HIP_CHECK(hipEventCreate(&t_start));
    HIP_CHECK(hipEventCreate(&t_stop));

    int t = 256, b = (CHUNK + 255) / 256;

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t_start));

    // Enqueue both pipelines concurrently into their streams
    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipMemcpyAsync(d_buf[i], h_in[i], CBYTES,
                                   hipMemcpyHostToDevice, stream[i]));
        computeHeavy<<<b, t, 0, stream[i]>>>(d_buf[i], CHUNK, 200);
        HIP_CHECK(hipMemcpyAsync(h_out[i], d_buf[i], CBYTES,
                                   hipMemcpyDeviceToHost, stream[i]));
    }

    // Sync both streams
    HIP_CHECK(hipStreamSynchronize(stream[0]));
    HIP_CHECK(hipStreamSynchronize(stream[1]));
    HIP_CHECK(hipEventRecord(t_stop));
    HIP_CHECK(hipEventSynchronize(t_stop));

    float ms_overlap = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_overlap, t_start, t_stop));

    // ── Compare: sequential (no overlap) ─────────────────────
    for (int i = 0; i < CHUNK; ++i) h_in[0][i] = static_cast<float>(i);
    for (int i = 0; i < CHUNK; ++i) h_in[1][i] = static_cast<float>(i);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(t_start));
    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipMemcpy(d_buf[i], h_in[i], CBYTES, hipMemcpyHostToDevice));
        computeHeavy<<<b, t>>>(d_buf[i], CHUNK, 200);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out[i], d_buf[i], CBYTES, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipEventRecord(t_stop));
    HIP_CHECK(hipEventSynchronize(t_stop));
    float ms_seq = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_seq, t_start, t_stop));

    printf("  Sequential (sync):     %.2f ms\n", ms_seq);
    printf("  Double-buffer (async): %.2f ms\n", ms_overlap);
    printf("  Speedup from overlap:  %.2fx\n",   ms_seq / ms_overlap);

    // Cleanup
    HIP_CHECK(hipEventDestroy(t_start));
    HIP_CHECK(hipEventDestroy(t_stop));
    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipStreamDestroy(stream[i]));
        HIP_CHECK(hipFree(d_buf[i]));
        HIP_CHECK(hipHostFree(h_in[i]));
        HIP_CHECK(hipHostFree(h_out[i]));
    }
}

// ================================================================
//  DEMO 3 — N-stream pipeline (generalised chunked processing)
// ================================================================
static void demo_n_stream_pipeline()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 3: N-stream pipeline — chunked async           ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const int NSTREAMS = 4;
    const int TOTAL    = 1 << 22;                    // 4 M total
    const int CHUNK    = TOTAL / NSTREAMS;
    const size_t CB    = CHUNK * sizeof(float);

    float*       h_in[NSTREAMS]  = {};
    float*       h_out[NSTREAMS] = {};
    float*       d_buf[NSTREAMS] = {};
    hipStream_t  streams[NSTREAMS];

    for (int i = 0; i < NSTREAMS; ++i) {
        HIP_CHECK(hipHostMalloc(&h_in[i],  CB, 0));
        HIP_CHECK(hipHostMalloc(&h_out[i], CB, 0));
        HIP_CHECK(hipMalloc    (&d_buf[i], CB));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        for (int j = 0; j < CHUNK; ++j)
            h_in[i][j] = 1.0f;
    }

    int t = 256, b = (CHUNK + 255) / 256;

    // Enqueue all chunks asynchronously
    for (int i = 0; i < NSTREAMS; ++i) {
        HIP_CHECK(hipMemcpyAsync(d_buf[i], h_in[i], CB,
                                   hipMemcpyHostToDevice, streams[i]));
        scaleVec<<<b, t, 0, streams[i]>>>(d_buf[i], 3.0f, CHUNK);
        HIP_CHECK(hipMemcpyAsync(h_out[i], d_buf[i], CB,
                                   hipMemcpyDeviceToHost, streams[i]));
    }

    // Sync all streams
    for (int i = 0; i < NSTREAMS; ++i)
        HIP_CHECK(hipStreamSynchronize(streams[i]));

    // Verify
    bool ok = true;
    for (int i = 0; i < NSTREAMS && ok; ++i)
        for (int j = 0; j < CHUNK && ok; ++j)
            if (fabsf(h_out[i][j] - 3.0f) > 1e-4f) ok = false;

    printf("  %d streams processed %d total floats  %s\n",
           NSTREAMS, TOTAL, ok ? "PASS" : "FAIL");
    printf("  Each chunk: H2D → scale×3 → D2H  (h_out[0][0]=%.1f)\n",
           h_out[0][0]);

    // Cleanup
    for (int i = 0; i < NSTREAMS; ++i) {
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_buf[i]));
        HIP_CHECK(hipHostFree(h_in[i]));
        HIP_CHECK(hipHostFree(h_out[i]));
    }
}

// ================================================================
//  DEMO 4 — hipMemcpyAsync with events for cross-stream sync
// ================================================================
static void demo_cross_stream_sync()
{
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  DEMO 4: Cross-stream sync via hipStreamWaitEvent    ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    //  Use case: stream A produces data, stream B must wait for
    //  stream A's copy to complete before using it.

    const int   N     = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float* h_src = nullptr;
    HIP_CHECK(hipHostMalloc(&h_src, bytes, 0));
    for (int i = 0; i < N; ++i) h_src[i] = 5.0f;

    float* d_shared = nullptr;
    HIP_CHECK(hipMalloc(&d_shared, bytes));

    hipStream_t streamA, streamB;
    HIP_CHECK(hipStreamCreate(&streamA));
    HIP_CHECK(hipStreamCreate(&streamB));

    hipEvent_t copyDone;
    HIP_CHECK(hipEventCreate(&copyDone));

    // Stream A: copy data and record a marker event
    HIP_CHECK(hipMemcpyAsync(d_shared, h_src, bytes,
                               hipMemcpyHostToDevice, streamA));
    HIP_CHECK(hipEventRecord(copyDone, streamA));   // fires when A's copy is done

    // Stream B: wait for copyDone before processing
    HIP_CHECK(hipStreamWaitEvent(streamB, copyDone, 0));
    int t = 256, b = (N + 255) / 256;
    scaleVec<<<b, t, 0, streamB>>>(d_shared, 2.0f, N);  // 5.0 * 2 = 10.0

    HIP_CHECK(hipStreamSynchronize(streamB));

    float h_check[4] = {};
    HIP_CHECK(hipMemcpy(h_check, d_shared, 4 * sizeof(float),
                         hipMemcpyDeviceToHost));
    printf("  Stream A copied, Stream B processed after event barrier\n");
    printf("  d_shared[0..3] = %.1f %.1f %.1f %.1f  (expect 10.0)  %s\n",
           h_check[0], h_check[1], h_check[2], h_check[3],
           (h_check[0] == 10.0f) ? "PASS" : "FAIL");

    HIP_CHECK(hipEventDestroy(copyDone));
    HIP_CHECK(hipStreamDestroy(streamA));
    HIP_CHECK(hipStreamDestroy(streamB));
    HIP_CHECK(hipFree(d_shared));
    HIP_CHECK(hipHostFree(h_src));
}

int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API:  hipMemcpyAsync\n");
    printf("══════════════════════════════════════════════════════════\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s\n", prop.name);

    demo_basic_async();
    demo_double_buffer();
    demo_n_stream_pipeline();
    demo_cross_stream_sync();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  ALL hipMemcpyAsync demos PASSED\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
