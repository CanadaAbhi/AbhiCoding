// ============================================================
//  04_streams_events.cpp
//  Demonstrates: hipStream_t  hipEvent_t  async operations
//                hipStreamCreate  hipStreamSynchronize
//                hipEventCreate   hipEventRecord
//                hipEventElapsedTime  overlap H2D + kernel
//
//  Compile:
//    hipcc -O2 04_streams_events.cpp -o 04_demo
//  Run:
//    ./04_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── kernels ───────────────────────────────────────────────────
__global__ void scaleKernel(float* data, float alpha, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] *= alpha;
}

__global__ void addConstKernel(float* data, float val, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] += val;
}

// ── workload: fill chunk, process, read back ──────────────────
static void processChunk(hipStream_t stream,
                          float* d_buf, float* h_src, float* h_dst,
                          int n, float alpha, float addVal)
{
    size_t bytes  = n * sizeof(float);
    int threads   = 256;
    int blocks    = (n + threads - 1) / threads;

    // async H->D in this stream
    HIP_CHECK(hipMemcpyAsync(d_buf, h_src, bytes,
                              hipMemcpyHostToDevice, stream));
    // kernel in same stream — runs AFTER the copy in this stream
    scaleKernel<<<blocks, threads, 0, stream>>>(d_buf, alpha, n);
    addConstKernel<<<blocks, threads, 0, stream>>>(d_buf, addVal, n);
    // async D->H
    HIP_CHECK(hipMemcpyAsync(h_dst, d_buf, bytes,
                              hipMemcpyDeviceToHost, stream));
}

int main()
{
    // ── 1. hipStream_t basics ─────────────────────────────────
    printf("=== 1. hipStream_t — creating and using streams ===\n");

    hipStream_t stream0, stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream0));
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));
    printf("  Created 3 streams: stream0=%p  stream1=%p  stream2=%p\n",
           (void*)stream0, (void*)stream1, (void*)stream2);

    // ── 2. hipEvent_t basics and timing ──────────────────────
    printf("\n=== 2. hipEvent_t — kernel timing ===\n");

    hipEvent_t evStart, evStop;
    HIP_CHECK(hipEventCreate(&evStart));
    HIP_CHECK(hipEventCreate(&evStop));

    const int N = 1 << 22;   // 4 M floats
    size_t bytes = N * sizeof(float);

    float* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, bytes));
    HIP_CHECK(hipMemset(d_data, 0, bytes));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // Record start event (in default stream)
    HIP_CHECK(hipEventRecord(evStart, 0));

    scaleKernel<<<blocks, threads>>>(d_data, 2.0f, N);
    addConstKernel<<<blocks, threads>>>(d_data, 1.0f, N);

    // Record stop event, then wait for it
    HIP_CHECK(hipEventRecord(evStop, 0));
    HIP_CHECK(hipEventSynchronize(evStop));   // CPU blocks until evStop fires

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, evStart, evStop));
    printf("  Two kernels on %d elements took %.3f ms (via hipEvent)\n", N, ms);

    // ── 3. Stream-specific event timing ──────────────────────
    printf("\n=== 3. Per-stream event timing ===\n");

    hipEvent_t s1Start, s1Stop;
    HIP_CHECK(hipEventCreate(&s1Start));
    HIP_CHECK(hipEventCreate(&s1Stop));

    // All four ops below are enqueued into stream1, run in order
    HIP_CHECK(hipMemsetAsync(d_data, 0, bytes, stream1));
    HIP_CHECK(hipEventRecord(s1Start, stream1));
    scaleKernel<<<blocks, threads, 0, stream1>>>(d_data, 3.14f, N);
    HIP_CHECK(hipEventRecord(s1Stop, stream1));

    HIP_CHECK(hipStreamSynchronize(stream1));   // wait for stream1 only
    float ms1 = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms1, s1Start, s1Stop));
    printf("  scaleKernel in stream1: %.3f ms\n", ms1);

    // ── 4. Two-stream overlap: copy + compute ─────────────────
    printf("\n=== 4. Two-stream overlap: H2D copy || kernel ===\n");
    const int CHUNK = 1 << 20;   // 1 M per chunk
    size_t chunkBytes = CHUNK * sizeof(float);

    float* h_A = nullptr; float* h_B = nullptr;
    float* h_out0 = nullptr; float* h_out1 = nullptr;
    // Pinned host memory is REQUIRED for async transfers to overlap with kernels
    HIP_CHECK(hipHostMalloc(&h_A,    chunkBytes, 0));
    HIP_CHECK(hipHostMalloc(&h_B,    chunkBytes, 0));
    HIP_CHECK(hipHostMalloc(&h_out0, chunkBytes, 0));
    HIP_CHECK(hipHostMalloc(&h_out1, chunkBytes, 0));

    float* d_buf0 = nullptr; float* d_buf1 = nullptr;
    HIP_CHECK(hipMalloc(&d_buf0, chunkBytes));
    HIP_CHECK(hipMalloc(&d_buf1, chunkBytes));

    for (int i = 0; i < CHUNK; ++i) { h_A[i] = 1.f; h_B[i] = 2.f; }

    hipEvent_t tStart, tStop;
    HIP_CHECK(hipEventCreate(&tStart));
    HIP_CHECK(hipEventCreate(&tStop));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(tStart, 0));

    // stream0 handles chunk A; stream1 handles chunk B — can overlap
    processChunk(stream0, d_buf0, h_A, h_out0, CHUNK, 2.f, 5.f);
    processChunk(stream1, d_buf1, h_B, h_out1, CHUNK, 3.f, 7.f);

    HIP_CHECK(hipStreamSynchronize(stream0));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipEventRecord(tStop, 0));
    HIP_CHECK(hipEventSynchronize(tStop));

    float tms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&tms, tStart, tStop));
    printf("  Two-stream overlap: %.3f ms  "
           "(out0[0]=%.1f  out1[0]=%.1f)\n",
           tms, h_out0[0], h_out1[0]);
    printf("  Expected out0[0]=7.0 (1*2+5)  out1[0]=13.0 (2*3+7)\n");

    // ── 5. hipStreamWaitEvent — cross-stream dependency ───────
    printf("\n=== 5. hipStreamWaitEvent — cross-stream barrier ===\n");
    // Scenario: stream2 must not start until stream0's marker fires
    hipEvent_t marker;
    HIP_CHECK(hipEventCreate(&marker));

    // Do some work in stream0, then mark it
    scaleKernel<<<blocks, threads, 0, stream0>>>(d_buf0, 1.5f, CHUNK);
    HIP_CHECK(hipEventRecord(marker, stream0));   // marker fires when stream0 reaches here

    // stream2 waits for marker before starting — even though we launch it now
    HIP_CHECK(hipStreamWaitEvent(stream2, marker, 0));
    addConstKernel<<<blocks, threads, 0, stream2>>>(d_buf0, 0.5f, CHUNK);
    // This kernel in stream2 will not start until marker fires
    HIP_CHECK(hipStreamSynchronize(stream2));
    printf("  Cross-stream wait complete: stream2 waited for stream0 marker\n");

    // ── 6. Query stream/event completion non-blocking ─────────
    printf("\n=== 6. Non-blocking query ===\n");
    hipEvent_t pollEv;
    HIP_CHECK(hipEventCreate(&pollEv));
    scaleKernel<<<blocks, threads, 0, stream1>>>(d_data, 1.0f, N);
    HIP_CHECK(hipEventRecord(pollEv, stream1));

    // hipEventQuery does NOT block — returns hipSuccess or hipErrorNotReady
    hipError_t queryResult = hipEventQuery(pollEv);
    if (queryResult == hipSuccess)
        printf("  Event already complete (very fast kernel)\n");
    else if (queryResult == hipErrorNotReady)
        printf("  Event not yet complete — would poll here in real app\n");

    HIP_CHECK(hipStreamSynchronize(stream1));
    printf("  After stream sync: event is complete\n");

    // ── Cleanup ───────────────────────────────────────────────
    HIP_CHECK(hipEventDestroy(evStart));   HIP_CHECK(hipEventDestroy(evStop));
    HIP_CHECK(hipEventDestroy(s1Start));   HIP_CHECK(hipEventDestroy(s1Stop));
    HIP_CHECK(hipEventDestroy(tStart));    HIP_CHECK(hipEventDestroy(tStop));
    HIP_CHECK(hipEventDestroy(marker));    HIP_CHECK(hipEventDestroy(pollEv));
    HIP_CHECK(hipStreamDestroy(stream0));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_buf0));  HIP_CHECK(hipFree(d_buf1));
    HIP_CHECK(hipHostFree(h_A)); HIP_CHECK(hipHostFree(h_B));
    HIP_CHECK(hipHostFree(h_out0)); HIP_CHECK(hipHostFree(h_out1));

    printf("\nAll hipStream_t / hipEvent_t demos PASSED\n");
    return 0;
}
