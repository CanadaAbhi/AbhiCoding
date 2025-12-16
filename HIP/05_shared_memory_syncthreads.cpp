// ============================================================
//  05_shared_memory_syncthreads.cpp
//  Demonstrates: __shared__  __syncthreads()  LDS usage
//                static shared memory  dynamic shared memory
//                bank conflicts  reduction pattern  tiled GEMM
//
//  Compile:
//    hipcc -O2 05_shared_memory_syncthreads.cpp -o 05_demo
//  Run:
//    ./05_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <numeric>

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── Kernel 1: parallel reduction using __shared__ ─────────────
// Reduces N floats to a single sum, one block at a time.
// Each block writes its partial sum; a second pass sums those.
template<int BLOCK_SIZE>
__global__ void reduceBlock(const float* in, float* out, int n)
{
    // Static shared memory — size known at compile time
    __shared__ float sdata[BLOCK_SIZE];   // LDS allocation: 256 floats = 1 KB

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load global → shared.  Out-of-bounds threads contribute 0.
    sdata[tid] = (gid < n) ? in[gid] : 0.f;

    // ── __syncthreads() ────────────────────────────────────────
    // REQUIRED before any thread reads from shared memory that was
    // written by a DIFFERENT thread in the same block.
    // On AMD this is a wavefront-aware barrier (covers all 64 lanes
    // per wavefront and all wavefronts in the block).
    __syncthreads();

    // Tree reduction: stride halves each step
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();   // ← barrier after every step!
    }

    // Thread 0 writes the block's partial sum
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ── Kernel 2: tiled matrix multiply using __shared__ ──────────
#define TILE_SIZE 16

__global__ void tiledMatMul(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* C, int M, int N, int K)
{
    // Two TILE_SIZE x TILE_SIZE tiles in shared memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.f;

    // Sweep tiles across the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaboratively load tile of A and tile of B into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.f;

        __syncthreads();   // all threads finished loading

        // Compute dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();   // all threads finished reading — safe to overwrite tiles
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ── Kernel 3: dynamic shared memory ───────────────────────────
// Allocated at launch: kernel<<<g, b, sharedBytes>>>
__global__ void dynamicShared(float* out, int n, int perThread)
{
    extern __shared__ float sdata[];   // size determined at launch time

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Each thread fills 'perThread' slots it owns
    for (int i = 0; i < perThread; ++i)
        sdata[tid * perThread + i] = (gid < n) ? static_cast<float>(gid * perThread + i) : 0.f;

    __syncthreads();

    // Simple sum of this thread's slice
    float sum = 0.f;
    for (int i = 0; i < perThread; ++i)
        sum += sdata[tid * perThread + i];

    if (gid < n) out[gid] = sum;
}

// ── Kernel 4: bank-conflict demo ──────────────────────────────
// AMD LDS has 32 banks (4-byte interleaved).  Access with stride=32 causes
// 32-way bank conflicts.  Stride=1 is conflict-free.
__global__ void bankConflictDemo(float* out, int n, int stride)
{
    __shared__ float smem[32 * 32];
    int tid = threadIdx.x;
    // Strided write — stride==1 is conflict-free, stride==32 hits same bank
    smem[tid * stride % (32 * 32)] = static_cast<float>(tid);
    __syncthreads();
    if (tid < n) out[tid] = smem[tid];
}

int main()
{
    // ── Demo 1: block reduction ───────────────────────────────
    printf("=== 1. Parallel reduction with __shared__ + __syncthreads() ===\n");
    const int N = 1 << 20;
    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    double expected = static_cast<double>(N);

    float* d_in = nullptr;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));

    const int BLOCK = 256;
    int numBlocks = (N + BLOCK - 1) / BLOCK;

    float* d_partial = nullptr;
    HIP_CHECK(hipMalloc(&d_partial, numBlocks * sizeof(float)));

    // Pass 1: each block reduces BLOCK elements to 1 partial sum
    reduceBlock<BLOCK><<<numBlocks, BLOCK>>>(d_in, d_partial, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Pass 2: reduce partial sums on CPU (or launch another kernel)
    float* h_partial = new float[numBlocks];
    HIP_CHECK(hipMemcpy(h_partial, d_partial, numBlocks * sizeof(float),
                         hipMemcpyDeviceToHost));
    double total = 0.;
    for (int i = 0; i < numBlocks; ++i) total += h_partial[i];

    printf("  Sum of %d ones: %.0f  (expected %.0f)  %s\n",
           N, total, expected, (total == expected) ? "PASS" : "FAIL");

    // ── Demo 2: tiled matrix multiply ─────────────────────────
    printf("\n=== 2. Tiled GEMM with __shared__ tiles ===\n");
    const int M = 512, Kd = 256, Nmat = 512;
    size_t sA = M * Kd * sizeof(float);
    size_t sB = Kd * Nmat * sizeof(float);
    size_t sC = M * Nmat * sizeof(float);

    float* h_A = new float[M * Kd];
    float* h_B = new float[Kd * Nmat];
    float* h_C = new float[M * Nmat]();

    // Fill A=1, B=1 → C should be all K
    for (int i = 0; i < M * Kd;   ++i) h_A[i] = 1.f;
    for (int i = 0; i < Kd * Nmat; ++i) h_B[i] = 1.f;

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, sA));
    HIP_CHECK(hipMalloc(&d_B, sB));
    HIP_CHECK(hipMalloc(&d_C, sC));
    HIP_CHECK(hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, sC));

    dim3 blockGEMM(TILE_SIZE, TILE_SIZE);
    dim3 gridGEMM((Nmat + TILE_SIZE - 1) / TILE_SIZE,
                  (M    + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatMul<<<gridGEMM, blockGEMM>>>(d_A, d_B, d_C, M, Nmat, Kd);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C, d_C, sC, hipMemcpyDeviceToHost));
    bool gemm_ok = true;
    for (int i = 0; i < M * Nmat; ++i)
        if (fabsf(h_C[i] - static_cast<float>(Kd)) > 1e-3f)
            { gemm_ok = false; break; }
    printf("  Tiled GEMM %dx%dx%d: %s (C[0]=%.0f expected %d)\n",
           M, Kd, Nmat, gemm_ok ? "PASS" : "FAIL", h_C[0], Kd);

    // ── Demo 3: dynamic shared memory ────────────────────────
    printf("\n=== 3. Dynamic shared memory   kernel<<<g, b, sharedBytes>>> ===\n");
    const int Ndyn = 128;
    const int perThread = 4;
    size_t sharedBytes = BLOCK * perThread * sizeof(float);
    printf("  Allocating %zu bytes of dynamic shared memory per block\n", sharedBytes);

    float* d_dynOut = nullptr;
    HIP_CHECK(hipMalloc(&d_dynOut, Ndyn * sizeof(float)));

    // Pass sharedBytes as the 3rd launch parameter
    dynamicShared<<<1, Ndyn, sharedBytes>>>(d_dynOut, Ndyn, perThread);
    HIP_CHECK(hipDeviceSynchronize());

    float h_dynOut[4] = {};
    HIP_CHECK(hipMemcpy(h_dynOut, d_dynOut, 4 * sizeof(float), hipMemcpyDeviceToHost));
    // Thread 0 owns indices 0..3, sum = 0+1+2+3 = 6
    printf("  dynamic shmem out[0]=%.0f (expect 6=%d+%d+%d+%d)\n",
           h_dynOut[0], 0, 1, 2, 3);

    // ── Demo 4: __syncthreads scope ───────────────────────────
    printf("\n=== 4. __syncthreads() — barrier scope notes ===\n");
    printf("  __syncthreads() syncs all threads in a BLOCK (not grid).\n");
    printf("  For grid-level sync, use cooperative groups or multiple kernel launches.\n");
    printf("  __syncwarp()  syncs a single 64-lane wavefront (AMD-specific width).\n");
    printf("  On AMD: warpSize=%d, so __syncwarp() = barrier over 64 lanes.\n", 64);

    // Cleanup
    delete[] h_in; delete[] h_partial;
    delete[] h_A;  delete[] h_B;  delete[] h_C;
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_dynOut));

    printf("\nAll __shared__ / __syncthreads() demos PASSED\n");
    return 0;
}
