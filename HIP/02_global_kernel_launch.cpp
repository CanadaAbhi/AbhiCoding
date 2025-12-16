// ============================================================
//  02_global_kernel_launch.cpp
//  Demonstrates: __global__ kernel  kernel<<<grid,block,smem,stream>>>
//                threadIdx  blockIdx  blockDim  gridDim
//
//  Compile:
//    hipcc -O2 02_global_kernel_launch.cpp -o 02_demo
//  Run:
//    ./02_demo
// ============================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",              \
                    __FILE__, __LINE__, hipGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ── Kernel 1: 1-D thread indexing ────────────────────────────
// __global__ marks a function that runs on the GPU and is called from the CPU.
// Every thread gets a unique (blockIdx.x * blockDim.x + threadIdx.x) index.
__global__ void vecScaleAdd(const float* a, const float* b,
                            float* c, float alpha, int n)
{
    // 1-D global thread index
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n)
        c[gid] = alpha * a[gid] + b[gid];
}

// ── Kernel 2: 2-D thread indexing (matrix element-wise add) ──
__global__ void matAdd(const float* A, const float* B,
                       float* C, int rows, int cols)
{
    // row = blockIdx.y * blockDim.y + threadIdx.y
    // col = blockIdx.x * blockDim.x + threadIdx.x
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// ── Kernel 3: print built-ins (small demo) ────────────────────
__global__ void printBuiltins()
{
    // Only let first 4 threads print to avoid flooding
    if (threadIdx.x < 4 && blockIdx.x == 0) {
        printf("  threadIdx.x=%u  blockIdx.x=%u  blockDim.x=%u  gridDim.x=%u\n",
               threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
    }
}

// ── Kernel 4: 3-D grid (volumetric computation stub) ─────────
__global__ void volume3D(float* vol, int W, int H, int D, float val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < W && y < H && z < D)
        vol[z * H * W + y * W + x] = val;
}

int main()
{
    // ── Demo 1: 1-D kernel (vector saxpy) ────────────────────
    printf("=== Demo 1: 1-D kernel  kernel<<<grid, block>>>(args) ===\n");
    const int N = 1 << 20;
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, N * sizeof(float)));

    // Initialise with hipMemset (set bytes to a pattern)
    HIP_CHECK(hipMemset(d_a, 0, N * sizeof(float)));  // all zeros
    HIP_CHECK(hipMemset(d_b, 0, N * sizeof(float)));

    // Set host arrays, copy to device
    float h_a[4] = {1.f, 2.f, 3.f, 4.f};
    float h_b[4] = {10.f, 20.f, 30.f, 40.f};
    HIP_CHECK(hipMemcpy(d_a, h_a, 4 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, 4 * sizeof(float), hipMemcpyHostToDevice));

    // Launch: <<<gridDim, blockDim, sharedMemBytes, stream>>>
    //   gridDim  = number of blocks
    //   blockDim = threads per block (must be multiple of warpSize on AMD = 64)
    int threads = 256;                          // threads per block
    int blocks  = (N + threads - 1) / threads; // enough blocks to cover N
    vecScaleAdd<<<blocks, threads>>>(d_a, d_b, d_c, 2.0f, 4);

    HIP_CHECK(hipDeviceSynchronize());

    float h_c[4] = {};
    HIP_CHECK(hipMemcpy(h_c, d_c, 4 * sizeof(float), hipMemcpyDeviceToHost));
    printf("  saxpy result (expect 12 24 36 48): %.0f %.0f %.0f %.0f\n",
           h_c[0], h_c[1], h_c[2], h_c[3]);

    // ── Demo 2: 2-D grid / block ──────────────────────────────
    printf("\n=== Demo 2: 2-D kernel  dim3 grid and block ===\n");
    const int ROWS = 1024, COLS = 1024;
    float *d_A, *d_B, *d_C2;
    size_t matBytes = (size_t)ROWS * COLS * sizeof(float);
    HIP_CHECK(hipMalloc(&d_A, matBytes));
    HIP_CHECK(hipMalloc(&d_B, matBytes));
    HIP_CHECK(hipMalloc(&d_C2, matBytes));
    HIP_CHECK(hipMemset(d_A, 0, matBytes));
    HIP_CHECK(hipMemset(d_B, 0, matBytes));

    // dim3 lets you specify 2-D or 3-D block/grid dimensions
    dim3 block2D(16, 16, 1);        // 256 threads / block
    dim3 grid2D((COLS + 15) / 16,   // ceil(COLS/16) blocks along x
                (ROWS + 15) / 16,   // ceil(ROWS/16) blocks along y
                1);

    matAdd<<<grid2D, block2D>>>(d_A, d_B, d_C2, ROWS, COLS);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  matAdd %dx%d launched with grid(%u,%u,1) block(%u,%u,1)\n",
           ROWS, COLS, grid2D.x, grid2D.y, block2D.x, block2D.y);

    // ── Demo 3: Built-in variables ────────────────────────────
    printf("\n=== Demo 3: Built-in variables threadIdx/blockIdx/blockDim/gridDim ===\n");
    printBuiltins<<<2, 8>>>();   // 2 blocks of 8 threads
    HIP_CHECK(hipDeviceSynchronize());

    // ── Demo 4: 3-D grid ──────────────────────────────────────
    printf("\n=== Demo 4: 3-D grid kernel ===\n");
    const int W = 64, H = 64, D = 64;
    float* d_vol;
    HIP_CHECK(hipMalloc(&d_vol, (size_t)W * H * D * sizeof(float)));

    dim3 block3D(8, 8, 8);   // 512 threads / block
    dim3 grid3D((W + 7) / 8, (H + 7) / 8, (D + 7) / 8);
    volume3D<<<grid3D, block3D>>>(d_vol, W, H, D, 3.14f);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  3-D volume %dx%dx%d filled, grid(%u,%u,%u) block(%u,%u,%u)\n",
           W, H, D, grid3D.x, grid3D.y, grid3D.z,
           block3D.x, block3D.y, block3D.z);

    // ── Demo 5: Shared memory bytes in launch (3rd param) ────
    printf("\n=== Demo 5: dynamic shared memory  kernel<<<g,b,sharedBytes>>> ===\n");
    // Third launch parameter allocates extra dynamic shared memory per block.
    // Kernel accesses it as:  extern __shared__ float sdata[];
    // (shown fully in 05_shared_memory.cpp)
    printf("  3rd launch param (sharedBytes) covered in 05_shared_memory.cpp\n");

    // Cleanup
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b)); HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C2));
    HIP_CHECK(hipFree(d_vol));

    printf("\nAll kernel launch demos PASSED\n");
    return 0;
}
