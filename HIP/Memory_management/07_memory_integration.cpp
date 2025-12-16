// ================================================================
//  07_memory_integration.cpp
//  HIP Memory Management — Complete Integration Demo
//
//  Exercises ALL 10 memory management APIs in a single coherent
//  pipeline: batch neural network forward pass simulation.
//
//  APIs USED:
//    hipMalloc            — activations, weights in VRAM
//    hipFree              — release VRAM
//    hipMemcpy            — load weights H2D (sync, one-time)
//    hipMemcpyAsync       — stream input batches H2D
//    hipHostMalloc        — pinned input/output host buffers
//    hipHostFree          — release pinned buffers
//    hipMallocManaged     — config / hyperparameters (CPU+GPU)
//    hipMemset            — zero gradients before backward
//    hipMemGetInfo        — VRAM budget check before alloc
//    hipPointerGetAttributes — validate pointer types
//
//  COMPILE:
//    hipcc -O2 07_memory_integration.cpp -o 07_integration
//  RUN:
//    ./07_integration
// ================================================================
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>

#define HIP_CHECK(call)                                               \
    do {                                                              \
        hipError_t _e = (call);                                       \
        if (_e != hipSuccess) {                                       \
            fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",            \
                    hipGetErrorString(_e), __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// ── GPU Kernels ───────────────────────────────────────────────
// Linear layer: out = in @ W^T  (simplified, no bias)
__global__ void linearFwd(const float* in, const float* W,
                           float* out, int B, int IN, int OUT) {
    int b   = blockIdx.y;                           // batch index
    int o   = blockIdx.x * blockDim.x + threadIdx.x; // output neuron
    if (b >= B || o >= OUT) return;
    float acc = 0.f;
    for (int i = 0; i < IN; ++i)
        acc += in[b * IN + i] * W[o * IN + i];
    out[b * OUT + o] = acc;
}

// ReLU activation
__global__ void reluInplace(float* x, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) x[g] = fmaxf(0.f, x[g]);
}

// L2 loss: loss[b] = sum((pred - target)^2) / OUT
__global__ void l2Loss(const float* pred, const float* target,
                        float* loss, int B, int OUT) {
    int b = blockIdx.x;
    if (b >= B) return;
    float s = 0.f;
    for (int o = 0; o < OUT; ++o) {
        float d = pred[b * OUT + o] - target[b * OUT + o];
        s += d * d;
    }
    loss[b] = s / OUT;
}

// Zero gradients
__global__ void zeroGrad(float* g, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) g[idx] = 0.f;
}

// ── Helper: print pointer type ────────────────────────────────
static const char* ptrTypeName(int type) {
    switch (type) {
        case hipMemoryTypeHost:    return "Host(pinned)";
        case hipMemoryTypeDevice:  return "Device";
        case hipMemoryTypeArray:   return "Array";
        case hipMemoryTypeUnified: return "Managed(UM)";
        default:                   return "Unknown";
    }
}

static void checkPtr(const char* name, const void* ptr)
{
    hipPointerAttribute_t a = {};
    hipError_t e = hipPointerGetAttributes(&a, ptr);
    if (e == hipSuccess)
        printf("  [ptr] %-20s → %s  device=%d  managed=%d\n",
               name, ptrTypeName(a.type), a.device, a.isManaged);
    else {
        printf("  [ptr] %-20s → unregistered (CPU-only)\n", name);
        hipGetLastError();
    }
}

// ================================================================
//  Model configuration in Unified Memory (GPU + CPU readable)
// ================================================================
struct ModelConfig {
    int batch_size;
    int input_dim;
    int hidden_dim;
    int output_dim;
    float learning_rate;
    int num_epochs;
};

// ================================================================
//  Main Integration Demo
// ================================================================
int main()
{
    printf("══════════════════════════════════════════════════════════\n");
    printf("  HIP Memory API: Complete Integration Demo\n");
    printf("  Simulated MLP forward pass — 3 streams, double-buffer\n");
    printf("══════════════════════════════════════════════════════════\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  |  VRAM: %.2f GB\n\n",
           prop.name, (double)prop.totalGlobalMem / (1 << 30));

    // ── STEP 1: hipMemGetInfo — check VRAM budget ─────────────
    printf("─── Step 1: hipMemGetInfo — VRAM budget check ───\n");
    size_t vram_free = 0, vram_total = 0;
    HIP_CHECK(hipMemGetInfo(&vram_free, &vram_total));
    printf("  VRAM: %.2f / %.2f GB free\n",
           (double)vram_free / (1 << 30),
           (double)vram_total / (1 << 30));

    // ── STEP 2: hipMallocManaged — model config (CPU+GPU) ─────
    printf("\n─── Step 2: hipMallocManaged — model config in UM ───\n");
    ModelConfig* cfg = nullptr;
    HIP_CHECK(hipMallocManaged(&cfg, sizeof(ModelConfig)));

    // CPU writes config
    cfg->batch_size    = 64;
    cfg->input_dim     = 512;
    cfg->hidden_dim    = 256;
    cfg->output_dim    = 10;
    cfg->learning_rate = 1e-3f;
    cfg->num_epochs    = 3;
    checkPtr("cfg (ModelConfig)", cfg);
    printf("  Config: B=%d  IN=%d  H=%d  OUT=%d  lr=%.4f\n",
           cfg->batch_size, cfg->input_dim, cfg->hidden_dim,
           cfg->output_dim, cfg->learning_rate);

    const int B   = cfg->batch_size;
    const int IN  = cfg->input_dim;
    const int H   = cfg->hidden_dim;
    const int OUT = cfg->output_dim;

    // ── STEP 3: hipMalloc — weights & activations in VRAM ─────
    printf("\n─── Step 3: hipMalloc — weights + activations ───\n");
    float *d_W1=nullptr, *d_W2=nullptr;            // weights
    float *d_act1=nullptr, *d_act2=nullptr;        // activations
    float *d_grad_W1=nullptr, *d_grad_W2=nullptr;  // gradients
    float *d_loss=nullptr;                          // per-sample loss

    size_t W1_bytes  = (size_t)H   * IN  * sizeof(float);
    size_t W2_bytes  = (size_t)OUT * H   * sizeof(float);
    size_t act1_bytes= (size_t)B   * H   * sizeof(float);
    size_t act2_bytes= (size_t)B   * OUT * sizeof(float);

    HIP_CHECK(hipMalloc(&d_W1,      W1_bytes));
    HIP_CHECK(hipMalloc(&d_W2,      W2_bytes));
    HIP_CHECK(hipMalloc(&d_act1,    act1_bytes));
    HIP_CHECK(hipMalloc(&d_act2,    act2_bytes));
    HIP_CHECK(hipMalloc(&d_grad_W1, W1_bytes));
    HIP_CHECK(hipMalloc(&d_grad_W2, W2_bytes));
    HIP_CHECK(hipMalloc(&d_loss,    B * sizeof(float)));

    size_t allocated = W1_bytes + W2_bytes + act1_bytes + act2_bytes
                     + W1_bytes + W2_bytes + B * sizeof(float);
    printf("  Allocated %.2f MB in VRAM\n", (double)allocated / (1 << 20));
    checkPtr("d_W1", d_W1);
    checkPtr("d_act1", d_act1);

    // ── STEP 4: hipMemcpy — load weights H2D (synchronous) ────
    printf("\n─── Step 4: hipMemcpy H2D — load weights ───\n");
    float* h_W1 = static_cast<float*>(malloc(W1_bytes));
    float* h_W2 = static_cast<float*>(malloc(W2_bytes));
    for (size_t i = 0; i < H * IN;  ++i) h_W1[i] = 0.01f * sinf(i);
    for (size_t i = 0; i < OUT * H; ++i) h_W2[i] = 0.01f * cosf(i);

    HIP_CHECK(hipMemcpy(d_W1, h_W1, W1_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W2, W2_bytes, hipMemcpyHostToDevice));
    printf("  Weights W1[%dx%d] and W2[%dx%d] loaded to GPU\n",
           H, IN, OUT, H);

    // ── STEP 5: hipHostMalloc — pinned input/output buffers ───
    printf("\n─── Step 5: hipHostMalloc — pinned I/O buffers ───\n");
    size_t in_bytes  = (size_t)B * IN  * sizeof(float);
    size_t out_bytes = (size_t)B * OUT * sizeof(float);

    // Double-buffer: [0] being processed, [1] being loaded
    float* h_in[2]     = {};
    float* h_target[2] = {};
    float* h_out[2]    = {};
    float* d_in[2]     = {};
    float* d_target[2] = {};

    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipHostMalloc(&h_in[i],     in_bytes,  0));
        HIP_CHECK(hipHostMalloc(&h_target[i], out_bytes, 0));
        HIP_CHECK(hipHostMalloc(&h_out[i],    out_bytes, 0));
        HIP_CHECK(hipMalloc    (&d_in[i],     in_bytes));
        HIP_CHECK(hipMalloc    (&d_target[i], out_bytes));
    }
    checkPtr("h_in[0]",  h_in[0]);
    checkPtr("h_out[0]", h_out[0]);
    printf("  Pinned buffers: 2 × (input + target + output)\n");

    // ── STEP 6: hipMemset — zero gradients ────────────────────
    printf("\n─── Step 6: hipMemset — zero gradients ───\n");
    HIP_CHECK(hipMemset(d_grad_W1, 0, W1_bytes));
    HIP_CHECK(hipMemset(d_grad_W2, 0, W2_bytes));
    printf("  Gradients zeroed (hipMemset)\n");

    // ── STEP 7: hipMemcpyAsync — training loop ────────────────
    printf("\n─── Step 7: hipMemcpyAsync — training loop ───\n");

    hipStream_t stream[2];
    HIP_CHECK(hipStreamCreate(&stream[0]));
    HIP_CHECK(hipStreamCreate(&stream[1]));

    hipEvent_t epoch_start, epoch_stop;
    HIP_CHECK(hipEventCreate(&epoch_start));
    HIP_CHECK(hipEventCreate(&epoch_stop));

    const int STEPS_PER_EPOCH = 4;

    for (int epoch = 0; epoch < cfg->num_epochs; ++epoch) {

        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(epoch_start));

        // Zero grads at epoch start
        int tg = 256;
        int bg1 = ((int)(H * IN)  + 255) / 256;
        int bg2 = ((int)(OUT * H) + 255) / 256;
        zeroGrad<<<bg1, tg, 0, stream[0]>>>(d_grad_W1, H * IN);
        zeroGrad<<<bg2, tg, 0, stream[1]>>>(d_grad_W2, OUT * H);

        for (int step = 0; step < STEPS_PER_EPOCH; ++step) {
            int buf = step % 2;

            // CPU fills input (simulating data loader)
            for (int j = 0; j < B * IN;  ++j) h_in[buf][j]     = 0.1f * sinf(step * j);
            for (int j = 0; j < B * OUT; ++j) h_target[buf][j] = (j % OUT == step % OUT) ? 1.f : 0.f;

            // Async H2D in this step's stream
            HIP_CHECK(hipMemcpyAsync(d_in[buf], h_in[buf], in_bytes,
                                       hipMemcpyHostToDevice, stream[buf]));
            HIP_CHECK(hipMemcpyAsync(d_target[buf], h_target[buf], out_bytes,
                                       hipMemcpyHostToDevice, stream[buf]));

            // Forward pass (in same stream → runs after copy)
            dim3 blk(256), grd1((H   + 255)/256, B), grd2((OUT + 255)/256, B);
            linearFwd<<<grd1, blk, 0, stream[buf]>>>(
                d_in[buf], d_W1, d_act1, B, IN, H);
            reluInplace<<<(B*H+255)/256, 256, 0, stream[buf]>>>(d_act1, B*H);
            linearFwd<<<grd2, blk, 0, stream[buf]>>>(
                d_act1, d_W2, d_act2, B, H, OUT);

            // Loss
            l2Loss<<<B, 1, 0, stream[buf]>>>(d_act2, d_target[buf], d_loss, B, OUT);

            // Async D2H — copy predictions
            HIP_CHECK(hipMemcpyAsync(h_out[buf], d_act2, out_bytes,
                                       hipMemcpyDeviceToHost, stream[buf]));
        }

        // Sync all streams
        HIP_CHECK(hipStreamSynchronize(stream[0]));
        HIP_CHECK(hipStreamSynchronize(stream[1]));
        HIP_CHECK(hipEventRecord(epoch_stop));
        HIP_CHECK(hipEventSynchronize(epoch_stop));

        float epoch_ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&epoch_ms, epoch_start, epoch_stop));

        // Read loss (sync, small copy)
        float h_loss[4] = {};
        HIP_CHECK(hipMemcpy(h_loss, d_loss, 4 * sizeof(float),
                             hipMemcpyDeviceToHost));
        printf("  Epoch %d/%d  time=%.2f ms  loss[0]=%.4f  loss[1]=%.4f\n",
               epoch + 1, cfg->num_epochs, epoch_ms,
               h_loss[0], h_loss[1]);
    }

    // ── STEP 8: hipPointerGetAttributes — audit all ptrs ──────
    printf("\n─── Step 8: hipPointerGetAttributes — pointer audit ───\n");
    checkPtr("d_W1",      d_W1);
    checkPtr("d_act1",    d_act1);
    checkPtr("d_loss",    d_loss);
    checkPtr("cfg (UM)",  cfg);
    checkPtr("h_in[0]",  h_in[0]);
    checkPtr("h_W1",     h_W1);   // regular malloc — unregistered

    // ── STEP 9: Final VRAM check ──────────────────────────────
    printf("\n─── Step 9: hipMemGetInfo — final VRAM status ───\n");
    size_t vram_after = 0;
    HIP_CHECK(hipMemGetInfo(&vram_after, &vram_total));
    printf("  VRAM free: %.2f GB  (consumed %.2f MB)\n",
           (double)vram_after / (1 << 30),
           (double)(vram_free - vram_after) / (1 << 20));

    // ── STEP 10: hipFree + hipHostFree — release everything ───
    printf("\n─── Step 10: hipFree + hipHostFree — cleanup ───\n");
    HIP_CHECK(hipFree(d_W1));         HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_act1));       HIP_CHECK(hipFree(d_act2));
    HIP_CHECK(hipFree(d_grad_W1));    HIP_CHECK(hipFree(d_grad_W2));
    HIP_CHECK(hipFree(d_loss));
    HIP_CHECK(hipFree(cfg));          // hipFree works for UM too

    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipHostFree(h_in[i]));
        HIP_CHECK(hipHostFree(h_target[i]));
        HIP_CHECK(hipHostFree(h_out[i]));
        HIP_CHECK(hipFree(d_in[i]));
        HIP_CHECK(hipFree(d_target[i]));
    }

    HIP_CHECK(hipEventDestroy(epoch_start));
    HIP_CHECK(hipEventDestroy(epoch_stop));
    HIP_CHECK(hipStreamDestroy(stream[0]));
    HIP_CHECK(hipStreamDestroy(stream[1]));

    free(h_W1); free(h_W2);

    size_t vram_final = 0;
    HIP_CHECK(hipMemGetInfo(&vram_final, &vram_total));
    printf("  VRAM free after cleanup: %.2f GB  (%.2f MB recovered)\n",
           (double)vram_final / (1 << 30),
           (double)(vram_final - vram_after) / (1 << 20));

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  SUMMARY — All 10 Memory APIs Used:\n");
    printf("  ✓ hipMalloc             weights + activations + grads\n");
    printf("  ✓ hipFree               all device allocations\n");
    printf("  ✓ hipMemcpy             synchronous weight load\n");
    printf("  ✓ hipMemcpyAsync        streaming batch H2D + D2H\n");
    printf("  ✓ hipHostMalloc         pinned I/O double-buffers\n");
    printf("  ✓ hipHostFree           pinned buffer release\n");
    printf("  ✓ hipMallocManaged      model config (CPU+GPU)\n");
    printf("  ✓ hipMemset             gradient zeroing\n");
    printf("  ✓ hipMemGetInfo         VRAM budget checks\n");
    printf("  ✓ hipPointerGetAttributes pointer type audit\n");
    printf("══════════════════════════════════════════════════════════\n\n");
    return 0;
}
