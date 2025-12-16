# HIP API Working Code Implementations

Complete, compilable implementations of every HIP API in the CUDA→HIP mapping table.

## Prerequisites

```bash
# Install ROCm (Ubuntu 22.04)
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
sudo apt install ./amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install --usecase=rocm,hip
sudo usermod -aG render,video $USER
newgrp render

# Verify
hipcc --version
rocminfo | grep Name
```

---

## Files & API Coverage

| File | APIs Covered |
|------|-------------|
| `01_hip_malloc_memcpy_free.cpp` | `hipMalloc` `hipFree` `hipMemcpy` (H2D/D2H/D2D) `hipMemset` |
| `02_global_kernel_launch.cpp` | `__global__` `kernel<<<g,b,smem,stream>>>` `threadIdx` `blockIdx` `blockDim` `gridDim` `dim3` |
| `03_device_synchronize.cpp` | `hipDeviceSynchronize` `hipGetDeviceCount` `hipGetDeviceProperties` `hipSetDevice` `hipDeviceReset` |
| `04_streams_events.cpp` | `hipStream_t` `hipStreamCreate` `hipStreamSynchronize` `hipStreamWaitEvent` `hipEvent_t` `hipEventCreate` `hipEventRecord` `hipEventElapsedTime` `hipMemcpyAsync` |
| `05_shared_memory_syncthreads.cpp` | `__shared__` (static & dynamic) `__syncthreads()` `extern __shared__` tiled GEMM reduction |
| `06_warpsize_atomics.cpp` | `warpSize` (=64 on AMD) `atomicAdd` `atomicSub` `atomicMin` `atomicMax` `atomicAnd` `atomicOr` `atomicXor` `atomicCAS` `atomicExch` `__ballot64` `__shfl_down` |
| `07_visible_devices_and_master.cpp` | `HIP_VISIBLE_DEVICES` multi-GPU `hipSetDevice` full pipeline integration |

---

## Build

```bash
# Option 1: Makefile (simplest)
cd hip_demos
make               # build all 7 demos
make run_all       # build and run all

# Build for specific GPU architecture:
make ARCH=gfx90a   # MI250X
make ARCH=gfx942   # MI300X
make ARCH=gfx1100  # RX 7900 XTX

# Option 2: CMake
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=$(which hipcc)
make -j$(nproc)

# Option 3: Compile individual file
hipcc -O2 01_hip_malloc_memcpy_free.cpp -o 01_demo
./01_demo
```

---

## API Quick Reference

### Memory
```cpp
// Allocate GPU memory
float* d_ptr;
hipMalloc(&d_ptr, N * sizeof(float));       // device alloc
hipHostMalloc(&h_ptr, bytes, 0);            // pinned host (for async)
hipMallocManaged(&um_ptr, bytes);           // unified memory

// Copy
hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);   // sync
hipMemcpyAsync(dst, src, bytes, dir, stream);         // async (needs pinned)
hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);  // GPU-to-GPU

// Free
hipFree(d_ptr);
hipHostFree(h_ptr);
```

### Kernel Launch
```cpp
// Define kernel
__global__ void myKernel(float* data, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) data[gid] *= 2.f;
}

// Launch: <<<gridDim, blockDim, sharedBytes, stream>>>
dim3 block(256);                           // 256 threads/block
dim3 grid((N + 255) / 256);               // enough blocks
myKernel<<<grid, block, 0, stream>>>(d_data, N);

// 2-D launch
dim3 block2(16, 16);
dim3 grid2(ceil(W/16.0), ceil(H/16.0));
kernel2D<<<grid2, block2>>>(d_mat, W, H);
```

### Synchronisation
```cpp
hipDeviceSynchronize();            // wait for ALL GPU work
hipStreamSynchronize(stream);      // wait for specific stream
hipEventRecord(event, stream);     // timestamp in stream
hipEventSynchronize(event);        // wait for specific event

float ms;
hipEventElapsedTime(&ms, start, stop);  // timing in milliseconds
```

### Shared Memory
```cpp
// Static (size known at compile time)
__shared__ float tile[256];

// Dynamic (size passed at launch)
extern __shared__ float sdata[];
myKernel<<<grid, block, 256*sizeof(float)>>>(d_data);

// Always sync before reading data written by other threads
__syncthreads();
```

### Atomics
```cpp
atomicAdd(&addr, val);    // addr += val; returns old
atomicSub(&addr, val);    // addr -= val
atomicMin(&addr, val);    // addr = min(addr, val)
atomicMax(&addr, val);    // addr = max(addr, val)
atomicAnd(&addr, val);    // addr &= val
atomicOr (&addr, val);    // addr |= val
atomicXor(&addr, val);    // addr ^= val
atomicExch(&addr, val);   // addr = val; returns old
atomicCAS(&addr, cmp, val); // if addr==cmp: addr=val; returns old
```

### AMD-Specific (warpSize=64)
```cpp
warpSize                  // 64 on AMD (vs 32 NVIDIA)
uint64_t mask = __ballot64(condition);   // 64-bit lane mask
int n = __popcll(mask);                  // count active lanes
float v = __shfl_down(val, offset);      // shuffle within wavefront
```

### Device Selection (HIP_VISIBLE_DEVICES)
```bash
# Shell — restrict visible GPUs
export HIP_VISIBLE_DEVICES=0       # only GPU 0
export HIP_VISIBLE_DEVICES=0,2     # GPUs 0 and 2 (remapped to 0,1)
export HIP_VISIBLE_DEVICES=""      # hide all GPUs

# Python (before importing torch)
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"
```

```cpp
// C++ — programmatic device selection
hipSetDevice(1);           // switch to GPU 1
int dev;
hipGetDevice(&dev);        // query current device
hipGetDeviceCount(&count); // count visible GPUs
```

---

## Key Differences from CUDA

| Feature | CUDA | HIP / AMD |
|---------|------|-----------|
| Warp size | 32 threads | **64 threads** (wavefront) |
| Ballot mask | `uint32_t __ballot_sync()` | `uint64_t __ballot64()` |
| Shuffle | `__shfl_down_sync(mask, val, offset)` | `__shfl_down(val, offset)` |
| Visible GPUs env | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` |
| Device prefix | `cuda*` | `hip*` |
| Compiler | `nvcc` | `hipcc` |
| Architecture flag | `--arch=sm_80` | `--offload-arch=gfx90a` |
| ISA | PTX → SASS | LLVM IR → GCN/RDNA ISA |

---

## Troubleshooting

```bash
# GPU not found
ls /dev/kfd /dev/dri/renderD*      # check device nodes
groups                              # must include 'render' and 'video'

# Check which GPU architecture you have
rocminfo | grep -E 'Name:|gfx'

# Run with verbose HIP logging
AMD_LOG_LEVEL=4 ./01_demo 2>&1 | head -50

# Check memory
rocm-smi --showmemuse

# Fix: wrong architecture
hipcc --offload-arch=$(rocminfo | grep 'gfx' | head -1 | tr -d ' ') code.cpp
```
