#include <cuda.h>

extern "C" __global__ void addKernel(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] += 5;
}
