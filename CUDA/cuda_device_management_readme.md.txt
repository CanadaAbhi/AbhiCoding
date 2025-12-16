Headers and Setup
#include <iostream>: Standard C++ library for printing to the terminal (std::cout).
#include <cuda_runtime.h>: The core CUDA library that allows your CPU to talk to the NVIDIA GPU.
int deviceCount = 0;: A variable to store how many GPUs the system finds (on a Jetson Nano, this will usually be 1).
2. Finding the GPU
cudaGetDeviceCount(&deviceCount);: Asks the system: "How many NVIDIA GPUs are plugged in?" It stores the answer in deviceCount.
if (err != cudaSuccess): Error handling. If the drivers are broken or the GPU is missing, it prints a human-readable error message using cudaGetErrorString.
3. Selecting the Device
cudaGetDevice(&currentDevice);: Asks which GPU is currently being used (defaults to 0).
cudaSetDevice(deviceToUse);: Tells the program: "I want to use GPU #0 for all following tasks." On systems with multiple GPUs (like a desktop with two RTX cards), this is how you switch between them.
4. Reading "The Specs" (The most important part)
cudaDeviceProp prop;: Creates a "struct" (a container) to hold all the technical details of the GPU.
cudaGetDeviceProperties(&prop, currentDevice);: Fills that container with real data from your hardware.
prop.name: The model name (e.g., "NVIDIA Tegra X1" for the Nano).
prop.major . minor: The Compute Capability. This tells you which CUDA features the chip supports (Nano is 5.3).
prop.totalGlobalMem: Total VRAM available. The code divides by (1024 * 1024) to convert bytes into Megabytes (MB).
prop.multiProcessorCount: How many Streaming Multiprocessors (SMs) the GPU has. Think of these as the "cores" of the GPU.
prop.maxThreadsPerBlock: The maximum number of parallel "workers" you can pack into a single block (usually 1024).
5. Cleanup
cudaDeviceSynchronize();: Forces the CPU to wait until the GPU has finished all previous commands before moving to the next line of code.
cudaDeviceReset();: Cleans up. It destroys all allocations and resets the GPU state for the next program that wants to use it.