# ROCm-Style GPU Driver Stack with OpenGL Integration

A complete GPU driver stack implementation demonstrating kernel-space driver, userspace library, and OpenGL application integration - similar to AMD's ROCm architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                         │
│                  (rocm_opengl_app.c)                        │
│                                                             │
│  ┌─────────────────┐          ┌──────────────────┐        │
│  │   OpenGL API    │          │   ROCm API       │        │
│  │   (Rendering)   │          │   (Compute)      │        │
│  └────────┬────────┘          └────────┬─────────┘        │
└───────────┼──────────────────────────────┼─────────────────┘
            │                              │
            │                              │ ioctl/mmap
┌───────────┼──────────────────────────────┼─────────────────┐
│           │                              │                 │
│     ┌─────▼──────┐            ┌──────────▼────────┐       │
│     │   OpenGL   │            │   librocm.so      │       │
│     │   Driver   │            │   (Userspace)     │       │
│     └────────────┘            └──────────┬────────┘       │
│                                          │                 │
└──────────────────────────────────────────┼─────────────────┘
                                           │
                              ioctl/mmap   │
┌──────────────────────────────────────────┼─────────────────┐
│                    Kernel Space          │                 │
│                                          │                 │
│                           ┌──────────────▼────────┐        │
│                           │  rocm_gpu_driver.ko   │        │
│                           │  (Character Device)   │        │
│                           │                       │        │
│                           │  • Memory Manager     │        │
│                           │  • Command Queue      │        │
│                           │  • DMA Buffers        │        │
│                           └──────────┬────────────┘        │
│                                      │                     │
│                           ┌──────────▼────────────┐        │
│                           │   GPU Hardware        │        │
│                           │   (Simulated)         │        │
│                           └───────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Kernel Space Driver (`rocm_gpu_driver.c`)
Low-level GPU hardware abstraction with character device interface, DMA memory management, and command queue processing.

### 2. Userspace Library (`librocm.so`)
High-level API providing memory allocation, data transfer, and command submission abstractions.

### 3. OpenGL Application (`rocm_opengl_app.c`)
Demo application showing ROCm + OpenGL integration with rotating colored triangle.

## Quick Start

### Build Everything
```bash
./build.sh
```

### Load and Run
```bash
# Load kernel module
sudo insmod rocm_gpu_driver.ko

# Verify device
ls -l /dev/rocm_gpu0

# Run application
./rocm_opengl_app

# Unload module
sudo rmmod rocm_gpu_driver
```

## Prerequisites

**Ubuntu/Debian**:
```bash
sudo apt-get install build-essential linux-headers-$(uname -r)
sudo apt-get install libgl1-mesa-dev libglew-dev libglfw3-dev
```

**Fedora/RHEL**:
```bash
sudo dnf install kernel-devel gcc make mesa-libGL-devel glew-devel glfw-devel
```

## Key Features

- **Kernel Driver**: DMA-coherent memory, IOCTL interface, command queues
- **Userspace Library**: Memory management, command buffers, error handling
- **OpenGL Integration**: Shared memory, compute + graphics pipeline
- **Complete Stack**: Demonstrates full kernel-to-application data flow

## Files

- `rocm_gpu_driver.c` - Kernel module (~500 lines)
- `librocm.h/c` - Userspace library (~550 lines)
- `rocm_opengl_app.c` - Demo application (~500 lines)
- `Makefile.kernel` - Kernel build system
- `Makefile.userspace` - Userspace build system
- `build.sh` - Automated build script

## Debugging

```bash
# Kernel messages
dmesg | grep ROCM

# Module info
lsmod | grep rocm
modinfo rocm_gpu_driver.ko

# Fix permissions
sudo chmod 666 /dev/rocm_gpu0
```

## Architecture Details

See detailed documentation in this README for:
- IOCTL interface specifications
- Memory management lifecycle
- Command buffer format
- Data flow diagrams
- Security considerations
- Performance optimization

## License

Educational/demonstration code.
