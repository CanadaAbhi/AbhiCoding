# ROCm Driver Stack - Detailed Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Kernel Space Driver](#kernel-space-driver)
3. [Userspace Library](#userspace-library)
4. [Data Flow](#data-flow)
5. [Memory Management](#memory-management)
6. [Command Processing](#command-processing)
7. [IOCTL Interface](#ioctl-interface)
8. [Security Model](#security-model)
9. [Performance](#performance)
10. [Extending the Driver](#extending-the-driver)

## System Overview

### Three-Tier Architecture

```
┌────────────────────────────────────┐
│  Application Layer                 │  User programs using OpenGL + ROCm
│  - rocm_opengl_app.c               │
└────────────────────────────────────┘
              ↕ API calls
┌────────────────────────────────────┐
│  Userspace Library Layer           │  High-level abstractions
│  - librocm.so                      │
└────────────────────────────────────┘
              ↕ ioctl/mmap
┌────────────────────────────────────┐
│  Kernel Driver Layer               │  Hardware access & management
│  - rocm_gpu_driver.ko              │
└────────────────────────────────────┘
              ↕ Hardware I/O
┌────────────────────────────────────┐
│  GPU Hardware (Simulated)          │
└────────────────────────────────────┘
```

## Kernel Space Driver

### Module Structure

```c
struct rocm_device {
    struct cdev cdev;              // Character device
    struct device *dev;            // Device pointer
    struct class *class;           // Device class
    dev_t devno;                   // Device number
    struct mutex lock;             // Global lock
    struct list_head mem_list;     // Memory object list
    uint64_t next_handle;          // Handle allocator
    
    // GPU state
    uint32_t status_reg;           // Status register
    uint32_t cmd_queue_head;       // Command queue head
    uint32_t cmd_queue_tail;       // Command queue tail
};
```

### Memory Object

```c
struct rocm_mem_object {
    uint64_t handle;               // Unique identifier
    uint64_t size;                 // Size in bytes
    void *kernel_addr;             // Kernel virtual address
    dma_addr_t dma_addr;           // DMA/GPU physical address
    struct list_head list;         // List linkage
};
```

### File Operations

```c
static struct file_operations rocm_fops = {
    .owner = THIS_MODULE,
    .open = rocm_open,
    .release = rocm_release,
    .unlocked_ioctl = rocm_ioctl,
    .mmap = rocm_mmap,
};
```

### Initialization Sequence

1. Allocate device structure
2. Register character device region
3. Initialize cdev with file operations
4. Create device class
5. Create device node (`/dev/rocm_gpu0`)
6. Initialize mutex and lists
7. Set initial GPU state

### Cleanup Sequence

1. Free all allocated memory objects
2. Destroy device node
3. Destroy device class
4. Delete cdev
5. Unregister character device region
6. Free device structure

## Userspace Library

### Context Structure

```c
struct rocm_context_s {
    int fd;                        // Device file descriptor
    rocm_gpu_info_t gpu_info;      // Cached GPU information
};
```

### API Layers

```
Application API
    ↓
High-level Helpers (memcpy, create_cmdbuf)
    ↓
Core API (malloc, free, submit)
    ↓
IOCTL Wrappers
    ↓
Kernel Driver
```

### Error Handling

All functions return error codes:
- `ROCM_SUCCESS` (0) - Operation succeeded
- `ROCM_ERROR_INVALID` (-1) - Invalid parameter
- `ROCM_ERROR_NOMEM` (-2) - Out of memory
- `ROCM_ERROR_IO` (-3) - I/O error
- `ROCM_ERROR_NOT_FOUND` (-4) - Resource not found

## Data Flow

### Memory Allocation Flow

```
┌─────────────┐
│ Application │
└──────┬──────┘
       │ rocm_malloc(ctx, &mem, size, flags)
       ↓
┌─────────────┐
│  librocm.so │
└──────┬──────┘
       │ ioctl(fd, ROCM_IOCTL_ALLOC_MEM, &req)
       ↓
┌─────────────────┐
│ Kernel Driver   │
│                 │
│ 1. Create mem_object
│ 2. dma_alloc_coherent()
│ 3. Assign handle
│ 4. Add to list
│ 5. Return handle + GPU addr
└──────┬──────────┘
       │ Return success
       ↓
┌─────────────┐
│  librocm.so │
└──────┬──────┘
       │ Return handle to app
       ↓
┌─────────────┐
│ Application │
└─────────────┘
```

### Memory Copy Flow (H2D)

```
Application Data (Host)
       ↓
1. rocm_memcpy_h2d(ctx, dst_mem, src_ptr, size)
       ↓
2. rocm_map_memory(ctx, dst_mem, &mapped, size)
       ↓
3. mmap(NULL, size, ..., fd, dst_mem)
       ↓
4. Kernel: remap_pfn_range()
       ↓
5. Userspace: memcpy(mapped, src_ptr, size)
       ↓
6. rocm_unmap_memory(ctx, mapped, size)
       ↓
7. munmap(mapped, size)
       ↓
Data now in GPU Memory
```

### Command Submission Flow

```
Application
    ↓ rocm_create_cmdbuf()
Command Buffer (CPU)
    ↓ rocm_cmdbuf_add() × N
Commands Built
    ↓ rocm_submit(ctx, cmdbuf, flags)
    
    ┌─────────────────────────────┐
    │ 1. Allocate GPU memory      │
    │ 2. Copy commands to GPU     │
    │ 3. ioctl(SUBMIT_CMD)        │
    └─────────────────────────────┘
                ↓
    ┌─────────────────────────────┐
    │ Kernel Driver               │
    │ 1. Validate cmd buffer      │
    │ 2. Queue commands           │
    │ 3. Execute on GPU           │
    │ 4. Update queue pointers    │
    └─────────────────────────────┘
                ↓
    ┌─────────────────────────────┐
    │ Cleanup                     │
    │ 1. Wait for completion      │
    │ 2. Free command buffer      │
    └─────────────────────────────┘
```

## Memory Management

### DMA Allocation

The kernel driver uses DMA-coherent memory for zero-copy access:

```c
mem_obj->kernel_addr = dma_alloc_coherent(
    rocm_dev->dev,              // Device pointer
    req.size,                   // Size in bytes
    &mem_obj->dma_addr,         // Output: DMA address
    GFP_KERNEL                  // Allocation flags
);
```

**Benefits**:
- No cache coherency issues
- Direct GPU access
- Efficient userspace mapping

### Memory Mapping

Userspace accesses GPU memory via mmap:

```c
void *addr = mmap(
    NULL,                       // Kernel chooses address
    size,                       // Size to map
    PROT_READ | PROT_WRITE,     // Permissions
    MAP_SHARED,                 // Shared with kernel
    fd,                         // Device file descriptor
    handle                      // Offset = memory handle
);
```

In the kernel, `remap_pfn_range()` establishes the mapping:

```c
remap_pfn_range(
    vma,                        // VMA structure
    vma->vm_start,              // Start address
    mem_obj->dma_addr >> PAGE_SHIFT,  // PFN
    size,                       // Size
    pgprot_noncached(vma->vm_page_prot)  // Non-cached
);
```

### Handle System

- Handles are 64-bit unsigned integers
- Start from 1, increment sequentially
- Used for both memory and command buffers
- Validated on every operation

## Command Processing

### Command Buffer Structure

```c
typedef struct {
    uint32_t *commands;         // Command array
    uint32_t size;              // Number of commands
    uint32_t capacity;          // Allocated capacity
} rocm_cmdbuf_t;
```

### Command Format

Commands are 32-bit words with opcode and operands:

```
┌────────────┬────────────────────┐
│  Bits      │  Description       │
├────────────┼────────────────────┤
│  31-24     │  Opcode            │
│  23-0      │  Operand/Address   │
└────────────┴────────────────────┘
```

**Example Commands**:
- `0x01000000` - Upload vertices
- `0x02000000` - Upload colors
- `0x03000000` - Upload indices
- `0xFF000000` - Synchronization barrier

### Submission Process

1. Create command buffer in userspace
2. Add commands using `rocm_cmdbuf_add()`
3. Allocate GPU memory for commands
4. Copy commands to GPU memory
5. Submit via IOCTL with handle
6. Kernel executes commands
7. Free command buffer memory

## IOCTL Interface

### IOCTL Commands

```c
#define ROCM_IOC_MAGIC 'R'
#define ROCM_IOCTL_ALLOC_MEM    _IOWR(ROCM_IOC_MAGIC, 1, struct rocm_mem_alloc)
#define ROCM_IOCTL_FREE_MEM     _IOW(ROCM_IOC_MAGIC, 2, struct rocm_mem_free)
#define ROCM_IOCTL_SUBMIT_CMD   _IOW(ROCM_IOC_MAGIC, 3, struct rocm_cmd_submit)
#define ROCM_IOCTL_GET_INFO     _IOR(ROCM_IOC_MAGIC, 4, struct rocm_gpu_info)
```

### Data Structures

**Memory Allocation**:
```c
struct rocm_mem_alloc {
    uint64_t size;              // Input: size to allocate
    uint64_t handle;            // Output: memory handle
    uint64_t gpu_addr;          // Output: GPU address
};
```

**Command Submission**:
```c
struct rocm_cmd_submit {
    uint64_t cmd_buffer_handle; // Handle of command buffer
    uint32_t cmd_size;          // Size of commands in bytes
    uint32_t flags;             // Submission flags
};
```

**Device Info**:
```c
struct rocm_gpu_info {
    uint32_t compute_units;     // Number of compute units
    uint32_t max_clock_freq;    // Max clock in MHz
    uint64_t vram_size;         // VRAM size in bytes
    char device_name[64];       // Device name string
};
```

## Security Model

### Privilege Separation

- Kernel driver runs with kernel privileges
- Userspace library runs with user privileges
- Device file controls access permissions

### Access Control

```bash
# Default: root only
crw------- 1 root root 240, 0 /dev/rocm_gpu0

# Development: world-writable
crw-rw-rw- 1 root root 240, 0 /dev/rocm_gpu0

# Production: group-based
crw-rw---- 1 root render 240, 0 /dev/rocm_gpu0
```

### Validation

All kernel entry points validate:
1. Handle existence
2. Buffer sizes
3. Memory boundaries
4. User pointers (copy_from_user/copy_to_user)

### Resource Limits

- Maximum memory allocations
- Maximum command buffer size
- Per-process quotas (not implemented in demo)

## Performance

### Optimization Strategies

**Memory**:
- DMA-coherent allocation (zero-copy)
- Memory mapping (avoid data copies)
- Handle-based tracking (O(1) lookups)

**Synchronization**:
- Single global lock (simple, correct)
- Per-object locks (for production)
- Lock-free queues (advanced)

**Command Processing**:
- Batch command submission
- Asynchronous execution
- Command buffer reuse

### Bottlenecks

1. **System calls**: Each IOCTL is a context switch
2. **Memory copies**: Even with mmap, initial setup copies
3. **Synchronization**: Mutex contention with multiple threads

### Profiling

```bash
# Kernel profiling
perf record -g -e syscalls:sys_enter_ioctl ./rocm_opengl_app
perf report

# Trace IOCTL calls
strace -e ioctl ./rocm_opengl_app

# Memory profiling
valgrind --tool=massif ./rocm_opengl_app
```

## Extending the Driver

### Adding a New IOCTL

**1. Define in kernel header**:
```c
struct rocm_new_feature {
    uint64_t param1;
    uint32_t param2;
};
#define ROCM_IOCTL_NEW_FEATURE _IOWR(ROCM_IOC_MAGIC, 6, struct rocm_new_feature)
```

**2. Implement handler**:
```c
static long rocm_new_feature(struct rocm_new_feature __user *arg) {
    struct rocm_new_feature req;
    
    if (copy_from_user(&req, arg, sizeof(req)))
        return -EFAULT;
    
    // Implementation
    
    if (copy_to_user(arg, &req, sizeof(req)))
        return -EFAULT;
    
    return 0;
}
```

**3. Add to ioctl switch**:
```c
case ROCM_IOCTL_NEW_FEATURE:
    return rocm_new_feature((struct rocm_new_feature __user *)arg);
```

**4. Add userspace wrapper**:
```c
int rocm_new_feature(rocm_context_t ctx, param1, param2) {
    struct rocm_new_feature req;
    req.param1 = param1;
    req.param2 = param2;
    
    if (ioctl(ctx->fd, ROCM_IOCTL_NEW_FEATURE, &req) < 0)
        return ROCM_ERROR_IO;
    
    return ROCM_SUCCESS;
}
```

### Adding Interrupt Support

```c
// In init
if (request_irq(IRQ_NUMBER, rocm_irq_handler, IRQF_SHARED, 
                DRIVER_NAME, rocm_dev)) {
    // Error handling
}

// Handler
static irqreturn_t rocm_irq_handler(int irq, void *dev_id) {
    struct rocm_device *dev = dev_id;
    
    // Read interrupt status
    // Process interrupt
    // Clear interrupt
    
    return IRQ_HANDLED;
}
```

### Adding Async Operations

```c
// Add wait queue
wait_queue_head_t cmd_wait_queue;

// In submit
add_wait_queue(&rocm_dev->cmd_wait_queue, &wait);

// In completion
wake_up_interruptible(&rocm_dev->cmd_wait_queue);
```

## Comparison with Production ROCm

### Implemented Features
✓ Character device interface
✓ Memory allocation/mapping
✓ Command submission
✓ IOCTL communication
✓ Handle-based resource management

### Missing Features (Production ROCm)
✗ Multiple queue support
✗ Interrupt handling
✗ Power management
✗ Thermal management
✗ Firmware loading
✗ Multi-GPU support
✗ PCIe BAR mapping
✗ Hardware scheduling
✗ Virtual memory management
✗ Process isolation

## Debugging Guide

### Kernel Messages

```bash
# All ROCM messages
dmesg | grep ROCM

# Real-time monitoring
dmesg -w | grep ROCM

# With timestamps
dmesg -T | grep ROCM
```

### Module Debugging

```bash
# Load with debug symbols
sudo insmod rocm_gpu_driver.ko

# Check loaded
lsmod | grep rocm

# Module parameters
cat /sys/module/rocm_gpu_driver/parameters/*

# Force unload
sudo rmmod -f rocm_gpu_driver
```

### Application Debugging

```bash
# GDB
gdb ./rocm_opengl_app
(gdb) break rocm_init
(gdb) run

# Valgrind
valgrind --leak-check=full ./rocm_opengl_app

# Strace
strace -e ioctl,mmap ./rocm_opengl_app
```

## References

1. Linux Device Drivers (3rd Edition) - Corbet, Rubini, Kroah-Hartman
2. AMD ROCm Documentation - https://rocm.docs.amd.com/
3. Linux Kernel Documentation - https://www.kernel.org/doc/
4. OpenGL Programming Guide
5. DMA API - https://www.kernel.org/doc/html/latest/core-api/dma-api.html
