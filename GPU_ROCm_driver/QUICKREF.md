# ROCm Driver Stack - Quick Reference

## Build Commands

```bash
# Automated build (recommended)
./build.sh

# Manual kernel module build
make -f Makefile.kernel

# Manual userspace build
make -f Makefile.userspace
```

## Load/Unload Module

```bash
# Load
sudo insmod rocm_gpu_driver.ko

# Check loaded
lsmod | grep rocm

# View messages
dmesg | tail -20

# Unload
sudo rmmod rocm_gpu_driver
```

## Run Application

```bash
# Basic run
./rocm_opengl_app

# With library path
LD_LIBRARY_PATH=. ./rocm_opengl_app

# With debugging
gdb ./rocm_opengl_app
```

## Device Access

```bash
# Check device
ls -l /dev/rocm_gpu0

# Fix permissions (development only)
sudo chmod 666 /dev/rocm_gpu0

# Create udev rule (production)
echo 'KERNEL=="rocm_gpu*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-rocm.rules
sudo udevadm control --reload-rules
```

## Debugging

```bash
# Kernel messages
dmesg | grep ROCM
dmesg -w | grep ROCM          # Watch mode

# System calls
strace -e ioctl ./rocm_opengl_app

# Memory leaks
valgrind --leak-check=full ./rocm_opengl_app

# Library dependencies
ldd rocm_opengl_app
```

## API Quick Reference

### Initialization
```c
rocm_context_t ctx;
rocm_init(&ctx);              // Initialize context
rocm_get_device_info(ctx, &info);  // Get GPU info
rocm_destroy(ctx);            // Cleanup
```

### Memory Management
```c
rocm_mem_t mem;
rocm_malloc(ctx, &mem, size, flags);  // Allocate
rocm_free(ctx, mem);          // Free

void *ptr;
rocm_map_memory(ctx, mem, &ptr, size);   // Map
rocm_unmap_memory(ctx, ptr, size);       // Unmap
```

### Data Transfer
```c
rocm_memcpy_h2d(ctx, dst_mem, src_ptr, size);  // Host to device
rocm_memcpy_d2h(ctx, dst_ptr, src_mem, size);  // Device to host
```

### Command Submission
```c
rocm_cmdbuf_t *cmdbuf;
rocm_create_cmdbuf(&cmdbuf, capacity);     // Create
rocm_cmdbuf_add(cmdbuf, cmd);              // Add command
rocm_submit(ctx, cmdbuf, flags);           // Submit
rocm_sync(ctx);                            // Wait
rocm_destroy_cmdbuf(cmdbuf);               // Cleanup
```

## Common Issues

### Module won't load
- Check kernel headers: `ls /lib/modules/$(uname -r)/build`
- Install headers: `sudo apt-get install linux-headers-$(uname -r)`

### Device not found
- Check module loaded: `lsmod | grep rocm`
- Check device exists: `ls -l /dev/rocm_gpu0`
- Check dmesg: `dmesg | grep ROCM`

### Permission denied
- Fix permissions: `sudo chmod 666 /dev/rocm_gpu0`
- Or use sudo: `sudo ./rocm_opengl_app`

### Library not found
- Set path: `export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH`
- Or install: `sudo cp librocm.so /usr/local/lib && sudo ldconfig`

### Application crashes
- Check kernel messages: `dmesg | tail -30`
- Run with gdb: `gdb ./rocm_opengl_app`
- Check valgrind: `valgrind ./rocm_opengl_app`

## File Structure

```
rocm-driver/
├── rocm_gpu_driver.c      # Kernel module source
├── librocm.h              # Userspace API header
├── librocm.c              # Userspace implementation
├── rocm_opengl_app.c      # Demo application
├── Makefile.kernel        # Kernel build system
├── Makefile.userspace     # Userspace build system
├── build.sh               # Build automation
├── README.md              # Main documentation
└── ARCHITECTURE.md        # Detailed architecture
```

## Performance Tips

1. **Batch operations**: Submit multiple commands at once
2. **Reuse buffers**: Avoid frequent alloc/free
3. **Use mapping**: Direct memory access vs copies
4. **Minimize ioctl**: Each call is a syscall
5. **Async submission**: Don't wait unless necessary

## Production Checklist

- [ ] Implement proper error handling
- [ ] Add logging/tracing
- [ ] Security audit
- [ ] Performance profiling
- [ ] Memory leak testing
- [ ] Multi-threading tests
- [ ] Resource cleanup verification
- [ ] Documentation updates
- [ ] Udev rules setup
- [ ] SELinux/AppArmor policies

## Further Reading

- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - Detailed technical documentation
- Kernel source comments - Implementation details
- ROCm documentation - Real-world comparison
