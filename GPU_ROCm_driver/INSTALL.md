# ROCm Driver Stack - Installation & Testing Guide

## Prerequisites Installation

### Ubuntu/Debian 20.04+
```bash
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y libgl1-mesa-dev libglew-dev libglfw3-dev
sudo apt-get install -y pkg-config
```

### Ubuntu/Debian (Alternative)
```bash
sudo apt install -y build-essential linux-headers-generic
sudo apt install -y mesa-common-dev libglew2.1 libglfw3-dev
```

### Fedora/RHEL/CentOS 8+
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install kernel-devel kernel-headers
sudo dnf install mesa-libGL-devel glew-devel glfw-devel
```

### Arch Linux
```bash
sudo pacman -Syu
sudo pacman -S base-devel linux-headers
sudo pacman -S mesa glew glfw-x11
```

## Step-by-Step Installation

### Step 1: Verify Prerequisites
```bash
# Check kernel headers
ls /lib/modules/$(uname -r)/build
# Should show kernel source tree

# Check GCC
gcc --version
# Should show GCC version 7.0 or higher

# Check OpenGL libraries
pkg-config --modversion gl glew glfw3
# Should show library versions
```

### Step 2: Extract and Build
```bash
# Navigate to project directory
cd rocm-driver-stack

# Make build script executable
chmod +x build.sh

# Run build
./build.sh
```

**Expected Output**:
```
╔════════════════════════════════════════════════════════╗
║     ROCm GPU Driver Stack Build Script                ║
╚════════════════════════════════════════════════════════╝

[1/5] Checking dependencies...
✓ All dependencies satisfied

[2/5] Building kernel module...
make -C /lib/modules/.../build M=/path/to/dir modules
  CC [M]  /path/to/rocm_gpu_driver.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /path/to/rocm_gpu_driver.mod.o
  LD [M]  /path/to/rocm_gpu_driver.ko
✓ Kernel module built successfully

[3/5] Building userspace library...
gcc -Wall -Wextra -std=c11 -fPIC -shared librocm.c -o librocm.so
✓ Userspace library built successfully

[4/5] Building OpenGL application...
gcc -Wall -Wextra -std=c11 rocm_opengl_app.c -o rocm_opengl_app ...
✓ Application built successfully

[5/5] Build Summary

✓ Build completed successfully!
```

### Step 3: Load Kernel Module
```bash
# Load the module
sudo insmod rocm_gpu_driver.ko

# Verify it loaded
lsmod | grep rocm
# Should show: rocm_gpu_driver

# Check kernel messages
dmesg | tail -10
# Should show:
#   ROCM: Initializing GPU driver
#   ROCM: Driver initialized successfully (major=XXX, minor=0)
```

### Step 4: Verify Device Creation
```bash
# Check device file
ls -l /dev/rocm_gpu0
# Should show: crw------- 1 root root 240, 0 ... /dev/rocm_gpu0

# For testing, make it world-accessible
sudo chmod 666 /dev/rocm_gpu0

# Verify permissions
ls -l /dev/rocm_gpu0
# Should show: crw-rw-rw- 1 root root 240, 0 ... /dev/rocm_gpu0
```

### Step 5: Run Application
```bash
# Run the demo application
./rocm_opengl_app
```

**Expected Output**:
```
╔════════════════════════════════════════════════════════╗
║     ROCm Kernel + Userspace + OpenGL Application      ║
╚════════════════════════════════════════════════════════╝

ROCm: Initialized context (fd=3)
  Device: Simulated ROCM GPU
  Compute Units: 64
  Max Clock: 2400 MHz
  VRAM: 16 GB

=== ROCm GPU Memory Allocated ===
Vertex buffer handle: 1
Color buffer handle: 2

=== Transferring Data via ROCm ===
ROCm: Copied 36 bytes from host to device
ROCm: Copied 36 bytes from host to device
ROCm: Submitted 5 commands to GPU

=== OpenGL Buffers Created ===
VAO: 1
Vertex VBO: 2
Color VBO: 3

╔════════════════════════════════════════════════════════╗
║       ROCm + OpenGL Integration Statistics            ║
╠════════════════════════════════════════════════════════╣
║ GPU Device: Simulated ROCM GPU                         ║
║ Compute Units: 64                                      ║
║ Max Clock: 2400 MHz                                    ║
║ VRAM Size: 16 GB                                       ║
╠════════════════════════════════════════════════════════╣
║ ROCm Memory Allocated:                                 ║
║   Vertex Buffer: 1 (handle)                            ║
║   Color Buffer:  2 (handle)                            ║
╠════════════════════════════════════════════════════════╣
║ OpenGL Resources:                                      ║
║   Shader Program: 1                                    ║
║   VAO: 1                                               ║
║   Vertex VBO: 2                                        ║
║   Color VBO: 3                                         ║
╚════════════════════════════════════════════════════════╝

Starting render loop...
Press ESC or close window to exit

Frame: 60 (ROCm+OpenGL rendering)
Frame: 120 (ROCm+OpenGL rendering)
...
```

**Visual Result**: A window opens showing a rotating triangle with red, green, and blue vertices.

### Step 6: Monitor System
```bash
# In another terminal, watch kernel messages
dmesg -w | grep ROCM

# You should see:
# ROCm: Allocated memory
# ROCm: Copied data
# ROCm: Submitted commands
```

### Step 7: Cleanup
```bash
# Close the application (ESC key or close window)

# Unload kernel module
sudo rmmod rocm_gpu_driver

# Verify unloaded
lsmod | grep rocm
# Should show nothing

# Check cleanup messages
dmesg | tail -10
# Should show:
#   ROCM: Cleaning up driver
#   ROCM: Driver unloaded
```

## Testing Checklist

### Basic Functionality
- [ ] Module loads without errors
- [ ] Device file `/dev/rocm_gpu0` created
- [ ] Application connects to device
- [ ] Memory allocation succeeds
- [ ] Data transfer works
- [ ] Commands submit successfully
- [ ] OpenGL window opens
- [ ] Triangle renders and rotates
- [ ] Application exits cleanly
- [ ] Module unloads without errors

### Advanced Testing

#### Memory Stress Test
```bash
# Run application multiple times
for i in {1..10}; do
    echo "Run $i"
    timeout 5s ./rocm_opengl_app || true
    sleep 1
done

# Check for memory leaks
dmesg | grep -i "memory\|leak"
```

#### Multi-Process Test
```bash
# In terminal 1
./rocm_opengl_app &

# In terminal 2
./rocm_opengl_app &

# Both should work (or fail gracefully)
```

#### Error Handling Test
```bash
# Try running without loading module
./rocm_opengl_app
# Should fail with clear error message

# Try loading module twice
sudo insmod rocm_gpu_driver.ko
sudo insmod rocm_gpu_driver.ko
# Second should fail appropriately
```

## Troubleshooting

### Issue: Module fails to load

**Error**: `insmod: ERROR: could not insert module`

**Solutions**:
```bash
# Check detailed error
sudo dmesg | tail -20

# Common causes:
# 1. Kernel headers mismatch
uname -r
ls /lib/modules/$(uname -r)/build

# 2. Symbol resolution issues
modprobe --show-depends rocm_gpu_driver

# 3. Already loaded
lsmod | grep rocm
sudo rmmod rocm_gpu_driver  # Unload first
```

### Issue: Device not created

**Error**: `/dev/rocm_gpu0` doesn't exist

**Solutions**:
```bash
# Check module loaded
lsmod | grep rocm

# Check kernel logs
dmesg | grep ROCM

# Manually create device (emergency)
sudo mknod /dev/rocm_gpu0 c 240 0
sudo chmod 666 /dev/rocm_gpu0
```

### Issue: Application can't open device

**Error**: `Failed to open ROCm device: Permission denied`

**Solutions**:
```bash
# Check permissions
ls -l /dev/rocm_gpu0

# Fix permissions
sudo chmod 666 /dev/rocm_gpu0

# Or run as root
sudo ./rocm_opengl_app
```

### Issue: Library not found

**Error**: `error while loading shared libraries: librocm.so`

**Solutions**:
```bash
# Set library path
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

# Or copy to system location
sudo cp librocm.so /usr/local/lib/
sudo ldconfig

# Or use rpath (rebuild needed)
gcc ... -Wl,-rpath,/path/to/lib
```

### Issue: OpenGL window doesn't open

**Error**: `Failed to initialize GLFW`

**Solutions**:
```bash
# Check display
echo $DISPLAY

# For SSH sessions
export DISPLAY=:0

# For WSL
export DISPLAY=:0
# May need VcXsrv or similar X server

# Check OpenGL support
glxinfo | grep "OpenGL version"
```

### Issue: Segmentation fault

**Solutions**:
```bash
# Run with gdb
gdb ./rocm_opengl_app
(gdb) run
# When it crashes:
(gdb) backtrace

# Check kernel logs
dmesg | tail -50

# Run with valgrind
valgrind --leak-check=full ./rocm_opengl_app
```

## Performance Verification

### Check Frame Rate
```bash
# Monitor frames
./rocm_opengl_app 2>&1 | grep "Frame:"

# Should see regular frame updates (60 FPS target)
```

### Memory Usage
```bash
# While application running, in another terminal:
ps aux | grep rocm_opengl_app

# Check kernel memory
cat /proc/meminfo | grep -i slab
```

### System Calls
```bash
# Trace ioctl calls
strace -e ioctl ./rocm_opengl_app 2>&1 | grep ROCM

# Count system calls
strace -c ./rocm_opengl_app
```

## Production Deployment

### Create Udev Rules
```bash
# Create rule file
sudo tee /etc/udev/rules.d/99-rocm.rules << EOF
KERNEL=="rocm_gpu*", MODE="0666", GROUP="video"
EOF

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Auto-load Module at Boot
```bash
# Copy module to system location
sudo cp rocm_gpu_driver.ko /lib/modules/$(uname -r)/extra/

# Update module database
sudo depmod -a

# Add to auto-load
echo "rocm_gpu_driver" | sudo tee -a /etc/modules-load.d/rocm.conf

# Verify
sudo modprobe rocm_gpu_driver
```

### Install Library System-wide
```bash
# Copy library
sudo cp librocm.so /usr/local/lib/
sudo cp librocm.h /usr/local/include/

# Update library cache
sudo ldconfig

# Verify
ldconfig -p | grep rocm
```

## Next Steps

After successful installation and testing:

1. **Read ARCHITECTURE.md** - Understand the internals
2. **Modify the code** - Experiment with changes
3. **Add features** - Extend the driver
4. **Profile performance** - Optimize bottlenecks
5. **Write applications** - Build on the API

## Support

For issues:
1. Check kernel logs: `dmesg | grep ROCM`
2. Verify dependencies
3. Review QUICKREF.md
4. Check ARCHITECTURE.md for details
