#!/bin/bash

# ROCm GPU Driver Stack Build Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ROCm GPU Driver Stack Build Script                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root for kernel module
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root${NC}"
fi

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}[1/5] Checking dependencies...${NC}"
    
    local missing=""
    
    # Check for kernel headers
    if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
        missing="$missing linux-headers-$(uname -r)"
    fi
    
    # Check for build tools
    command -v gcc >/dev/null 2>&1 || missing="$missing gcc"
    command -v make >/dev/null 2>&1 || missing="$missing make"
    
    # Check for OpenGL libraries
    if ! pkg-config --exists gl glew glfw3 2>/dev/null; then
        missing="$missing libgl1-mesa-dev libglew-dev libglfw3-dev"
    fi
    
    if [ -n "$missing" ]; then
        echo -e "${RED}Missing dependencies:${NC} $missing"
        echo ""
        echo "Install with:"
        echo "  Ubuntu/Debian: sudo apt-get install $missing"
        echo "  Fedora/RHEL:   sudo dnf install kernel-devel gcc make mesa-libGL-devel glew-devel glfw-devel"
        echo "  Arch Linux:    sudo pacman -S linux-headers gcc make mesa glew glfw-x11"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All dependencies satisfied${NC}"
}

# Build kernel module
build_kernel_module() {
    echo ""
    echo -e "${BLUE}[2/5] Building kernel module...${NC}"
    
    make -f Makefile.kernel clean 2>/dev/null || true
    make -f Makefile.kernel
    
    if [ -f "rocm_gpu_driver.ko" ]; then
        echo -e "${GREEN}✓ Kernel module built successfully${NC}"
    else
        echo -e "${RED}✗ Kernel module build failed${NC}"
        exit 1
    fi
}

# Build userspace library
build_userspace_library() {
    echo ""
    echo -e "${BLUE}[3/5] Building userspace library...${NC}"
    
    make -f Makefile.userspace clean 2>/dev/null || true
    make -f Makefile.userspace librocm.so
    
    if [ -f "librocm.so" ]; then
        echo -e "${GREEN}✓ Userspace library built successfully${NC}"
    else
        echo -e "${RED}✗ Userspace library build failed${NC}"
        exit 1
    fi
}

# Build application
build_application() {
    echo ""
    echo -e "${BLUE}[4/5] Building OpenGL application...${NC}"
    
    make -f Makefile.userspace rocm_opengl_app
    
    if [ -f "rocm_opengl_app" ]; then
        echo -e "${GREEN}✓ Application built successfully${NC}"
    else
        echo -e "${RED}✗ Application build failed${NC}"
        exit 1
    fi
}

# Summary
print_summary() {
    echo ""
    echo -e "${BLUE}[5/5] Build Summary${NC}"
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo ""
    echo "Generated files:"
    echo "  • rocm_gpu_driver.ko  - Kernel module"
    echo "  • librocm.so          - Userspace library"
    echo "  • rocm_opengl_app     - OpenGL application"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo ""
    echo "1. Load the kernel module:"
    echo "   ${GREEN}sudo insmod rocm_gpu_driver.ko${NC}"
    echo ""
    echo "2. Verify the device was created:"
    echo "   ${GREEN}ls -l /dev/rocm_gpu0${NC}"
    echo ""
    echo "3. Check kernel messages:"
    echo "   ${GREEN}dmesg | tail -20${NC}"
    echo ""
    echo "4. Run the application:"
    echo "   ${GREEN}./rocm_opengl_app${NC}"
    echo ""
    echo "5. To unload the module:"
    echo "   ${GREEN}sudo rmmod rocm_gpu_driver${NC}"
    echo ""
}

# Main build process
main() {
    check_dependencies
    build_kernel_module
    build_userspace_library
    build_application
    print_summary
}

# Run main
main
