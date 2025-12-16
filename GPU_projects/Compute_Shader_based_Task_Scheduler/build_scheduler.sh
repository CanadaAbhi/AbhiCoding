#!/bin/bash

# Shader Task Scheduler Build Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   GPU Compute Shader Task Scheduler - Build Script    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
echo -e "${BLUE}[1/3] Checking dependencies...${NC}"

missing=""

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
    echo "  Fedora/RHEL:   sudo dnf install gcc make mesa-libGL-devel glew-devel glfw-devel"
    echo "  Arch Linux:    sudo pacman -S gcc make mesa glew glfw-x11"
    exit 1
fi

echo -e "${GREEN}✓ All dependencies satisfied${NC}"

# Check OpenGL version
echo ""
echo -e "${BLUE}[2/3] Checking OpenGL support...${NC}"
echo -e "${YELLOW}Note: This program requires OpenGL 4.3+ for compute shader support${NC}"
echo -e "${YELLOW}If the program fails to run, your GPU may not support compute shaders${NC}"

# Build
echo ""
echo -e "${BLUE}[3/3] Building...${NC}"

make -f Makefile.scheduler clean 2>/dev/null || true
make -f Makefile.scheduler

if [ -f "shader_task_scheduler" ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Ready to run!${NC}"
    echo ""
    echo "Run with:"
    echo -e "  ${GREEN}./shader_task_scheduler${NC}"
    echo ""
    echo "What to expect:"
    echo "  • A window showing 10,000 particles"
    echo "  • Particles simulated on GPU using compute shaders"
    echo "  • Physics with gravity and boundary bouncing"
    echo "  • FPS and performance metrics in title bar"
    echo "  • Press ESC to exit"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi