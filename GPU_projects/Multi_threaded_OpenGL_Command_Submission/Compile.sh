#!/bin/bash

echo "Compiling multithreaded OpenGL renderer..."

gcc -Wall -Wextra -std=c11 thread.c -o thread_renderer -lGL -lGLEW -lglfw -lpthread -lm

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Run with: ./thread_renderer"
else
    echo "✗ Compilation failed!"
    echo ""
    echo "Make sure you have the required libraries installed:"
    echo "  Ubuntu/Debian: sudo apt-get install libgl1-mesa-dev libglew-dev libglfw3-dev"
    echo "  Fedora/RHEL:   sudo dnf install mesa-libGL-devel glew-devel glfw-devel"
    echo "  Arch Linux:    sudo pacman -S mesa glew glfw-x11"
    exit 1
fi