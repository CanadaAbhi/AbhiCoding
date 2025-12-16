# Simple Makefile for First-Person Camera Demo
# Alternative to CMake for quick compilation

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
INCLUDES = -I. -I./include
LIBS = -lglfw -lGL -ldl -lpthread

# Source files
SOURCES = main.cpp glad.c
HEADERS = Camera.h Collision.h

# Output
TARGET = camera_demo

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

# Build instructions:
# 1. Download GLAD from https://glad.dav1d.de/
# 2. Place glad.c in current directory
# 3. Place glad.h in ./include/glad/glad.h
# 4. Place KHR/khrplatform.h in ./include/KHR/khrplatform.h
# 5. Run: make
# 6. Run: ./camera_demo