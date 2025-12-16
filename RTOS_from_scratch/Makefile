# Makefile for Minimal RTOS

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -Iinclude -std=c11
LDFLAGS = -lpthread

# Directories
SRC_DIR = src
INC_DIR = include
EXAMPLE_DIR = examples
BUILD_DIR = build
BIN_DIR = bin

# Source files
RTOS_SRCS = $(wildcard $(SRC_DIR)/*.c)
EXAMPLE_SRCS = $(wildcard $(EXAMPLE_DIR)/*.c)

# Object files
RTOS_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(RTOS_SRCS))
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%.o,$(EXAMPLE_SRCS))

# Output binaries
EXAMPLE_BINS = $(patsubst $(EXAMPLE_DIR)/%.c,$(BIN_DIR)/%,$(EXAMPLE_SRCS))

# Default target
all: directories $(EXAMPLE_BINS)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Build RTOS objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

# Build example objects
$(BUILD_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

# Link examples
$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(RTOS_OBJS)
	@echo "Linking $@..."
	@$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Run example
run: all
	@echo "Running example application..."
	@./$(BIN_DIR)/example_app

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

# Clean and rebuild
rebuild: clean all

# Help
help:
	@echo "Minimal RTOS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all examples (default)"
	@echo "  run      - Build and run example application"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and rebuild"
	@echo "  help     - Show this help message"

.PHONY: all directories run clean rebuild help