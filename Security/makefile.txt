# Makefile — AES-256-GCM for ARM Linux
# Targets: aarch64 (Cortex-A53/A55/A72) and armv7l

# ----------------------------------------------------------------
# Toolchain Configuration
# ----------------------------------------------------------------
ARCH            ?= aarch64
ifeq ($(ARCH), aarch64)
    CROSS       := aarch64-linux-gnu-
    MARCH       := -march=armv8-a+crypto    # Enable ARMv8 Crypto Ext
    MFLOAT      :=
else ifeq ($(ARCH), armv7)
    CROSS       := arm-linux-gnueabihf-
    MARCH       := -march=armv7-a -mfpu=neon-vfpv4
    MFLOAT      := -mfloat-abi=hard
else
    $(error Unsupported ARCH=$(ARCH). Use aarch64 or armv7)
endif

CC              := $(CROSS)gcc
AR              := $(CROSS)ar
OBJDUMP         := $(CROSS)objdump
SIZE            := $(CROSS)size

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
MBEDTLS_DIR     ?= /opt/mbedtls
MBEDTLS_INC     := $(MBEDTLS_DIR)/include
MBEDTLS_LIB     := $(MBEDTLS_DIR)/library

SRC_DIR         := .
OBJ_DIR         := build/obj
BIN_DIR         := build/bin

# ----------------------------------------------------------------
# Sources
# ----------------------------------------------------------------
SRCS            := $(SRC_DIR)/aes_gcm.c      \
                   $(SRC_DIR)/key_manager.c  \
                   $(SRC_DIR)/secure_mem.c   \
                   $(SRC_DIR)/main.c

OBJS            := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET          := $(BIN_DIR)/aes_gcm_demo

# ----------------------------------------------------------------
# Compiler Flags
# ----------------------------------------------------------------

# Security hardening flags
SECURITY_FLAGS  := \
    -fstack-protector-strong        \
    -fstack-clash-protection        \
    -D_FORTIFY_SOURCE=2             \
    -fPIE                           \
    -fno-strict-overflow            \
    -fwrapv                         \
    -Wformat=2                      \
    -Wformat-security               \
    -Werror=format-security

# Optimization: -O2 for balance; -Os for size-constrained targets
# SECURITY: Never use -O0 in production (misses dead-store elim.
#           of secure_memzero). Never use -O3 (may break ct-code).
OPT_FLAGS       := -O2 -fomit-frame-pointer

# Warning flags — treat warnings as errors in production
WARN_FLAGS      := \
    -Wall -Wextra -Wpedantic        \
    -Wshadow                        \
    -Wpointer-arith                 \
    -Wcast-align                    \
    -Wstrict-prototypes             \
    -Wmissing-prototypes            \
    -Wredundant-decls               \
    -Wnull-dereference              \
    -Werror

# Debug build (adds sanitizers)
DEBUG_FLAGS     := \
    -DDEBUG_BUILD                   \
    -g3 -ggdb                       \
    -fsanitize=address              \
    -fsanitize=undefined            \
    -fno-sanitize-recover=all

CFLAGS          := \
    $(MARCH) $(MFLOAT)              \
    $(OPT_FLAGS)                    \
    $(SECURITY_FLAGS)               \
    $(WARN_FLAGS)                   \
    -std=c11                        \
    -DMBEDTLS_CONFIG_FILE=\"mbedtls_config.h\" \
    -I$(SRC_DIR)                    \
    -I$(MBEDTLS_INC)

# Linker flags
LDFLAGS         := \
    -pie                            \
    -Wl,-z,relro                    \
    -Wl,-z,now                      \
    -Wl,-z,noexecstack              \
    -L$(MBEDTLS_LIB)               \
    -lmbedcrypto                    \
    -lpthread

# ----------------------------------------------------------------
# Build Rules
# ----------------------------------------------------------------
.PHONY: all debug clean size disasm check-tools

all: check-tools $(TARGET)
    @echo "=== Build complete: $(TARGET) ==="
    $(SIZE) $(TARGET)

debug: CFLAGS  += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: all

$(TARGET): $(OBJS) | $(BIN_DIR)
    $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
    $(CC) $(CFLAGS) -c $< -o $@
    @echo "  CC  $<"

$(OBJ_DIR) $(BIN_DIR):
    mkdir -p $@

# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------
size: $(TARGET)
    $(SIZE) --format=berkeley $(TARGET)

disasm: $(TARGET)
    $(OBJDUMP) -d -S $(TARGET) > $(BIN_DIR)/disasm.txt
    @echo "Disassembly written to $(BIN_DIR)/disasm.txt"

check-tools:
    @which $(CC) > /dev/null 2>&1 || \
        (echo "ERROR: $(CC) not found. Install cross-toolchain." && exit 1)

clean:
    rm -rf build/

# ----------------------------------------------------------------
# Static Analysis (requires cppcheck)
# ----------------------------------------------------------------
analyze:
    cppcheck --enable=all              \
             --std=c11                 \
             --platform=unix64        \
             --suppress=missingInclude \
             -I$(MBEDTLS_INC)         \
             $(SRCS)

# ----------------------------------------------------------------
# mbedTLS build helper (builds from source with our config)
# ----------------------------------------------------------------
mbedtls-build:
    @test -d $(MBEDTLS_DIR)/library || \
        (echo "ERROR: mbedTLS not found at $(MBEDTLS_DIR)" && exit 1)
    cd $(MBEDTLS_DIR) && \
    CFLAGS="-DMBEDTLS_CONFIG_FILE='<mbedtls_config.h>' \
            -I$(SRC_DIR) $(MARCH) $(OPT_FLAGS)" \
    make lib CC=$(CC) AR=$(AR)