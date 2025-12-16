# ============================================================
#  Makefile — Build all HIP API demos
#
#  Usage:
#    make               # build all
#    make run_all       # build and run all sequentially
#    make 01_demo       # build specific demo
#    make clean         # remove all binaries
#
#  Prerequisites: ROCm installed, hipcc in PATH
# ============================================================

HIPCC    := hipcc
CXXFLAGS := -O2 -Wall -std=c++17

# GPU architecture — auto-detect or override:
#   make ARCH=gfx90a     (MI250X)
#   make ARCH=gfx942     (MI300X)
#   make ARCH=gfx1100    (RX 7900 XTX)
ifdef ARCH
    CXXFLAGS += --offload-arch=$(ARCH)
endif

SRCS := 01_hip_malloc_memcpy_free.cpp   \
        02_global_kernel_launch.cpp      \
        03_device_synchronize.cpp        \
        04_streams_events.cpp            \
        05_shared_memory_syncthreads.cpp \
        06_warpsize_atomics.cpp          \
        07_visible_devices_and_master.cpp

BINS := $(SRCS:.cpp=)

.PHONY: all run_all clean info

all: $(BINS)

%: %.cpp
	@echo "  Compiling $< → $@"
	$(HIPCC) $(CXXFLAGS) $< -o $@

run_all: all
	@echo "============================================================"
	@echo " Running all HIP API demos"
	@echo "============================================================"
	@for bin in $(BINS); do \
	    echo ""; \
	    echo ">>> $$bin"; \
	    echo "------------------------------------------------------------"; \
	    ./$$bin || echo "  [SKIPPED — GPU not available or error]"; \
	done

clean:
	rm -f $(BINS)
	@echo "Cleaned."

info:
	@echo "ROCm / HIP environment:"
	@which hipcc 2>/dev/null && hipcc --version | head -3 || echo "  hipcc not found"
	@rocminfo 2>/dev/null | grep -E 'Name.*:|Arch.*:|CU.*Count' | head -10 || echo "  rocminfo not found"
	@echo ""
	@echo "Available targets:"
	@for b in $(BINS); do echo "  make $$b"; done
