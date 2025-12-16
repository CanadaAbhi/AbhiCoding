# ================================================================
#  Makefile — HIP Memory Management API Demos
#
#  BUILD ALL:    make
#  RUN ALL:      make run_all
#  SINGLE:       make 01_malloc_free && ./01_malloc_free
#  CLEAN:        make clean
#  ARCH OVERRIDE: make ARCH=gfx1100   (RX 7900)
#                 make ARCH=gfx90a    (MI250X)
#                 make ARCH=gfx942    (MI300X)
# ================================================================

HIPCC    := hipcc
CXXFLAGS := -O2 -Wall -std=c++17

ifdef ARCH
    CXXFLAGS += --offload-arch=$(ARCH)
endif

SRCS := 01_hip_malloc_free.cpp           \
        02_hip_memcpy.cpp                \
        03_hip_memcpy_async.cpp          \
        04_hip_host_malloc_free.cpp      \
        05_hip_malloc_managed.cpp        \
        06_memset_meminfo_attributes.cpp \
        07_memory_integration.cpp

BINS := $(SRCS:.cpp=)

.PHONY: all run_all clean info

all: $(BINS)

# Pattern rule: compile each .cpp → binary
%: %.cpp
	@echo "  Compiling $< → $@"
	$(HIPCC) $(CXXFLAGS) $< -o $@

run_all: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║    Running All HIP Memory Management API Demos      ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@for bin in $(BINS); do                                       \
	    echo "";                                                  \
	    echo ">>> $$bin";                                         \
	    echo "────────────────────────────────────────────────────";\
	    ./$$bin || echo "  [SKIP — GPU unavailable]";             \
	done

clean:
	rm -f $(BINS)
	@echo "Cleaned."

info:
	@echo "=== ROCm Environment ==="
	@which hipcc 2>/dev/null && hipcc --version | head -2 || echo "hipcc not found"
	@echo ""
	@echo "=== GPU Info ==="
	@rocminfo 2>/dev/null | grep -E 'Name.*:|Arch.*:|CU.*Count|gfx' | head -8 \
	    || echo "rocminfo not found"
	@echo ""
	@echo "=== Build targets ==="
	@for b in $(BINS); do echo "  make $$b"; done

# ── File-by-file description ──────────────────────────────────
help:
	@echo ""
	@echo "HIP Memory Management API Demos"
	@echo ""
	@echo "  01_hip_malloc_free.cpp"
	@echo "     hipMalloc  hipFree  hipMallocPitch  hipMalloc3D"
	@echo ""
	@echo "  02_hip_memcpy.cpp"
	@echo "     hipMemcpy H2D/D2H/D2D/Default  hipMemcpy2D  bandwidth"
	@echo ""
	@echo "  03_hip_memcpy_async.cpp"
	@echo "     hipMemcpyAsync  double-buffer  N-stream pipeline  cross-stream sync"
	@echo ""
	@echo "  04_hip_host_malloc_free.cpp"
	@echo "     hipHostMalloc flags: Default/Mapped/WriteCombined/Portable"
	@echo "     hipHostFree  hipHostGetDevicePointer  bandwidth vs pageable"
	@echo ""
	@echo "  05_hip_malloc_managed.cpp"
	@echo "     hipMallocManaged  hipMemAdvise  hipMemPrefetchAsync"
	@echo "     CPU+GPU shared pointer  UM vs explicit comparison"
	@echo ""
	@echo "  06_memset_meminfo_attributes.cpp"
	@echo "     hipMemset  hipMemsetAsync  hipMemset2D"
	@echo "     hipMemGetInfo  hipPointerGetAttributes"
	@echo ""
	@echo "  07_memory_integration.cpp"
	@echo "     All 10 APIs in MLP forward pass pipeline"
	@echo ""
