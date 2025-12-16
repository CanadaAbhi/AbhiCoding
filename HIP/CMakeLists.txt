# ============================================================
#  CMakeLists.txt — Build all HIP API demos
#
#  Usage:
#    mkdir build && cd build
#    cmake .. -DCMAKE_CXX_COMPILER=hipcc
#    make -j$(nproc)
#
#  Or use the Makefile in the parent directory.
# ============================================================
cmake_minimum_required(VERSION 3.21)
project(HIP_API_Demos LANGUAGES CXX)

# Detect ROCm path
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    else()
        message(FATAL_ERROR "ROCm not found. Set ROCM_PATH or install ROCm.")
    endif()
endif()

message(STATUS "ROCm path: ${ROCM_PATH}")

# Use hipcc as the compiler
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# ── List of demo targets ──────────────────────────────────────
set(DEMOS
    01_hip_malloc_memcpy_free
    02_global_kernel_launch
    03_device_synchronize
    04_streams_events
    05_shared_memory_syncthreads
    06_warpsize_atomics
    07_visible_devices_and_master
)

foreach(DEMO ${DEMOS})
    add_executable(${DEMO} ${DEMO}.cpp)
    target_include_directories(${DEMO} PRIVATE
        ${ROCM_PATH}/include
        ${ROCM_PATH}/include/hip
    )
endforeach()

# ── Custom target: run all demos ──────────────────────────────
add_custom_target(run_all
    COMMAND bash -c "for d in ${DEMOS}; do echo '\\n>>> Running '$$d; ./$d || true; done"
    DEPENDS ${DEMOS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all HIP API demos"
)
