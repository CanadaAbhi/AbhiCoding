/*
 * librocm.h - ROCm Userspace Driver Library Header
 * 
 * This library provides userspace API for interacting with the ROCm kernel driver
 */

 #ifndef LIBROCM_H
 #define LIBROCM_H
 
 #include <stdint.h>
 #include <stddef.h>
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /* Error codes */
 #define ROCM_SUCCESS           0
 #define ROCM_ERROR_INVALID    -1
 #define ROCM_ERROR_NOMEM      -2
 #define ROCM_ERROR_IO         -3
 #define ROCM_ERROR_NOT_FOUND  -4
 
 /* Memory flags */
 #define ROCM_MEM_READ_WRITE   0x01
 #define ROCM_MEM_READ_ONLY    0x02
 #define ROCM_MEM_WRITE_ONLY   0x04
 
 /* Command flags */
 #define ROCM_CMD_COMPUTE      0x01
 #define ROCM_CMD_GRAPHICS     0x02
 #define ROCM_CMD_SYNC         0x04
 
 /* Opaque handle types */
 typedef struct rocm_context_s* rocm_context_t;
 typedef uint64_t rocm_mem_t;
 typedef uint64_t rocm_queue_t;
 
 /* GPU Information */
 typedef struct {
     uint32_t compute_units;
     uint32_t max_clock_freq;
     uint64_t vram_size;
     char device_name[64];
 } rocm_gpu_info_t;
 
 /* Command buffer */
 typedef struct {
     uint32_t *commands;
     uint32_t size;
     uint32_t capacity;
 } rocm_cmdbuf_t;
 
 /* API Functions */
 
 /**
  * Initialize ROCm runtime and create context
  */
 int rocm_init(rocm_context_t *ctx);
 
 /**
  * Destroy ROCm context and cleanup
  */
 int rocm_destroy(rocm_context_t ctx);
 
 /**
  * Get GPU device information
  */
 int rocm_get_device_info(rocm_context_t ctx, rocm_gpu_info_t *info);
 
 /**
  * Allocate GPU memory
  */
 int rocm_malloc(rocm_context_t ctx, rocm_mem_t *mem, size_t size, uint32_t flags);
 
 /**
  * Free GPU memory
  */
 int rocm_free(rocm_context_t ctx, rocm_mem_t mem);
 
 /**
  * Map GPU memory to CPU address space
  */
 int rocm_map_memory(rocm_context_t ctx, rocm_mem_t mem, void **cpu_addr, size_t size);
 
 /**
  * Unmap GPU memory
  */
 int rocm_unmap_memory(rocm_context_t ctx, void *cpu_addr, size_t size);
 
 /**
  * Copy data from host to device
  */
 int rocm_memcpy_h2d(rocm_context_t ctx, rocm_mem_t dst, const void *src, size_t size);
 
 /**
  * Copy data from device to host
  */
 int rocm_memcpy_d2h(rocm_context_t ctx, void *dst, rocm_mem_t src, size_t size);
 
 /**
  * Create command buffer
  */
 int rocm_create_cmdbuf(rocm_cmdbuf_t **cmdbuf, uint32_t initial_capacity);
 
 /**
  * Destroy command buffer
  */
 int rocm_destroy_cmdbuf(rocm_cmdbuf_t *cmdbuf);
 
 /**
  * Add command to buffer
  */
 int rocm_cmdbuf_add(rocm_cmdbuf_t *cmdbuf, uint32_t cmd);
 
 /**
  * Submit command buffer to GPU
  */
 int rocm_submit(rocm_context_t ctx, rocm_cmdbuf_t *cmdbuf, uint32_t flags);
 
 /**
  * Wait for GPU to finish processing
  */
 int rocm_sync(rocm_context_t ctx);
 
 /**
  * Get error string
  */
 const char* rocm_get_error_string(int error_code);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* LIBROCM_H */
 