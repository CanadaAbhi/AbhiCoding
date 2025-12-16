/*
 * librocm.c - ROCm Userspace Driver Library Implementation
 */

 #include "librocm.h"
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <sys/ioctl.h>
 #include <sys/mman.h>
 #include <errno.h>
 
 #define DEVICE_PATH "/dev/rocm_gpu0"
 
 /* IOCTL definitions - must match kernel driver */
 #define ROCM_IOC_MAGIC 'R'
 
 struct rocm_mem_alloc {
     uint64_t size;
     uint64_t handle;
     uint64_t gpu_addr;
 };
 
 struct rocm_mem_free {
     uint64_t handle;
 };
 
 struct rocm_cmd_submit {
     uint64_t cmd_buffer_handle;
     uint32_t cmd_size;
     uint32_t flags;
 };
 
 struct rocm_gpu_info_ioctl {
     uint32_t compute_units;
     uint32_t max_clock_freq;
     uint64_t vram_size;
     char device_name[64];
 };
 
 #define ROCM_IOCTL_ALLOC_MEM    _IOWR(ROCM_IOC_MAGIC, 1, struct rocm_mem_alloc)
 #define ROCM_IOCTL_FREE_MEM     _IOW(ROCM_IOC_MAGIC, 2, struct rocm_mem_free)
 #define ROCM_IOCTL_SUBMIT_CMD   _IOW(ROCM_IOC_MAGIC, 3, struct rocm_cmd_submit)
 #define ROCM_IOCTL_GET_INFO     _IOR(ROCM_IOC_MAGIC, 4, struct rocm_gpu_info_ioctl)
 
 /* Internal context structure */
 struct rocm_context_s {
     int fd;
     rocm_gpu_info_t gpu_info;
 };
 
 /* Memory mapping structure */
 typedef struct {
     rocm_mem_t handle;
     void *cpu_addr;
     size_t size;
 } mem_mapping_t;
 
 #define MAX_MAPPINGS 1024
 static mem_mapping_t mappings[MAX_MAPPINGS];
 static int mapping_count = 0;
 
 /* Command buffer structure */
 struct rocm_cmdbuf_s {
     uint32_t *commands;
     uint32_t size;
     uint32_t capacity;
 };
 
 /* Initialize ROCm context */
 int rocm_init(rocm_context_t *ctx)
 {
     struct rocm_context_s *context;
     struct rocm_gpu_info_ioctl info;
     
     if (!ctx)
         return ROCM_ERROR_INVALID;
     
     context = (struct rocm_context_s*)malloc(sizeof(struct rocm_context_s));
     if (!context)
         return ROCM_ERROR_NOMEM;
     
     /* Open device */
     context->fd = open(DEVICE_PATH, O_RDWR);
     if (context->fd < 0) {
         perror("Failed to open ROCm device");
         free(context);
         return ROCM_ERROR_IO;
     }
     
     /* Get device info */
     if (ioctl(context->fd, ROCM_IOCTL_GET_INFO, &info) < 0) {
         perror("Failed to get device info");
         close(context->fd);
         free(context);
         return ROCM_ERROR_IO;
     }
     
     context->gpu_info.compute_units = info.compute_units;
     context->gpu_info.max_clock_freq = info.max_clock_freq;
     context->gpu_info.vram_size = info.vram_size;
     strncpy(context->gpu_info.device_name, info.device_name, 
             sizeof(context->gpu_info.device_name) - 1);
     
     *ctx = context;
     
     printf("ROCm: Initialized context (fd=%d)\n", context->fd);
     printf("  Device: %s\n", context->gpu_info.device_name);
     printf("  Compute Units: %u\n", context->gpu_info.compute_units);
     printf("  Max Clock: %u MHz\n", context->gpu_info.max_clock_freq);
     printf("  VRAM: %llu GB\n", context->gpu_info.vram_size / (1024*1024*1024));
     
     return ROCM_SUCCESS;
 }
 
 /* Destroy ROCm context */
 int rocm_destroy(rocm_context_t ctx)
 {
     if (!ctx)
         return ROCM_ERROR_INVALID;
     
     close(ctx->fd);
     free(ctx);
     
     printf("ROCm: Context destroyed\n");
     
     return ROCM_SUCCESS;
 }
 
 /* Get GPU device information */
 int rocm_get_device_info(rocm_context_t ctx, rocm_gpu_info_t *info)
 {
     if (!ctx || !info)
         return ROCM_ERROR_INVALID;
     
     memcpy(info, &ctx->gpu_info, sizeof(rocm_gpu_info_t));
     
     return ROCM_SUCCESS;
 }
 
 /* Allocate GPU memory */
 int rocm_malloc(rocm_context_t ctx, rocm_mem_t *mem, size_t size, uint32_t flags)
 {
     struct rocm_mem_alloc req;
     
     if (!ctx || !mem || size == 0)
         return ROCM_ERROR_INVALID;
     
     memset(&req, 0, sizeof(req));
     req.size = size;
     
     if (ioctl(ctx->fd, ROCM_IOCTL_ALLOC_MEM, &req) < 0) {
         perror("Failed to allocate GPU memory");
         return ROCM_ERROR_IO;
     }
     
     *mem = req.handle;
     
     printf("ROCm: Allocated %zu bytes (handle=%llu, gpu_addr=0x%llx)\n",
            size, req.handle, req.gpu_addr);
     
     return ROCM_SUCCESS;
 }
 
 /* Free GPU memory */
 int rocm_free(rocm_context_t ctx, rocm_mem_t mem)
 {
     struct rocm_mem_free req;
     
     if (!ctx)
         return ROCM_ERROR_INVALID;
     
     req.handle = mem;
     
     if (ioctl(ctx->fd, ROCM_IOCTL_FREE_MEM, &req) < 0) {
         perror("Failed to free GPU memory");
         return ROCM_ERROR_IO;
     }
     
     printf("ROCm: Freed memory handle=%llu\n", mem);
     
     return ROCM_SUCCESS;
 }
 
 /* Map GPU memory to CPU address space */
 int rocm_map_memory(rocm_context_t ctx, rocm_mem_t mem, void **cpu_addr, size_t size)
 {
     void *addr;
     
     if (!ctx || !cpu_addr)
         return ROCM_ERROR_INVALID;
     
     /* Use mmap with handle as offset */
     addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, mem);
     if (addr == MAP_FAILED) {
         perror("Failed to map GPU memory");
         return ROCM_ERROR_IO;
     }
     
     *cpu_addr = addr;
     
     /* Store mapping */
     if (mapping_count < MAX_MAPPINGS) {
         mappings[mapping_count].handle = mem;
         mappings[mapping_count].cpu_addr = addr;
         mappings[mapping_count].size = size;
         mapping_count++;
     }
     
     printf("ROCm: Mapped memory handle=%llu to %p\n", mem, addr);
     
     return ROCM_SUCCESS;
 }
 
 /* Unmap GPU memory */
 int rocm_unmap_memory(rocm_context_t ctx, void *cpu_addr, size_t size)
 {
     if (!ctx || !cpu_addr)
         return ROCM_ERROR_INVALID;
     
     if (munmap(cpu_addr, size) < 0) {
         perror("Failed to unmap memory");
         return ROCM_ERROR_IO;
     }
     
     /* Remove from mappings */
     for (int i = 0; i < mapping_count; i++) {
         if (mappings[i].cpu_addr == cpu_addr) {
             for (int j = i; j < mapping_count - 1; j++) {
                 mappings[j] = mappings[j + 1];
             }
             mapping_count--;
             break;
         }
     }
     
     printf("ROCm: Unmapped memory at %p\n", cpu_addr);
     
     return ROCM_SUCCESS;
 }
 
 /* Copy host to device */
 int rocm_memcpy_h2d(rocm_context_t ctx, rocm_mem_t dst, const void *src, size_t size)
 {
     void *mapped_addr;
     int ret;
     
     ret = rocm_map_memory(ctx, dst, &mapped_addr, size);
     if (ret != ROCM_SUCCESS)
         return ret;
     
     memcpy(mapped_addr, src, size);
     
     rocm_unmap_memory(ctx, mapped_addr, size);
     
     printf("ROCm: Copied %zu bytes from host to device\n", size);
     
     return ROCM_SUCCESS;
 }
 
 /* Copy device to host */
 int rocm_memcpy_d2h(rocm_context_t ctx, void *dst, rocm_mem_t src, size_t size)
 {
     void *mapped_addr;
     int ret;
     
     ret = rocm_map_memory(ctx, src, &mapped_addr, size);
     if (ret != ROCM_SUCCESS)
         return ret;
     
     memcpy(dst, mapped_addr, size);
     
     rocm_unmap_memory(ctx, mapped_addr, size);
     
     printf("ROCm: Copied %zu bytes from device to host\n", size);
     
     return ROCM_SUCCESS;
 }
 
 /* Create command buffer */
 int rocm_create_cmdbuf(rocm_cmdbuf_t **cmdbuf, uint32_t initial_capacity)
 {
     rocm_cmdbuf_t *buf;
     
     if (!cmdbuf)
         return ROCM_ERROR_INVALID;
     
     buf = (rocm_cmdbuf_t*)malloc(sizeof(rocm_cmdbuf_t));
     if (!buf)
         return ROCM_ERROR_NOMEM;
     
     buf->commands = (uint32_t*)malloc(initial_capacity * sizeof(uint32_t));
     if (!buf->commands) {
         free(buf);
         return ROCM_ERROR_NOMEM;
     }
     
     buf->size = 0;
     buf->capacity = initial_capacity;
     
     *cmdbuf = buf;
     
     return ROCM_SUCCESS;
 }
 
 /* Destroy command buffer */
 int rocm_destroy_cmdbuf(rocm_cmdbuf_t *cmdbuf)
 {
     if (!cmdbuf)
         return ROCM_ERROR_INVALID;
     
     free(cmdbuf->commands);
     free(cmdbuf);
     
     return ROCM_SUCCESS;
 }
 
 /* Add command to buffer */
 int rocm_cmdbuf_add(rocm_cmdbuf_t *cmdbuf, uint32_t cmd)
 {
     if (!cmdbuf)
         return ROCM_ERROR_INVALID;
     
     if (cmdbuf->size >= cmdbuf->capacity) {
         uint32_t new_capacity = cmdbuf->capacity * 2;
         uint32_t *new_commands = (uint32_t*)realloc(cmdbuf->commands,
                                                      new_capacity * sizeof(uint32_t));
         if (!new_commands)
             return ROCM_ERROR_NOMEM;
         
         cmdbuf->commands = new_commands;
         cmdbuf->capacity = new_capacity;
     }
     
     cmdbuf->commands[cmdbuf->size++] = cmd;
     
     return ROCM_SUCCESS;
 }
 
 /* Submit command buffer */
 int rocm_submit(rocm_context_t ctx, rocm_cmdbuf_t *cmdbuf, uint32_t flags)
 {
     struct rocm_cmd_submit req;
     rocm_mem_t cmd_mem;
     int ret;
     
     if (!ctx || !cmdbuf)
         return ROCM_ERROR_INVALID;
     
     /* Allocate GPU memory for command buffer */
     ret = rocm_malloc(ctx, &cmd_mem, cmdbuf->size * sizeof(uint32_t), 0);
     if (ret != ROCM_SUCCESS)
         return ret;
     
     /* Copy commands to GPU */
     ret = rocm_memcpy_h2d(ctx, cmd_mem, cmdbuf->commands,
                           cmdbuf->size * sizeof(uint32_t));
     if (ret != ROCM_SUCCESS) {
         rocm_free(ctx, cmd_mem);
         return ret;
     }
     
     /* Submit to kernel */
     req.cmd_buffer_handle = cmd_mem;
     req.cmd_size = cmdbuf->size * sizeof(uint32_t);
     req.flags = flags;
     
     if (ioctl(ctx->fd, ROCM_IOCTL_SUBMIT_CMD, &req) < 0) {
         perror("Failed to submit command buffer");
         rocm_free(ctx, cmd_mem);
         return ROCM_ERROR_IO;
     }
     
     printf("ROCm: Submitted %u commands to GPU\n", cmdbuf->size);
     
     /* Free command buffer memory */
     rocm_free(ctx, cmd_mem);
     
     return ROCM_SUCCESS;
 }
 
 /* Synchronize with GPU */
 int rocm_sync(rocm_context_t ctx)
 {
     if (!ctx)
         return ROCM_ERROR_INVALID;
     
     /* In a real driver, this would wait for GPU to complete */
     usleep(1000); /* Simulate sync delay */
     
     return ROCM_SUCCESS;
 }
 
 /* Get error string */
 const char* rocm_get_error_string(int error_code)
 {
     switch (error_code) {
     case ROCM_SUCCESS:
         return "Success";
     case ROCM_ERROR_INVALID:
         return "Invalid parameter";
     case ROCM_ERROR_NOMEM:
         return "Out of memory";
     case ROCM_ERROR_IO:
         return "I/O error";
     case ROCM_ERROR_NOT_FOUND:
         return "Not found";
     default:
         return "Unknown error";
     }
 }
 