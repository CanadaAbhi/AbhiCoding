#ifndef KMEM_LAB_UAPI_H
#define KMEM_LAB_UAPI_H
#include <linux/types.h>
#include <linux/ioctl.h>

enum kmem_type {
    KMEM_TYPE_KMALLOC = 0,   /* physically contiguous, slab-backed */
    KMEM_TYPE_PAGES   = 1,   /* physically contiguous, alloc_pages(order) */
    KMEM_TYPE_VMALLOC = 2,   /* virtually contiguous, physically scattered */
    KMEM_TYPE_DMA     = 3,   /* dma_alloc_coherent, DMA-capable */
};

#define KMEM_FLAG_UNCACHED   (1u << 0)  /* PAGES/KMALLOC only: map pgprot_noncached */

struct kmem_alloc_req {
    __u32 type;
    __u32 flags;
    __u64 size;       /* requested size */
    __u64 handle;     /* out: opaque handle, also used as mmap offset<<PAGE_SHIFT */
    __u64 actual_size;/* out: actual backing size (reveals internal fragmentation) */
};

struct kmem_free_req {
    __u64 handle;
};

struct kmem_info_req {
    __u64 handle;
    __u32 type;
    __u64 size;
    __u64 actual_size;
    __u64 phys_addr;   /* 0 if not physically-contiguous (VMALLOC) */
    __u64 dma_addr;    /* only valid for DMA type */
};

#define KMEM_IOC_MAGIC 0xE5
#define KMEM_IOC_ALLOC   _IOWR(KMEM_IOC_MAGIC, 1, struct kmem_alloc_req)
#define KMEM_IOC_FREE    _IOW(KMEM_IOC_MAGIC,  2, struct kmem_free_req)
#define KMEM_IOC_GETINFO _IOWR(KMEM_IOC_MAGIC, 3, struct kmem_info_req)

#endif