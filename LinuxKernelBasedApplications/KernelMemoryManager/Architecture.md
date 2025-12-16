Application (bench_alloc.c)
       |
       v
Custom Memory API (libkmem.c/.h)  -- kmem_alloc()/kmem_free()/kmem_getinfo()
       |
       v  ioctl(ALLOC/FREE/GETINFO) + mmap(fd, offset=handle<<PAGE_SHIFT)
Kernel Driver (kmem_lab_drv.c)
       |
       +--- KMALLOC   -> kmalloc() + remap_pfn_range (physically contiguous)
       +--- PAGES     -> alloc_pages(order) + remap_pfn_range (cached/uncached toggle)
       +--- VMALLOC   -> vmalloc_user() + remap_vmalloc_range (non-contiguous phys)
       +--- DMA       -> dma_alloc_coherent() + dma_mmap_coherent() (fake platform_device)


#define VIRTIO_ID_TOY  63   /* pick an unused experimental id (>= 0x28 reserved range) */
