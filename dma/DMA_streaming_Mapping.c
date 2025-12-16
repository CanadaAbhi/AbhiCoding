#include <linux/dma-mapping.h>

struct streaming_dma_buf {
    void *cpu_addr;
    dma_addr_t dma_addr;
    size_t size;
};

// Allocate streaming DMA buffer
void streaming_dma_alloc(struct device *dev, 
                         struct streaming_dma_buf *buf,
                         size_t size) {
    
    buf->size = size;
    
    // Allocate regular memory
    buf->cpu_addr = kmalloc(size, GFP_KERNEL);
    if (!buf->cpu_addr) {
        return;
    }
    
    // Map for DMA (non-coherent)
    buf->dma_addr = dma_map_single(dev, buf->cpu_addr, size, DMA_TO_DEVICE);
    
    if (dma_mapping_error(dev, buf->dma_addr)) {
        pr_err("DMA mapping error\n");
        kfree(buf->cpu_addr);
        return;
    }
    
    pr_info("Streaming DMA mapped at 0x%llx\n", (unsigned long long)buf->dma_addr);
}

// Cache synchronization before device access
void dma_sync_before_device(struct device *dev, 
                            struct streaming_dma_buf *buf) {
    // Flush CPU cache to memory
    dma_sync_single_for_device(dev, buf->dma_addr, 
                               buf->size, DMA_TO_DEVICE);
}

// Cache synchronization after device access
void dma_sync_after_device(struct device *dev, 
                           struct streaming_dma_buf *buf) {
    // Invalidate CPU cache, read from memory
    dma_sync_single_for_cpu(dev, buf->dma_addr, 
                            buf->size, DMA_FROM_DEVICE);
}

void streaming_dma_free(struct device *dev, 
                        struct streaming_dma_buf *buf) {
    dma_unmap_single(dev, buf->dma_addr, buf->size, DMA_TO_DEVICE);
    kfree(buf->cpu_addr);
}
