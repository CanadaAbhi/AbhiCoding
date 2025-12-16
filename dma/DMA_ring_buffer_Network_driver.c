#include <linux/dma-mapping.h>
#include <linux/skbuff.h>

struct dma_ring_descriptor {
    u32 addr_low;
    u32 addr_high;
    u16 length;
    u16 csum_offset;
};

struct dma_ring_buffer {
    struct dma_ring_descriptor *desc_ring;
    dma_addr_t desc_ring_phys;
    void **data_buffers;
    dma_addr_t *data_phys;
    u32 head;
    u32 tail;
    u32 size;
};

// Initialize ring buffer
int init_dma_ring(struct device *dev,
                  struct dma_ring_buffer *ring,
                  u32 ring_size) {
    
    int i;
    size_t desc_size;
    
    ring->size = ring_size;
    
    // Allocate descriptor ring
    desc_size = ring_size * sizeof(struct dma_ring_descriptor);
    ring->desc_ring = dma_alloc_coherent(dev, desc_size,
                                         &ring->desc_ring_phys,
                                         GFP_KERNEL);
    if (!ring->desc_ring)
        return -ENOMEM;
    
    // Allocate data buffer pointers
    ring->data_buffers = kmalloc(sizeof(void *) * ring_size, GFP_KERNEL);
    ring->data_phys = kmalloc(sizeof(dma_addr_t) * ring_size, GFP_KERNEL);
    
    if (!ring->data_buffers || !ring->data_phys)
        return -ENOMEM;
    
    // Allocate data buffers
    for (i = 0; i < ring_size; i++) {
        ring->data_buffers[i] = dma_alloc_coherent(dev, 2048,
                                                   &ring->data_phys[i],
                                                   GFP_KERNEL);
        
        if (!ring->data_buffers[i]) {
            pr_err("Data buffer allocation failed\n");
            return -ENOMEM;
        }
        
        // Initialize descriptor
        ring->desc_ring[i].addr_low = (u32)ring->data_phys[i];
        ring->desc_ring[i].addr_high = (u32)(ring->data_phys[i] >> 32);
        ring->desc_ring[i].length = 2048;
        ring->desc_ring[i].csum_offset = 0;
    }
    
    ring->head = 0;
    ring->tail = 0;
    
    pr_info("DMA ring initialized: %d descriptors, phys: 0x%llx\n",
            ring_size, (unsigned long long)ring->desc_ring_phys);
    
    return 0;
}

// Add packet to ring
int add_packet_to_ring(struct dma_ring_buffer *ring,
                       struct sk_buff *skb) {
    
    if ((ring->head + 1) % ring->size == ring->tail) {
        pr_warn("Ring buffer full\n");
        return -ENOBUFS;
    }
    
    // Copy skb data to DMA buffer
    memcpy(ring->data_buffers[ring->head], skb->data, skb->len);
    
    // Update descriptor
    ring->desc_ring[ring->head].length = skb->len;
    ring->desc_ring[ring->head].csum_offset = skb->csum_offset;
    
    // Advance head pointer
    ring->head = (ring->head + 1) % ring->size;
    
    return 0;
}

void cleanup_dma_ring(struct device *dev,
                      struct dma_ring_buffer *ring) {
    
    int i;
    size_t desc_size = ring->size * sizeof(struct dma_ring_descriptor);
    
    // Free data buffers
    for (i = 0; i < ring->size; i++) {
        if (ring->data_buffers[i]) {
            dma_free_coherent(dev, 2048, ring->data_buffers[i],
                             ring->data_phys[i]);
        }
    }
    
    // Free descriptor ring
    if (ring->desc_ring) {
        dma_free_coherent(dev, desc_size, ring->desc_ring,
                         ring->desc_ring_phys);
    }
    
    kfree(ring->data_buffers);
    kfree(ring->data_phys);
}
