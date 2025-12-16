#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>

struct dma_mapped_buffer {
    struct scatterlist *sg;
    int nents;
    enum dma_data_direction direction;
};

// Map user buffer for DMA
int dma_map_user_buffer(struct device *dev, 
                        unsigned long user_addr,
                        size_t size,
                        struct dma_mapped_buffer *mapped) {
    
    struct page **pages;
    int i, nents;
    
    // Get pages from user address
    nents = get_user_pages_fast(user_addr, 
                                 size / PAGE_SIZE, 
                                 1,  // write
                                 pages);
    
    if (nents < 0) {
        pr_err("Failed to get user pages\n");
        return nents;
    }
    
    // Allocate scatterlist
    mapped->sg = kmalloc(sizeof(struct scatterlist) * nents, GFP_KERNEL);
    if (!mapped->sg) {
        return -ENOMEM;
    }
    
    // Initialize scatterlist
    sg_init_table(mapped->sg, nents);
    
    for (i = 0; i < nents; i++) {
        sg_set_page(&mapped->sg[i], pages[i], PAGE_SIZE, 0);
    }
    
    // Map for DMA
    mapped->direction = DMA_FROM_DEVICE;
    mapped->nents = dma_map_sg(dev, mapped->sg, nents, mapped->direction);
    
    if (!mapped->nents) {
        pr_err("DMA mapping failed\n");
        kfree(mapped->sg);
        return -ENOMEM;
    }
    
    pr_info("Mapped %d pages for DMA\n", mapped->nents);
    return 0;
}

// Unmap DMA buffer
void dma_unmap_user_buffer(struct device *dev, 
                           struct dma_mapped_buffer *mapped) {
    dma_unmap_sg(dev, mapped->sg, mapped->nents, mapped->direction);
    kfree(mapped->sg);
}
