#include <linux/bio.h>
#include <linux/blkdev.h>

struct sg_dma_transfer {
    struct scatterlist *sg_list;
    int sg_count;
    dma_addr_t *dma_addrs;
};

// Prepare DMA transfer from bio
int prepare_dma_from_bio(struct device *dev, 
                         struct bio *bio,
                         struct sg_dma_transfer *transfer) {
    
    struct bio_vec *bvec;
    int i = 0, nsegs;
    
    // Count segments
    nsegs = bio->bi_vcnt;
    
    // Allocate scatterlist
    transfer->sg_list = kmalloc(sizeof(struct scatterlist) * nsegs, 
                                GFP_KERNEL);
    transfer->dma_addrs = kmalloc(sizeof(dma_addr_t) * nsegs, 
                                  GFP_KERNEL);
    
    sg_init_table(transfer->sg_list, nsegs);
    
    // Build scatterlist from bio vectors
    bio_for_each_segment(bvec, bio, i) {
        sg_set_page(&transfer->sg_list[i], 
                    bvec->bv_page,
                    bvec->bv_len,
                    bvec->bv_offset);
    }
    
    // Map scatterlist
    transfer->sg_count = dma_map_sg(dev, transfer->sg_list, 
                                    nsegs, DMA_FROM_DEVICE);
    
    // Store DMA addresses
    for (i = 0; i < transfer->sg_count; i++) {
        transfer->dma_addrs[i] = sg_dma_address(&transfer->sg_list[i]);
    }
    
    return transfer->sg_count;
}

void cleanup_dma_transfer(struct device *dev,
                          struct sg_dma_transfer *transfer) {
    dma_unmap_sg(dev, transfer->sg_list, transfer->sg_count, DMA_FROM_DEVICE);
    kfree(transfer->sg_list);
    kfree(transfer->dma_addrs);
}
