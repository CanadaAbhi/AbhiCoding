#include <linux/dmapool.h>

struct dma_pool_context {
    struct dma_pool *pool;
    void **pool_buffers;
    dma_addr_t *pool_handles;
    int buffer_count;
};

// Create DMA pool
int create_dma_pool(struct device *dev,
                    struct dma_pool_context *ctx,
                    size_t size,
                    int count) {
    
    int i;
    
    // Create pool (size: element size, align: alignment)
    ctx->pool = dma_pool_create("dma-pool", dev, size, 16, 0);
    if (!ctx->pool) {
        pr_err("DMA pool creation failed\n");
        return -ENOMEM;
    }
    
    // Allocate pool buffers
    ctx->pool_buffers = kmalloc(sizeof(void *) * count, GFP_KERNEL);
    ctx->pool_handles = kmalloc(sizeof(dma_addr_t) * count, GFP_KERNEL);
    
    if (!ctx->pool_buffers || !ctx->pool_handles) {
        return -ENOMEM;
    }
    
    // Preallocate elements
    for (i = 0; i < count; i++) {
        ctx->pool_buffers[i] = dma_pool_alloc(ctx->pool, 
                                              GFP_KERNEL,
                                              &ctx->pool_handles[i]);
        if (!ctx->pool_buffers[i]) {
            pr_err("DMA pool alloc failed at %d\n", i);
            return -ENOMEM;
        }
    }
    
    ctx->buffer_count = count;
    pr_info("Created DMA pool with %d buffers\n", count);
    
    return 0;
}

// Get buffer from pool
void* get_dma_pool_buffer(struct dma_pool_context *ctx, int idx,
                          dma_addr_t *dma_addr) {
    
    if (idx >= ctx->buffer_count)
        return NULL;
    
    *dma_addr = ctx->pool_handles[idx];
    return ctx->pool_buffers[idx];
}

void destroy_dma_pool(struct dma_pool_context *ctx) {
    int i;
    
    // Free allocated elements
    for (i = 0; i < ctx->buffer_count; i++) {
        if (ctx->pool_buffers[i]) {
            dma_pool_free(ctx->pool, ctx->pool_buffers[i],
                         ctx->pool_handles[i]);
        }
    }
    
    // Destroy pool
    dma_pool_destroy(ctx->pool);
    
    kfree(ctx->pool_buffers);
    kfree(ctx->pool_handles);
}
