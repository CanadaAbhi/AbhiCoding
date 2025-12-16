#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>

struct dma_callback_context {
    struct completion done;
    int status;
};

// DMA completion callback
static void dma_completion_callback(void *param) {
    struct dma_callback_context *ctx = param;
    ctx->status = 0;  // Success
    complete(&ctx->done);
    pr_info("DMA transfer completed\n");
}

// Setup DMA with callback
int dma_transfer_with_callback(struct dma_chan *chan,
                               dma_addr_t src,
                               dma_addr_t dst,
                               size_t len) {
    
    struct dma_async_tx_descriptor *tx_desc;
    struct dma_callback_context ctx;
    
    // Initialize completion
    init_completion(&ctx.done);
    ctx.status = -1;
    
    // Prepare DMA descriptor
    tx_desc = dmaengine_prep_dma_memcpy(chan, dst, src, len, 0);
    
    if (!tx_desc) {
        pr_err("Failed to prepare DMA descriptor\n");
        return -EBUSY;
    }
    
    // Set callback
    tx_desc->callback = dma_completion_callback;
    tx_desc->callback_param = &ctx;
    
    // Submit transfer
    dmaengine_submit(tx_desc);
    dma_async_issue_pending(chan);
    
    // Wait for completion
    wait_for_completion(&ctx.done);
    
    return ctx.status;
}
