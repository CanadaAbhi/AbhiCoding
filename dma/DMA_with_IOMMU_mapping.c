#include <linux/iommu.h>
#include <linux/dma-mapping.h>

struct iommu_dma_context {
    struct iommu_domain *domain;
    dma_addr_t iova_start;
    size_t iova_size;
};

// Initialize IOMMU for DMA
int iommu_dma_setup(struct device *dev,
                    struct iommu_dma_context *ctx) {
    
    // Attach device to IOMMU domain
    ctx->domain = iommu_domain_alloc(&platform_bus_type);
    if (!ctx->domain) {
        pr_err("IOMMU domain allocation failed\n");
        return -ENOMEM;
    }
    
    if (iommu_attach_device(ctx->domain, dev)) {
        pr_err("Failed to attach device to IOMMU\n");
        iommu_domain_free(ctx->domain);
        return -EFAULT;
    }
    
    pr_info("Device attached to IOMMU\n");
    return 0;
}

// Map physical buffer via IOMMU
dma_addr_t iommu_map_dma_buffer(struct iommu_dma_context *ctx,
                                phys_addr_t phys_addr,
                                size_t size,
                                int prot) {
    
    dma_addr_t iova;
    
    // Allocate IOVA (I/O Virtual Address)
    iova = ctx->iova_start;
    
    // Map physical address to IOVA
    if (iommu_map(ctx->domain, iova, phys_addr, size, prot)) {
        pr_err("IOMMU mapping failed\n");
        return 0;
    }
    
    pr_info("Physical: 0x%llx -> IOVA: 0x%llx\n", 
            (unsigned long long)phys_addr,
            (unsigned long long)iova);
    
    return iova;
}

void iommu_unmap_dma_buffer(struct iommu_dma_context *ctx,
                            dma_addr_t iova,
                            size_t size) {
    iommu_unmap(ctx->domain, iova, size);
}

void iommu_dma_cleanup(struct device *dev,
                       struct iommu_dma_context *ctx) {
    iommu_detach_device(ctx->domain, dev);
    iommu_domain_free(ctx->domain);
}
