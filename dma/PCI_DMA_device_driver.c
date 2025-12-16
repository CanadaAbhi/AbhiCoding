#include <linux/pci.h>
#include <linux/dma-mapping.h>

struct pci_dma_device {
    struct pci_dev *pdev;
    void __iomem *bar0;
    void *dma_buffer;
    dma_addr_t dma_phys;
    size_t buffer_size;
};

static int pci_dma_probe(struct pci_dev *pdev,
                         const struct pci_device_id *id) {
    struct pci_dma_device *dev_ctx;
    int ret;
    
    pr_info("PCI DMA Device found\n");
    
    // Enable PCI device
    ret = pci_enable_device(pdev);
    if (ret) {
        pr_err("PCI enable failed\n");
        return ret;
    }
    
    // Set DMA mask
    ret = pci_set_dma_mask(pdev, DMA_BIT_MASK(64));
    if (ret) {
        ret = pci_set_dma_mask(pdev, DMA_BIT_MASK(32));
        if (ret) {
            pr_err("DMA mask setup failed\n");
            pci_disable_device(pdev);
            return ret;
        }
    }
    
    // Request BAR0
    ret = pci_request_regions(pdev, "pci-dma-driver");
    if (ret) {
        pr_err("Request regions failed\n");
        pci_disable_device(pdev);
        return ret;
    }
    
    // Map BAR0
    dev_ctx = devm_kzalloc(&pdev->dev, sizeof(*dev_ctx), GFP_KERNEL);
    dev_ctx->pdev = pdev;
    dev_ctx->bar0 = pci_ioremap_bar(pdev, 0);
    
    if (!dev_ctx->bar0) {
        pr_err("BAR0 mapping failed\n");
        pci_release_regions(pdev);
        pci_disable_device(pdev);
        return -ENOMEM;
    }
    
    // Allocate DMA buffer
    dev_ctx->buffer_size = 4096;
    dev_ctx->dma_buffer = dma_alloc_coherent(&pdev->dev,
                                             dev_ctx->buffer_size,
                                             &dev_ctx->dma_phys,
                                             GFP_KERNEL);
    
    if (!dev_ctx->dma_buffer) {
        pr_err("DMA buffer allocation failed\n");
        pci_iounmap(pdev, dev_ctx->bar0);
        pci_release_regions(pdev);
        pci_disable_device(pdev);
        return -ENOMEM;
    }
    
    // Write DMA address to device register
    iowrite32((u32)dev_ctx->dma_phys, dev_ctx->bar0 + 0x10);
    
    pci_set_drvdata(pdev, dev_ctx);
    pr_info("PCI DMA device initialized\n");
    
    return 0;
}

static void pci_dma_remove(struct pci_dev *pdev) {
    struct pci_dma_device *dev_ctx = pci_get_drvdata(pdev);
    
    if (dev_ctx) {
        dma_free_coherent(&pdev->dev, dev_ctx->buffer_size,
                         dev_ctx->dma_buffer, dev_ctx->dma_phys);
        pci_iounmap(pdev, dev_ctx->bar0);
    }
    
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    
    pr_info("PCI DMA device removed\n");
}

static struct pci_device_id pci_dma_ids[] = {
    { PCI_DEVICE(0x8086, 0x1234) },  // Intel vendor, device ID
    { 0 }
};

static struct pci_driver pci_dma_driver = {
    .name = "pci-dma-driver",
    .id_table = pci_dma_ids,
    .probe = pci_dma_probe,
    .remove = pci_dma_remove,
};

module_pci_driver(pci_dma_driver);
MODULE_LICENSE("GPL");
