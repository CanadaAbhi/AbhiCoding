#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/dma-mapping.h>

struct dma_device_data {
    struct platform_device *pdev;
    void __iomem *regs;
    void *dma_buf;
    dma_addr_t dma_handle;
};

static int dma_device_probe(struct platform_device *pdev) {
    struct dma_device_data *dma_dev;
    struct resource *res;
    int ret;
    
    pr_info("DMA Device Probe\n");
    
    // Allocate device data
    dma_dev = devm_kzalloc(&pdev->dev, sizeof(*dma_dev), GFP_KERNEL);
    if (!dma_dev)
        return -ENOMEM;
    
    dma_dev->pdev = pdev;
    
    // Get platform resources
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    dma_dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dma_dev->regs))
        return PTR_ERR(dma_dev->regs);
    
    // Set DMA mask
    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
    if (ret) {
        pr_err("DMA mask setup failed\n");
        return ret;
    }
    
    // Allocate DMA buffer
    dma_dev->dma_buf = dma_alloc_coherent(&pdev->dev, 
                                          PAGE_SIZE,
                                          &dma_dev->dma_handle,
                                          GFP_KERNEL);
    
    if (!dma_dev->dma_buf) {
        pr_err("DMA buffer allocation failed\n");
        return -ENOMEM;
    }
    
    platform_set_drvdata(pdev, dma_dev);
    pr_info("DMA device registered successfully\n");
    
    return 0;
}

static int dma_device_remove(struct platform_device *pdev) {
    struct dma_device_data *dma_dev = platform_get_drvdata(pdev);
    
    if (dma_dev->dma_buf) {
        dma_free_coherent(&pdev->dev, PAGE_SIZE,
                         dma_dev->dma_buf, dma_dev->dma_handle);
    }
    
    pr_info("DMA device removed\n");
    return 0;
}

static struct platform_driver dma_driver = {
    .probe = dma_device_probe,
    .remove = dma_device_remove,
    .driver = {
        .name = "simple-dma",
        .owner = THIS_MODULE,
    },
};

module_platform_driver(dma_driver);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("DMA Developer");
MODULE_DESCRIPTION("Simple DMA Device Driver");
