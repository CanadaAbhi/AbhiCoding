/**
 * @file pcie_dma_driver.c
 * @brief PCIe DMA Driver with Scatter-Gather support
 * 
 * Features:
 * - Coherent DMA allocation
 * - Streaming DMA mapping
 * - Scatter-gather DMA
 * - DMA transfer initiation
 * - Ring buffer implementation
 */

 #include "../common/pcie_common.h"
 #include <linux/scatterlist.h>
 
 #define DRIVER_NAME "pcie_dma"
 #define DRIVER_VERSION "1.0"
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("Abhi");
 MODULE_DESCRIPTION("PCIe DMA Driver with Scatter-Gather");
 MODULE_VERSION(DRIVER_VERSION);
 
 #define DMA_RING_SIZE 64
 
 struct dma_ring {
     struct dma_descriptor *desc;  /* Descriptor ring */
     dma_addr_t desc_dma;          /* DMA handle for descriptors */
     
     void **buffers;               /* Kernel buffers */
     dma_addr_t *buffer_dma;       /* DMA handles for buffers */
     
     u32 head;                     /* Producer index */
     u32 tail;                     /* Consumer index */
     u32 count;                    /* Number of entries */
 };
 
 struct dma_stats {
     u64 tx_count;
     u64 rx_count;
     u64 tx_bytes;
     u64 rx_bytes;
     u64 dma_errors;
 };
 
 struct dma_private {
     struct pci_dev *pdev;
     void __iomem *bar0;
     
     /* DMA rings */
     struct dma_ring tx_ring;
     struct dma_ring rx_ring;
     
     /* Coherent DMA buffer */
     void *coherent_buf;
     dma_addr_t coherent_dma;
     
     /* Scatter-gather */
     struct scatterlist *sg_list;
     int sg_count;
     
     /* Statistics */
     struct dma_stats stats;
     spinlock_t lock;
     
     /* Character device */
     struct cdev cdev;
     dev_t devt;
     struct class *class;
     struct device *device;
 };
 
 static struct dma_private *global_priv = NULL;
 
 /* Initialize DMA ring */
 static int init_dma_ring(struct pci_dev *pdev, struct dma_ring *ring, u32 size)
 {
     int i;
     
     ring->count = size;
     ring->head = 0;
     ring->tail = 0;
     
     /* Allocate descriptor ring (coherent DMA) */
     ring->desc = dma_alloc_coherent(&pdev->dev,
                                     size * sizeof(struct dma_descriptor),
                                     &ring->desc_dma,
                                     GFP_KERNEL);
     if (!ring->desc) {
         pcie_err("Failed to allocate descriptor ring");
         return -ENOMEM;
     }
     
     memset(ring->desc, 0, size * sizeof(struct dma_descriptor));
     
     /* Allocate buffer pointers */
     ring->buffers = kcalloc(size, sizeof(void *), GFP_KERNEL);
     if (!ring->buffers) {
         dma_free_coherent(&pdev->dev,
                          size * sizeof(struct dma_descriptor),
                          ring->desc, ring->desc_dma);
         return -ENOMEM;
     }
     
     ring->buffer_dma = kcalloc(size, sizeof(dma_addr_t), GFP_KERNEL);
     if (!ring->buffer_dma) {
         kfree(ring->buffers);
         dma_free_coherent(&pdev->dev,
                          size * sizeof(struct dma_descriptor),
                          ring->desc, ring->desc_dma);
         return -ENOMEM;
     }
     
     /* Allocate and map buffers */
     for (i = 0; i < size; i++) {
         ring->buffers[i] = kmalloc(DMA_BUF_SIZE, GFP_KERNEL);
         if (!ring->buffers[i]) {
             /* Free already allocated buffers */
             while (--i >= 0) {
                 dma_unmap_single(&pdev->dev, ring->buffer_dma[i],
                                 DMA_BUF_SIZE, DMA_FROM_DEVICE);
                 kfree(ring->buffers[i]);
             }
             kfree(ring->buffer_dma);
             kfree(ring->buffers);
             dma_free_coherent(&pdev->dev,
                              size * sizeof(struct dma_descriptor),
                              ring->desc, ring->desc_dma);
             return -ENOMEM;
         }
         
         /* Map buffer for DMA */
         ring->buffer_dma[i] = dma_map_single(&pdev->dev,
                                              ring->buffers[i],
                                              DMA_BUF_SIZE,
                                              DMA_FROM_DEVICE);
         
         if (dma_mapping_error(&pdev->dev, ring->buffer_dma[i])) {
             kfree(ring->buffers[i]);
             /* Free already mapped buffers */
             while (--i >= 0) {
                 dma_unmap_single(&pdev->dev, ring->buffer_dma[i],
                                 DMA_BUF_SIZE, DMA_FROM_DEVICE);
                 kfree(ring->buffers[i]);
             }
             kfree(ring->buffer_dma);
             kfree(ring->buffers);
             dma_free_coherent(&pdev->dev,
                              size * sizeof(struct dma_descriptor),
                              ring->desc, ring->desc_dma);
             return -ENOMEM;
         }
         
         /* Setup descriptor */
         ring->desc[i].buffer_addr = ring->buffer_dma[i];
         ring->desc[i].length = DMA_BUF_SIZE;
         ring->desc[i].flags = 0;
     }
     
     pcie_info("DMA ring initialized: %d descriptors", size);
     return 0;
 }
 
 /* Free DMA ring */
 static void free_dma_ring(struct pci_dev *pdev, struct dma_ring *ring)
 {
     int i;
     
     if (!ring->desc)
         return;
     
     /* Unmap and free buffers */
     for (i = 0; i < ring->count; i++) {
         if (ring->buffers[i]) {
             dma_unmap_single(&pdev->dev, ring->buffer_dma[i],
                             DMA_BUF_SIZE, DMA_FROM_DEVICE);
             kfree(ring->buffers[i]);
         }
     }
     
     kfree(ring->buffer_dma);
     kfree(ring->buffers);
     
     /* Free descriptor ring */
     dma_free_coherent(&pdev->dev,
                      ring->count * sizeof(struct dma_descriptor),
                      ring->desc, ring->desc_dma);
     
     memset(ring, 0, sizeof(*ring));
 }
 
 /* Start DMA transfer */
 static int start_dma_transfer(struct dma_private *priv,
                               dma_addr_t src, dma_addr_t dst, u32 size)
 {
     /* This is device-specific - write to device registers to start DMA */
     
     /* Example: Write DMA source address */
     iowrite32(lower_32_bits(src), priv->bar0 + 0x10);
     iowrite32(upper_32_bits(src), priv->bar0 + 0x14);
     
     /* Write DMA destination address */
     iowrite32(lower_32_bits(dst), priv->bar0 + 0x18);
     iowrite32(upper_32_bits(dst), priv->bar0 + 0x1C);
     
     /* Write size */
     iowrite32(size, priv->bar0 + 0x20);
     
     /* Start DMA (device-specific control register) */
     iowrite32(0x1, priv->bar0 + 0x24);
     
     wmb(); /* Ensure writes complete */
     
     pcie_dbg("DMA transfer started: src=0x%llx dst=0x%llx size=%u",
              (u64)src, (u64)dst, size);
     
     return 0;
 }
 
 /* Setup scatter-gather DMA */
 static int setup_scatter_gather(struct dma_private *priv,
                                 void *buf, size_t size)
 {
     int i, nents;
     struct scatterlist *sg;
     size_t chunk_size = PAGE_SIZE;
     int num_chunks = (size + chunk_size - 1) / chunk_size;
     
     /* Allocate scatter list */
     priv->sg_list = kcalloc(num_chunks, sizeof(struct scatterlist), GFP_KERNEL);
     if (!priv->sg_list)
         return -ENOMEM;
     
     sg_init_table(priv->sg_list, num_chunks);
     
     /* Setup scatter-gather entries */
     for_each_sg(priv->sg_list, sg, num_chunks, i) {
         size_t len = min(chunk_size, size - (i * chunk_size));
         sg_set_buf(sg, buf + (i * chunk_size), len);
     }
     
     /* Map scatter-gather list for DMA */
     nents = dma_map_sg(&priv->pdev->dev, priv->sg_list, num_chunks,
                        DMA_TO_DEVICE);
     if (nents == 0) {
         kfree(priv->sg_list);
         priv->sg_list = NULL;
         return -ENOMEM;
     }
     
     priv->sg_count = nents;
     
     pcie_info("Scatter-gather DMA setup: %d entries", nents);
     
     return 0;
 }
 
 /* Character device operations */
 static long dma_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
 {
     struct dma_private *priv = filp->private_data;
     struct pcie_dma_test dma_test;
     int ret = 0;
     
     switch (cmd) {
     case PCIE_IOC_DMA_TEST:
         if (copy_from_user(&dma_test, (void __user *)arg, sizeof(dma_test)))
             return -EFAULT;
         
         pcie_info("DMA test: src=0x%llx dst=0x%llx size=%u",
                  dma_test.src_addr, dma_test.dst_addr, dma_test.size);
         
         ret = start_dma_transfer(priv,
                                 (dma_addr_t)dma_test.src_addr,
                                 (dma_addr_t)dma_test.dst_addr,
                                 dma_test.size);
         break;
         
     default:
         ret = -EINVAL;
     }
     
     return ret;
 }
 
 static ssize_t dma_read_stats(struct file *filp, char __user *buf,
                               size_t count, loff_t *f_pos)
 {
     struct dma_private *priv = filp->private_data;
     char stats_buf[512];
     int len;
     
     if (*f_pos > 0)
         return 0;
     
     spin_lock(&priv->lock);
     len = snprintf(stats_buf, sizeof(stats_buf),
                   "DMA Statistics:\n"
                   "  TX Count:     %llu\n"
                   "  RX Count:     %llu\n"
                   "  TX Bytes:     %llu\n"
                   "  RX Bytes:     %llu\n"
                   "  DMA Errors:   %llu\n"
                   "  TX Ring:      %u/%u\n"
                   "  RX Ring:      %u/%u\n",
                   priv->stats.tx_count,
                   priv->stats.rx_count,
                   priv->stats.tx_bytes,
                   priv->stats.rx_bytes,
                   priv->stats.dma_errors,
                   priv->tx_ring.head, priv->tx_ring.count,
                   priv->rx_ring.head, priv->rx_ring.count);
     spin_unlock(&priv->lock);
     
     if (count > len)
         count = len;
     
     if (copy_to_user(buf, stats_buf, count))
         return -EFAULT;
     
     *f_pos += count;
     return count;
 }
 
 static int dma_open(struct inode *inode, struct file *filp)
 {
     filp->private_data = global_priv;
     return 0;
 }
 
 static const struct file_operations dma_fops = {
     .owner          = THIS_MODULE,
     .open           = dma_open,
     .read           = dma_read_stats,
     .unlocked_ioctl = dma_ioctl,
     .llseek         = default_llseek,
 };
 
 static int pcie_dma_probe(struct pci_dev *pdev,
                           const struct pci_device_id *id)
 {
     struct dma_private *priv;
     int ret;
     
     pcie_info("Probing device");
     
     priv = kzalloc(sizeof(*priv), GFP_KERNEL);
     if (!priv)
         return -ENOMEM;
     
     priv->pdev = pdev;
     pci_set_drvdata(pdev, priv);
     spin_lock_init(&priv->lock);
     
     /* Enable device */
     ret = pci_enable_device(pdev);
     if (ret) {
         pcie_err("Failed to enable device");
         goto err_free_priv;
     }
     
     /* Set DMA mask */
     ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
     if (ret) {
         pcie_info("64-bit DMA not available, trying 32-bit");
         ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
         if (ret) {
             pcie_err("Failed to set DMA mask");
             goto err_disable_device;
         }
     }
     
     /* Enable bus mastering (required for DMA) */
     pci_set_master(pdev);
     
     ret = pci_request_regions(pdev, DRIVER_NAME);
     if (ret) {
         pcie_err("Failed to request regions");
         goto err_clear_master;
     }
     
     /* Map BAR0 */
     priv->bar0 = pci_iomap(pdev, BAR_0, pci_resource_len(pdev, BAR_0));
     if (!priv->bar0) {
         pcie_err("Failed to map BAR0");
         ret = -ENOMEM;
         goto err_release_regions;
     }
     
     /* Allocate coherent DMA buffer */
     priv->coherent_buf = dma_alloc_coherent(&pdev->dev,
                                             DMA_BUF_SIZE,
                                             &priv->coherent_dma,
                                             GFP_KERNEL);
     if (!priv->coherent_buf) {
         pcie_err("Failed to allocate coherent DMA buffer");
         ret = -ENOMEM;
         goto err_unmap_bar;
     }
     
     pcie_info("Coherent DMA buffer: virt=%p dma=0x%llx",
              priv->coherent_buf, (u64)priv->coherent_dma);
     
     /* Initialize TX/RX rings */
     ret = init_dma_ring(pdev, &priv->tx_ring, DMA_RING_SIZE);
     if (ret)
         goto err_free_coherent;
     
     ret = init_dma_ring(pdev, &priv->rx_ring, DMA_RING_SIZE);
     if (ret)
         goto err_free_tx_ring;
     
     /* Create character device */
     ret = alloc_chrdev_region(&priv->devt, 0, 1, DRIVER_NAME);
     if (ret)
         goto err_free_rx_ring;
     
     cdev_init(&priv->cdev, &dma_fops);
     priv->cdev.owner = THIS_MODULE;
     
     ret = cdev_add(&priv->cdev, priv->devt, 1);
     if (ret)
         goto err_unregister_chrdev;
     
     priv->class = class_create(THIS_MODULE, DRIVER_NAME);
     if (IS_ERR(priv->class)) {
         ret = PTR_ERR(priv->class);
         goto err_del_cdev;
     }
     
     priv->device = device_create(priv->class, &pdev->dev, priv->devt,
                                  NULL, DRIVER_NAME);
     if (IS_ERR(priv->device)) {
         ret = PTR_ERR(priv->device);
         goto err_destroy_class;
     }
     
     global_priv = priv;
     
     pcie_info("Driver loaded successfully");
     pcie_info("TX Ring: %d descriptors at 0x%llx",
              priv->tx_ring.count, (u64)priv->tx_ring.desc_dma);
     pcie_info("RX Ring: %d descriptors at 0x%llx",
              priv->rx_ring.count, (u64)priv->rx_ring.desc_dma);
     
     return 0;
     
 err_destroy_class:
     class_destroy(priv->class);
 err_del_cdev:
     cdev_del(&priv->cdev);
 err_unregister_chrdev:
     unregister_chrdev_region(priv->devt, 1);
 err_free_rx_ring:
     free_dma_ring(pdev, &priv->rx_ring);
 err_free_tx_ring:
     free_dma_ring(pdev, &priv->tx_ring);
 err_free_coherent:
     dma_free_coherent(&pdev->dev, DMA_BUF_SIZE,
                      priv->coherent_buf, priv->coherent_dma);
 err_unmap_bar:
     pci_iounmap(pdev, priv->bar0);
 err_release_regions:
     pci_release_regions(pdev);
 err_clear_master:
     pci_clear_master(pdev);
 err_disable_device:
     pci_disable_device(pdev);
 err_free_priv:
     kfree(priv);
     return ret;
 }
 
 static void pcie_dma_remove(struct pci_dev *pdev)
 {
     struct dma_private *priv = pci_get_drvdata(pdev);
     
     pcie_info("Removing device");
     
     device_destroy(priv->class, priv->devt);
     class_destroy(priv->class);
     cdev_del(&priv->cdev);
     unregister_chrdev_region(priv->devt, 1);
     
     /* Free scatter-gather if allocated */
     if (priv->sg_list) {
         dma_unmap_sg(&pdev->dev, priv->sg_list, priv->sg_count,
                     DMA_TO_DEVICE);
         kfree(priv->sg_list);
     }
     
     free_dma_ring(pdev, &priv->rx_ring);
     free_dma_ring(pdev, &priv->tx_ring);
     
     dma_free_coherent(&pdev->dev, DMA_BUF_SIZE,
                      priv->coherent_buf, priv->coherent_dma);
     
     pci_iounmap(pdev, priv->bar0);
     pci_release_regions(pdev);
     pci_clear_master(pdev);
     pci_disable_device(pdev);
     
     global_priv = NULL;
     kfree(priv);
     
     pcie_info("Device removed");
 }
 
 static const struct pci_device_id pcie_dma_id_table[] = {
     { PCI_DEVICE(DEMO_VENDOR_ID, DEMO_DEVICE_ID) },
     { 0, }
 };
 MODULE_DEVICE_TABLE(pci, pcie_dma_id_table);
 
 static struct pci_driver pcie_dma_driver = {
     .name       = DRIVER_NAME,
     .id_table   = pcie_dma_id_table,
     .probe      = pcie_dma_probe,
     .remove     = pcie_dma_remove,
 };
 
 module_pci_driver(pcie_dma_driver);