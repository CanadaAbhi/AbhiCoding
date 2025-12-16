/**
 * @file pcie_interrupt_driver.c
 * @brief PCIe Interrupt Driver - INTx, MSI, MSI-X
 * 
 * Features:
 * - Legacy INTx interrupts
 * - MSI (Message Signaled Interrupts)
 * - MSI-X (Extended MSI)
 * - Interrupt handlers with tasklet bottom half
 */

 #include "../common/pcie_common.h"
 #include <linux/workqueue.h>
 
 #define DRIVER_NAME "pcie_interrupt"
 #define DRIVER_VERSION "1.0"
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("Abhi");
 MODULE_DESCRIPTION("PCIe Interrupt Driver with MSI/MSI-X support");
 MODULE_VERSION(DRIVER_VERSION);
 
 enum irq_mode {
     IRQ_MODE_INTX = 0,
     IRQ_MODE_MSI,
     IRQ_MODE_MSIX
 };
 
 struct irq_stats {
     u64 irq_count;
     u64 tasklet_count;
     u64 workqueue_count;
     u64 spurious;
 };
 
 struct interrupt_private {
     struct pci_dev *pdev;
     void __iomem *bar0;
     
     /* Interrupt mode */
     enum irq_mode irq_mode;
     int num_vectors;
     struct msix_entry *msix_entries;
     
     /* Bottom half handlers */
     struct tasklet_struct tasklet;
     struct work_struct work;
     struct workqueue_struct *wq;
     
     /* Statistics */
     struct irq_stats stats;
     spinlock_t stats_lock;
     
     /* Character device */
     struct cdev cdev;
     dev_t devt;
     struct class *class;
     struct device *device;
 };
 
 static struct interrupt_private *global_priv = NULL;
 
 /* Tasklet bottom half */
 static void interrupt_tasklet(unsigned long data)
 {
     struct interrupt_private *priv = (struct interrupt_private *)data;
     
     spin_lock(&priv->stats_lock);
     priv->stats.tasklet_count++;
     spin_unlock(&priv->stats_lock);
     
     pcie_dbg("Tasklet executed (count: %llu)", priv->stats.tasklet_count);
     
     /* Process interrupt in tasklet context */
     /* This is where you'd handle quick interrupt work */
 }
 
 /* Work queue bottom half */
 static void interrupt_work(struct work_struct *work)
 {
     struct interrupt_private *priv = container_of(work,
                                                    struct interrupt_private,
                                                    work);
     
     spin_lock(&priv->stats_lock);
     priv->stats.workqueue_count++;
     spin_unlock(&priv->stats_lock);
     
     pcie_dbg("Work executed (count: %llu)", priv->stats.workqueue_count);
     
     /* Process interrupt in process context */
     /* This is where you'd handle longer interrupt work */
 }
 
 /* Legacy INTx interrupt handler */
 static irqreturn_t intx_interrupt_handler(int irq, void *dev_id)
 {
     struct interrupt_private *priv = dev_id;
     u32 status;
     
     /* Read interrupt status register (device-specific) */
     status = ioread32(priv->bar0 + 0x0); /* Example offset */
     
     if (!(status & 0x1)) {
         /* Not our interrupt */
         spin_lock(&priv->stats_lock);
         priv->stats.spurious++;
         spin_unlock(&priv->stats_lock);
         return IRQ_NONE;
     }
     
     spin_lock(&priv->stats_lock);
     priv->stats.irq_count++;
     spin_unlock(&priv->stats_lock);
     
     /* Clear interrupt (device-specific) */
     iowrite32(0x1, priv->bar0 + 0x0);
     
     /* Schedule tasklet */
     tasklet_schedule(&priv->tasklet);
     
     /* Or schedule work */
     queue_work(priv->wq, &priv->work);
     
     return IRQ_HANDLED;
 }
 
 /* MSI interrupt handler */
 static irqreturn_t msi_interrupt_handler(int irq, void *dev_id)
 {
     struct interrupt_private *priv = dev_id;
     
     spin_lock(&priv->stats_lock);
     priv->stats.irq_count++;
     spin_unlock(&priv->stats_lock);
     
     pcie_dbg("MSI interrupt received (count: %llu)", priv->stats.irq_count);
     
     /* MSI interrupts are edge-triggered, no need to clear in device */
     
     tasklet_schedule(&priv->tasklet);
     
     return IRQ_HANDLED;
 }
 
 /* MSI-X interrupt handler */
 static irqreturn_t msix_interrupt_handler(int irq, void *dev_id)
 {
     struct interrupt_private *priv = dev_id;
     int vector = -1;
     int i;
     
     /* Find which vector triggered */
     for (i = 0; i < priv->num_vectors; i++) {
         if (priv->msix_entries[i].vector == irq) {
             vector = i;
             break;
         }
     }
     
     spin_lock(&priv->stats_lock);
     priv->stats.irq_count++;
     spin_unlock(&priv->stats_lock);
     
     pcie_dbg("MSI-X interrupt received on vector %d (count: %llu)",
              vector, priv->stats.irq_count);
     
     tasklet_schedule(&priv->tasklet);
     
     return IRQ_HANDLED;
 }
 
 /* Setup MSI-X */
 static int setup_msix(struct interrupt_private *priv)
 {
     int ret, i;
     
     /* Allocate MSI-X entries */
     priv->num_vectors = MAX_MSIX_VECTORS;
     priv->msix_entries = kcalloc(priv->num_vectors,
                                  sizeof(struct msix_entry),
                                  GFP_KERNEL);
     if (!priv->msix_entries)
         return -ENOMEM;
     
     for (i = 0; i < priv->num_vectors; i++)
         priv->msix_entries[i].entry = i;
     
     /* Enable MSI-X */
     ret = pci_enable_msix_range(priv->pdev, priv->msix_entries,
                                 1, priv->num_vectors);
     if (ret < 0) {
         pcie_err("Failed to enable MSI-X: %d", ret);
         kfree(priv->msix_entries);
         return ret;
     }
     
     priv->num_vectors = ret;
     pcie_info("Enabled %d MSI-X vectors", priv->num_vectors);
     
     /* Request IRQ for each vector */
     for (i = 0; i < priv->num_vectors; i++) {
         ret = request_irq(priv->msix_entries[i].vector,
                          msix_interrupt_handler,
                          0,
                          DRIVER_NAME,
                          priv);
         if (ret) {
             pcie_err("Failed to request MSI-X IRQ %d", i);
             /* Free already allocated IRQs */
             while (--i >= 0)
                 free_irq(priv->msix_entries[i].vector, priv);
             pci_disable_msix(priv->pdev);
             kfree(priv->msix_entries);
             return ret;
         }
     }
     
     priv->irq_mode = IRQ_MODE_MSIX;
     return 0;
 }
 
 /* Setup MSI */
 static int setup_msi(struct interrupt_private *priv)
 {
     int ret;
     
     ret = pci_enable_msi(priv->pdev);
     if (ret) {
         pcie_err("Failed to enable MSI: %d", ret);
         return ret;
     }
     
     ret = request_irq(priv->pdev->irq,
                      msi_interrupt_handler,
                      0,
                      DRIVER_NAME,
                      priv);
     if (ret) {
         pcie_err("Failed to request MSI IRQ");
         pci_disable_msi(priv->pdev);
         return ret;
     }
     
     priv->irq_mode = IRQ_MODE_MSI;
     pcie_info("MSI enabled");
     return 0;
 }
 
 /* Setup Legacy INTx */
 static int setup_intx(struct interrupt_private *priv)
 {
     int ret;
     
     ret = request_irq(priv->pdev->irq,
                      intx_interrupt_handler,
                      IRQF_SHARED,
                      DRIVER_NAME,
                      priv);
     if (ret) {
         pcie_err("Failed to request INTx IRQ");
         return ret;
     }
     
     priv->irq_mode = IRQ_MODE_INTX;
     pcie_info("Legacy INTx enabled");
     return 0;
 }
 
 /* Character device operations for stats */
 static ssize_t irq_read_stats(struct file *filp, char __user *buf,
                               size_t count, loff_t *f_pos)
 {
     struct interrupt_private *priv = filp->private_data;
     char stats_buf[512];
     int len;
     
     if (*f_pos > 0)
         return 0;
     
     spin_lock(&priv->stats_lock);
     len = snprintf(stats_buf, sizeof(stats_buf),
                   "Interrupt Statistics:\n"
                   "  IRQ Mode:        %s\n"
                   "  Vectors:         %d\n"
                   "  IRQ Count:       %llu\n"
                   "  Tasklet Count:   %llu\n"
                   "  Workqueue Count: %llu\n"
                   "  Spurious IRQs:   %llu\n",
                   priv->irq_mode == IRQ_MODE_MSIX ? "MSI-X" :
                   priv->irq_mode == IRQ_MODE_MSI ? "MSI" : "INTx",
                   priv->num_vectors,
                   priv->stats.irq_count,
                   priv->stats.tasklet_count,
                   priv->stats.workqueue_count,
                   priv->stats.spurious);
     spin_unlock(&priv->stats_lock);
     
     if (count > len)
         count = len;
     
     if (copy_to_user(buf, stats_buf, count))
         return -EFAULT;
     
     *f_pos += count;
     return count;
 }
 
 static int irq_open(struct inode *inode, struct file *filp)
 {
     filp->private_data = global_priv;
     return 0;
 }
 
 static const struct file_operations irq_fops = {
     .owner   = THIS_MODULE,
     .open    = irq_open,
     .read    = irq_read_stats,
     .llseek  = default_llseek,
 };
 
 static int pcie_interrupt_probe(struct pci_dev *pdev,
                                 const struct pci_device_id *id)
 {
     struct interrupt_private *priv;
     int ret;
     
     pcie_info("Probing device");
     
     priv = kzalloc(sizeof(*priv), GFP_KERNEL);
     if (!priv)
         return -ENOMEM;
     
     priv->pdev = pdev;
     pci_set_drvdata(pdev, priv);
     spin_lock_init(&priv->stats_lock);
     
     /* Enable device */
     ret = pci_enable_device(pdev);
     if (ret) {
         pcie_err("Failed to enable device");
         goto err_free_priv;
     }
     
     ret = pci_request_regions(pdev, DRIVER_NAME);
     if (ret) {
         pcie_err("Failed to request regions");
         goto err_disable_device;
     }
     
     /* Map BAR0 */
     priv->bar0 = pci_iomap(pdev, BAR_0, pci_resource_len(pdev, BAR_0));
     if (!priv->bar0) {
         pcie_err("Failed to map BAR0");
         ret = -ENOMEM;
         goto err_release_regions;
     }
     
     /* Initialize tasklet and workqueue */
     tasklet_init(&priv->tasklet, interrupt_tasklet, (unsigned long)priv);
     
     priv->wq = create_singlethread_workqueue(DRIVER_NAME);
     if (!priv->wq) {
         ret = -ENOMEM;
         goto err_unmap_bar;
     }
     INIT_WORK(&priv->work, interrupt_work);
     
     /* Try MSI-X first, then MSI, then INTx */
     ret = setup_msix(priv);
     if (ret) {
         pcie_info("MSI-X not available, trying MSI");
         ret = setup_msi(priv);
         if (ret) {
             pcie_info("MSI not available, using legacy INTx");
             ret = setup_intx(priv);
             if (ret)
                 goto err_destroy_wq;
         }
     }
     
     /* Create character device for stats */
     ret = alloc_chrdev_region(&priv->devt, 0, 1, DRIVER_NAME);
     if (ret)
         goto err_free_irq;
     
     cdev_init(&priv->cdev, &irq_fops);
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
     pcie_info("Read stats: cat /dev/%s", DRIVER_NAME);
     
     return 0;
     
 err_destroy_class:
     class_destroy(priv->class);
 err_del_cdev:
     cdev_del(&priv->cdev);
 err_unregister_chrdev:
     unregister_chrdev_region(priv->devt, 1);
 err_free_irq:
     if (priv->irq_mode == IRQ_MODE_MSIX) {
         int i;
         for (i = 0; i < priv->num_vectors; i++)
             free_irq(priv->msix_entries[i].vector, priv);
         pci_disable_msix(pdev);
         kfree(priv->msix_entries);
     } else if (priv->irq_mode == IRQ_MODE_MSI) {
         free_irq(pdev->irq, priv);
         pci_disable_msi(pdev);
     } else {
         free_irq(pdev->irq, priv);
     }
 err_destroy_wq:
     destroy_workqueue(priv->wq);
 err_unmap_bar:
     tasklet_kill(&priv->tasklet);
     pci_iounmap(pdev, priv->bar0);
 err_release_regions:
     pci_release_regions(pdev);
 err_disable_device:
     pci_disable_device(pdev);
 err_free_priv:
     kfree(priv);
     return ret;
 }
 
 static void pcie_interrupt_remove(struct pci_dev *pdev)
 {
     struct interrupt_private *priv = pci_get_drvdata(pdev);
     
     pcie_info("Removing device");
     
     device_destroy(priv->class, priv->devt);
     class_destroy(priv->class);
     cdev_del(&priv->cdev);
     unregister_chrdev_region(priv->devt, 1);
     
     /* Free IRQs */
     if (priv->irq_mode == IRQ_MODE_MSIX) {
         int i;
         for (i = 0; i < priv->num_vectors; i++)
             free_irq(priv->msix_entries[i].vector, priv);
         pci_disable_msix(pdev);
         kfree(priv->msix_entries);
     } else if (priv->irq_mode == IRQ_MODE_MSI) {
         free_irq(pdev->irq, priv);
         pci_disable_msi(pdev);
     } else {
         free_irq(pdev->irq, priv);
     }
     
     tasklet_kill(&priv->tasklet);
     cancel_work_sync(&priv->work);
     destroy_workqueue(priv->wq);
     
     pci_iounmap(pdev, priv->bar0);
     pci_release_regions(pdev);
     pci_disable_device(pdev);
     
     global_priv = NULL;
     kfree(priv);
     
     pcie_info("Device removed");
 }
 
 static const struct pci_device_id pcie_interrupt_id_table[] = {
     { PCI_DEVICE(DEMO_VENDOR_ID, DEMO_DEVICE_ID) },
     { 0, }
 };
 MODULE_DEVICE_TABLE(pci, pcie_interrupt_id_table);
 
 static struct pci_driver pcie_interrupt_driver = {
     .name       = DRIVER_NAME,
     .id_table   = pcie_interrupt_id_table,
     .probe      = pcie_interrupt_probe,
     .remove     = pcie_interrupt_remove,
 };
 
 module_pci_driver(pcie_interrupt_driver);