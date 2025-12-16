/**
 * @file pcie_bar_mmio_driver.c
 * @brief PCIe BAR Memory-Mapped I/O Driver
 * 
 * Features:
 * - Map BAR memory regions
 * - Read/write registers via MMIO
 * - Handle memory barriers
 * - Expose via character device
 */

 #include "../common/pcie_common.h"

 #define DRIVER_NAME "pcie_bar_mmio"
 #define DRIVER_VERSION "1.0"
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("Abhi");
 MODULE_DESCRIPTION("PCIe BAR Memory-Mapped I/O Driver");
 MODULE_VERSION(DRIVER_VERSION);
 
 struct bar_mmio_private {
     struct pci_dev *pdev;
     void __iomem *bar0;
     void __iomem *bar1;
     resource_size_t bar0_size;
     resource_size_t bar1_size;
     
     /* Character device */
     struct cdev cdev;
     dev_t devt;
     struct class *class;
     struct device *device;
     
     spinlock_t lock;
 };
 
 static struct bar_mmio_private *global_priv = NULL;
 
 /* Register read/write functions */
 static u32 bar_read_reg32(struct bar_mmio_private *priv, u32 offset)
 {
     u32 val;
     
     if (offset >= priv->bar0_size) {
         pcie_err("Read offset 0x%x beyond BAR0 size", offset);
         return 0xFFFFFFFF;
     }
     
     val = ioread32(priv->bar0 + offset);
     mb(); /* Memory barrier */
     
     pcie_dbg("Read BAR0[0x%x] = 0x%08x", offset, val);
     return val;
 }
 
 static void bar_write_reg32(struct bar_mmio_private *priv, u32 offset, u32 val)
 {
     if (offset >= priv->bar0_size) {
         pcie_err("Write offset 0x%x beyond BAR0 size", offset);
         return;
     }
     
     wmb(); /* Write memory barrier */
     iowrite32(val, priv->bar0 + offset);
     mb(); /* Memory barrier */
     
     pcie_dbg("Write BAR0[0x%x] = 0x%08x", offset, val);
 }
 
 /* Character device operations */
 static int bar_mmio_open(struct inode *inode, struct file *filp)
 {
     filp->private_data = global_priv;
     pcie_dbg("Device opened");
     return 0;
 }
 
 static int bar_mmio_release(struct inode *inode, struct file *filp)
 {
     pcie_dbg("Device closed");
     return 0;
 }
 
 static ssize_t bar_mmio_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
 {
     struct bar_mmio_private *priv = filp->private_data;
     u32 *kbuf;
     ssize_t ret;
     size_t i;
     
     if (*f_pos >= priv->bar0_size)
         return 0;
     
     if (*f_pos + count > priv->bar0_size)
         count = priv->bar0_size - *f_pos;
     
     /* Align to 4-byte boundary */
     count = (count / 4) * 4;
     if (count == 0)
         return 0;
     
     kbuf = kmalloc(count, GFP_KERNEL);
     if (!kbuf)
         return -ENOMEM;
     
     /* Read from BAR */
     for (i = 0; i < count / 4; i++)
         kbuf[i] = bar_read_reg32(priv, *f_pos + (i * 4));
     
     ret = copy_to_user(buf, kbuf, count);
     if (ret) {
         kfree(kbuf);
         return -EFAULT;
     }
     
     *f_pos += count;
     kfree(kbuf);
     
     return count;
 }
 
 static ssize_t bar_mmio_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
 {
     struct bar_mmio_private *priv = filp->private_data;
     u32 *kbuf;
     ssize_t ret;
     size_t i;
     
     if (*f_pos >= priv->bar0_size)
         return -EINVAL;
     
     if (*f_pos + count > priv->bar0_size)
         count = priv->bar0_size - *f_pos;
     
     /* Align to 4-byte boundary */
     count = (count / 4) * 4;
     if (count == 0)
         return -EINVAL;
     
     kbuf = kmalloc(count, GFP_KERNEL);
     if (!kbuf)
         return -ENOMEM;
     
     ret = copy_from_user(kbuf, buf, count);
     if (ret) {
         kfree(kbuf);
         return -EFAULT;
     }
     
     /* Write to BAR */
     for (i = 0; i < count / 4; i++)
         bar_write_reg32(priv, *f_pos + (i * 4), kbuf[i]);
     
     *f_pos += count;
     kfree(kbuf);
     
     return count;
 }
 
 static long bar_mmio_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
 {
     struct bar_mmio_private *priv = filp->private_data;
     struct pcie_reg_access reg_access;
     int ret = 0;
     
     switch (cmd) {
     case PCIE_IOC_READ_REG:
         if (copy_from_user(&reg_access, (void __user *)arg, sizeof(reg_access)))
             return -EFAULT;
         
         reg_access.value = bar_read_reg32(priv, reg_access.offset);
         
         if (copy_to_user((void __user *)arg, &reg_access, sizeof(reg_access)))
             return -EFAULT;
         break;
         
     case PCIE_IOC_WRITE_REG:
         if (copy_from_user(&reg_access, (void __user *)arg, sizeof(reg_access)))
             return -EFAULT;
         
         bar_write_reg32(priv, reg_access.offset, reg_access.value);
         break;
         
     default:
         ret = -EINVAL;
     }
     
     return ret;
 }
 
 static const struct file_operations bar_mmio_fops = {
     .owner          = THIS_MODULE,
     .open           = bar_mmio_open,
     .release        = bar_mmio_release,
     .read           = bar_mmio_read,
     .write          = bar_mmio_write,
     .unlocked_ioctl = bar_mmio_ioctl,
     .llseek         = default_llseek,
 };
 
 static int pcie_bar_mmio_probe(struct pci_dev *pdev,
                                const struct pci_device_id *id)
 {
     struct bar_mmio_private *priv;
     int ret;
     
     pcie_info("Probing device");
     
     priv = kzalloc(sizeof(*priv), GFP_KERNEL);
     if (!priv)
         return -ENOMEM;
     
     priv->pdev = pdev;
     pci_set_drvdata(pdev, priv);
     spin_lock_init(&priv->lock);
     
     /* Enable device */
     ret = pci_enable_device_mem(pdev);
     if (ret) {
         pcie_err("Failed to enable device");
         goto err_free_priv;
     }
     
     /* Request memory regions */
     ret = pci_request_regions(pdev, DRIVER_NAME);
     if (ret) {
         pcie_err("Failed to request regions");
         goto err_disable_device;
     }
     
     /* Map BAR0 */
     priv->bar0_size = pci_resource_len(pdev, BAR_0);
     priv->bar0 = pci_iomap(pdev, BAR_0, priv->bar0_size);
     if (!priv->bar0) {
         pcie_err("Failed to map BAR0");
         ret = -ENOMEM;
         goto err_release_regions;
     }
     
     pcie_info("BAR0 mapped: size=0x%llx", (u64)priv->bar0_size);
     
     /* Map BAR1 if available */
     if (pci_resource_start(pdev, BAR_1)) {
         priv->bar1_size = pci_resource_len(pdev, BAR_1);
         priv->bar1 = pci_iomap(pdev, BAR_1, priv->bar1_size);
         if (priv->bar1)
             pcie_info("BAR1 mapped: size=0x%llx", (u64)priv->bar1_size);
     }
     
     /* Create character device */
     ret = alloc_chrdev_region(&priv->devt, 0, 1, DRIVER_NAME);
     if (ret) {
         pcie_err("Failed to allocate chrdev region");
         goto err_unmap_bars;
     }
     
     cdev_init(&priv->cdev, &bar_mmio_fops);
     priv->cdev.owner = THIS_MODULE;
     
     ret = cdev_add(&priv->cdev, priv->devt, 1);
     if (ret) {
         pcie_err("Failed to add cdev");
         goto err_unregister_chrdev;
     }
     
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
     
     pcie_info("Device created: /dev/%s", DRIVER_NAME);
     pcie_info("Driver loaded successfully");
     
     return 0;
     
 err_destroy_class:
     class_destroy(priv->class);
 err_del_cdev:
     cdev_del(&priv->cdev);
 err_unregister_chrdev:
     unregister_chrdev_region(priv->devt, 1);
 err_unmap_bars:
     if (priv->bar1)
         pci_iounmap(pdev, priv->bar1);
     if (priv->bar0)
         pci_iounmap(pdev, priv->bar0);
 err_release_regions:
     pci_release_regions(pdev);
 err_disable_device:
     pci_disable_device(pdev);
 err_free_priv:
     kfree(priv);
     return ret;
 }
 
 static void pcie_bar_mmio_remove(struct pci_dev *pdev)
 {
     struct bar_mmio_private *priv = pci_get_drvdata(pdev);
     
     pcie_info("Removing device");
     
     device_destroy(priv->class, priv->devt);
     class_destroy(priv->class);
     cdev_del(&priv->cdev);
     unregister_chrdev_region(priv->devt, 1);
     
     if (priv->bar1)
         pci_iounmap(pdev, priv->bar1);
     if (priv->bar0)
         pci_iounmap(pdev, priv->bar0);
     
     pci_release_regions(pdev);
     pci_disable_device(pdev);
     
     global_priv = NULL;
     kfree(priv);
     
     pcie_info("Device removed");
 }
 
 static const struct pci_device_id pcie_bar_mmio_id_table[] = {
     { PCI_DEVICE(DEMO_VENDOR_ID, DEMO_DEVICE_ID) },
     { 0, }
 };
 MODULE_DEVICE_TABLE(pci, pcie_bar_mmio_id_table);
 
 static struct pci_driver pcie_bar_mmio_driver = {
     .name       = DRIVER_NAME,
     .id_table   = pcie_bar_mmio_id_table,
     .probe      = pcie_bar_mmio_probe,
     .remove     = pcie_bar_mmio_remove,
 };
 
 module_pci_driver(pcie_bar_mmio_driver);