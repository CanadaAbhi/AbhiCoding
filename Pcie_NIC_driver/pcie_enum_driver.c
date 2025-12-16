/**
 * @file pcie_enum_driver.c
 * @brief Basic PCIe Enumeration Driver
 * 
 * Features:
 * - Detect PCIe device
 * - Read Vendor/Device ID
 * - Read BAR regions
 * - Enable device
 */

 #include "../common/pcie_common.h"

 #define DRIVER_NAME "pcie_enum"
 #define DRIVER_VERSION "1.0"
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("Abhi");
 MODULE_DESCRIPTION("Basic PCIe Enumeration Driver");
 MODULE_VERSION(DRIVER_VERSION);
 
 /* Device private data */
 struct enum_dev_private {
     struct pci_dev *pdev;
     u16 vendor_id;
     u16 device_id;
     u8 revision;
     u32 bar_start[6];
     u32 bar_size[6];
     u8 irq;
 };
 
 static int pcie_enum_probe(struct pci_dev *pdev,
                            const struct pci_device_id *id)
 {
     struct enum_dev_private *priv;
     int ret, i;
     
     pcie_info("Probing device %04x:%04x", id->vendor, id->device);
     
     /* Allocate private data */
     priv = kzalloc(sizeof(*priv), GFP_KERNEL);
     if (!priv)
         return -ENOMEM;
     
     priv->pdev = pdev;
     pci_set_drvdata(pdev, priv);
     
     /* Enable PCI device */
     ret = pci_enable_device(pdev);
     if (ret) {
         pcie_err("Failed to enable PCI device: %d", ret);
         goto err_free_priv;
     }
     
     /* Read device information */
     pci_read_config_word(pdev, PCI_VENDOR_ID, &priv->vendor_id);
     pci_read_config_word(pdev, PCI_DEVICE_ID, &priv->device_id);
     pci_read_config_byte(pdev, PCI_REVISION_ID, &priv->revision);
     priv->irq = pdev->irq;
     
     pcie_info("Device Info:");
     pcie_info("  Vendor ID:  0x%04x", priv->vendor_id);
     pcie_info("  Device ID:  0x%04x", priv->device_id);
     pcie_info("  Revision:   0x%02x", priv->revision);
     pcie_info("  IRQ:        %d", priv->irq);
     
     /* Read BAR information */
     pcie_info("BAR Information:");
     for (i = 0; i < 6; i++) {
         if (pci_resource_start(pdev, i)) {
             priv->bar_start[i] = pci_resource_start(pdev, i);
             priv->bar_size[i] = pci_resource_len(pdev, i);
             
             pcie_info("  BAR%d: 0x%08x - 0x%08x (size: 0x%08x) [%s]",
                      i,
                      priv->bar_start[i],
                      priv->bar_start[i] + priv->bar_size[i] - 1,
                      priv->bar_size[i],
                      (pci_resource_flags(pdev, i) & IORESOURCE_MEM) ? "MEM" : "IO");
         }
     }
     
     /* Request regions */
     ret = pci_request_regions(pdev, DRIVER_NAME);
     if (ret) {
         pcie_err("Failed to request PCI regions: %d", ret);
         goto err_disable_device;
     }
     
     pcie_info("Successfully enumerated PCIe device");
     return 0;
     
 err_disable_device:
     pci_disable_device(pdev);
 err_free_priv:
     kfree(priv);
     return ret;
 }
 
 static void pcie_enum_remove(struct pci_dev *pdev)
 {
     struct enum_dev_private *priv = pci_get_drvdata(pdev);
     
     pcie_info("Removing device");
     
     pci_release_regions(pdev);
     pci_disable_device(pdev);
     kfree(priv);
     
     pcie_info("Device removed");
 }
 
 /* PCI device ID table */
 static const struct pci_device_id pcie_enum_id_table[] = {
     { PCI_DEVICE(DEMO_VENDOR_ID, DEMO_DEVICE_ID) },
     { PCI_ANY_ID, PCI_ANY_ID, PCI_ANY_ID, PCI_ANY_ID }, /* Match any device for demo */
     { 0, }
 };
 MODULE_DEVICE_TABLE(pci, pcie_enum_id_table);
 
 /* PCI driver structure */
 static struct pci_driver pcie_enum_driver = {
     .name       = DRIVER_NAME,
     .id_table   = pcie_enum_id_table,
     .probe      = pcie_enum_probe,
     .remove     = pcie_enum_remove,
 };
 
 static int __init pcie_enum_init(void)
 {
     int ret;
     
     pcie_info("Loading %s v%s", DRIVER_NAME, DRIVER_VERSION);
     
     ret = pci_register_driver(&pcie_enum_driver);
     if (ret) {
         pcie_err("Failed to register PCI driver: %d", ret);
         return ret;
     }
     
     pcie_info("Driver loaded successfully");
     return 0;
 }
 
 static void __exit pcie_enum_exit(void)
 {
     pcie_info("Unloading driver");
     pci_unregister_driver(&pcie_enum_driver);
     pcie_info("Driver unloaded");
 }
 
 module_init(pcie_enum_init);
 module_exit(pcie_enum_exit);