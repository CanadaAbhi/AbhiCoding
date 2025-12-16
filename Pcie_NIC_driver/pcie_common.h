/**
 * @file pcie_common.h
 * @brief Common definitions for PCIe and NIC drivers
 */

 #ifndef PCIE_COMMON_H
 #define PCIE_COMMON_H
 
 #include <linux/module.h>
 #include <linux/kernel.h>
 #include <linux/init.h>
 #include <linux/pci.h>
 #include <linux/interrupt.h>
 #include <linux/dma-mapping.h>
 #include <linux/cdev.h>
 #include <linux/fs.h>
 #include <linux/uaccess.h>
 #include <linux/slab.h>
 #include <linux/delay.h>
 #include <linux/netdevice.h>
 #include <linux/etherdevice.h>
 #include <linux/skbuff.h>
 
 /* Common Vendor/Device IDs (use generic values for demo) */
 #define DEMO_VENDOR_ID      0x10EC  /* Realtek example */
 #define DEMO_DEVICE_ID      0x8168  /* RTL8168 example */
 
 /* BAR Definitions */
 #define BAR_0               0
 #define BAR_1               1
 #define BAR_SIZE            0x1000  /* 4KB */
 
 /* DMA Definitions */
 #define DMA_BUF_SIZE        4096
 #define MAX_DMA_BUFFERS     16
 
 /* Ring Buffer Definitions */
 #define TX_RING_SIZE        256
 #define RX_RING_SIZE        256
 #define MAX_PACKET_SIZE     1536
 
 /* MSI/MSI-X */
 #define MAX_MSIX_VECTORS    8
 
 /* Device States */
 #define DEV_STATE_INIT      0
 #define DEV_STATE_RUNNING   1
 #define DEV_STATE_STOPPED   2
 #define DEV_STATE_ERROR     3
 
 /* IOCTL Commands */
 #define PCIE_IOC_MAGIC      'P'
 #define PCIE_IOC_RESET      _IO(PCIE_IOC_MAGIC, 0)
 #define PCIE_IOC_GET_INFO   _IOR(PCIE_IOC_MAGIC, 1, struct pcie_dev_info)
 #define PCIE_IOC_READ_REG   _IOWR(PCIE_IOC_MAGIC, 2, struct pcie_reg_access)
 #define PCIE_IOC_WRITE_REG  _IOW(PCIE_IOC_MAGIC, 3, struct pcie_reg_access)
 #define PCIE_IOC_DMA_TEST   _IOW(PCIE_IOC_MAGIC, 4, struct pcie_dma_test)
 
 /* Structures */
 struct pcie_dev_info {
     u16 vendor_id;
     u16 device_id;
     u32 bar_size[6];
     u8 irq;
     u32 state;
 };
 
 struct pcie_reg_access {
     u32 offset;
     u32 value;
 };
 
 struct pcie_dma_test {
     u64 src_addr;
     u64 dst_addr;
     u32 size;
 };
 
 /* DMA Descriptor (common format) */
 struct dma_descriptor {
     u64 buffer_addr;
     u32 length;
     u32 flags;
     u32 status;
     u32 reserved;
 } __attribute__((packed));
 
 /* RX Descriptor */
 struct rx_descriptor {
     u64 buffer_addr;
     u16 length;
     u16 vlan;
     u32 rss_hash;
     u16 packet_checksum;
     u16 status;
 } __attribute__((packed));
 
 /* TX Descriptor */
 struct tx_descriptor {
     u64 buffer_addr;
     u16 length;
     u8 cmd;
     u8 status;
     u16 vlan;
     u16 special;
 } __attribute__((packed));
 
 /* Common device private structure */
 struct pcie_dev_common {
     struct pci_dev *pdev;
     void __iomem *bar0_addr;
     void __iomem *bar1_addr;
     
     /* DMA */
     dma_addr_t dma_handle;
     void *dma_virt;
     
     /* Character device */
     struct cdev cdev;
     dev_t devt;
     struct class *class;
     struct device *device;
     
     /* State */
     u32 state;
     spinlock_t lock;
     
     /* Statistics */
     u64 tx_packets;
     u64 rx_packets;
     u64 tx_bytes;
     u64 rx_bytes;
     u64 errors;
 };
 
 /* Helper macros */
 #define pcie_read32(addr, offset)   ioread32((addr) + (offset))
 #define pcie_write32(addr, offset, val) iowrite32((val), (addr) + (offset))
 
 #define pcie_read16(addr, offset)   ioread16((addr) + (offset))
 #define pcie_write16(addr, offset, val) iowrite16((val), (addr) + (offset))
 
 #define pcie_read8(addr, offset)    ioread8((addr) + (offset))
 #define pcie_write8(addr, offset, val)  iowrite8((val), (addr) + (offset))
 
 /* Debugging */
 #define PCIE_DEBUG
 #ifdef PCIE_DEBUG
 #define pcie_dbg(fmt, ...) \
     pr_debug("PCIE_DRV: " fmt "\n", ##__VA_ARGS__)
 #else
 #define pcie_dbg(fmt, ...) do {} while(0)
 #endif
 
 #define pcie_info(fmt, ...) \
     pr_info("PCIE_DRV: " fmt "\n", ##__VA_ARGS__)
 
 #define pcie_err(fmt, ...) \
     pr_err("PCIE_DRV: " fmt "\n", ##__VA_ARGS__)
 
 #endif /* PCIE_COMMON_H */