# PCIe & NIC Driver Quick Reference

## Build Commands

```bash
# Build all
make

# Build specific
make pcie_enum_driver.ko
make nic_napi_driver.ko

# Clean
make clean
```

## Load/Unload

```bash
# Load
sudo insmod pcie_dma_driver.ko
sudo insmod nic_napi_driver.ko

# Unload
sudo rmmod nic_napi_driver
sudo rmmod pcie_dma_driver

# View logs
dmesg | tail -30
```

## Core PCIe APIs

```c
/* Device Setup */
pci_enable_device(pdev);
pci_request_regions(pdev, "driver_name");
pci_set_master(pdev);  // Enable DMA

/* BAR Mapping */
bar = pci_iomap(pdev, BAR_0, size);
val = ioread32(bar + offset);
iowrite32(val, bar + offset);

/* Interrupts */
request_irq(pdev->irq, handler, IRQF_SHARED, name, dev);
pci_enable_msi(pdev);
pci_enable_msix_range(pdev, entries, min, max);

/* DMA */
dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
buf = dma_alloc_coherent(&pdev->dev, size, &dma_handle, GFP_KERNEL);
dma_addr = dma_map_single(&pdev->dev, buf, size, direction);

/* Cleanup */
pci_iounmap(pdev, bar);
pci_release_regions(pdev);
pci_disable_device(pdev);
```

## Core NIC APIs

```c
/* Device Setup */
netdev = alloc_etherdev(sizeof(priv));
netdev->netdev_ops = &my_netdev_ops;
register_netdev(netdev);

/* NAPI */
netif_napi_add(netdev, &napi, poll_fn, weight);
napi_enable(&napi);
napi_schedule(&napi);
napi_complete_done(&napi, work_done);

/* TX */
netif_start_queue(netdev);
netif_stop_queue(netdev);
dev_kfree_skb(skb);

/* RX */
skb = netdev_alloc_skb(netdev, size);
skb_put(skb, len);
skb->protocol = eth_type_trans(skb, netdev);
napi_gro_receive(&napi, skb);

/* Cleanup */
unregister_netdev(netdev);
free_netdev(netdev);
```

## net_device_ops

```c
static const struct net_device_ops ops = {
    .ndo_open            = my_open,
    .ndo_stop            = my_stop,
    .ndo_start_xmit      = my_xmit,
    .ndo_get_stats       = my_stats,
    .ndo_set_mac_address = my_set_mac,
    .ndo_validate_addr   = eth_validate_addr,
};
```

## Interrupt Flow

```c
/* Top Half (IRQ context) */
irqreturn_t handler(int irq, void *dev_id) {
    // Read status
    status = ioread32(bar + STATUS_REG);
    
    // Clear interrupt
    iowrite32(status, bar + STATUS_REG);
    
    // Disable further interrupts
    iowrite32(0, bar + MASK_REG);
    
    // Schedule bottom half
    napi_schedule(&napi);  // or tasklet_schedule()
    
    return IRQ_HANDLED;
}

/* Bottom Half */
int poll(struct napi_struct *napi, int budget) {
    // Process packets
    work_done = process_rx(budget);
    
    if (work_done < budget) {
        napi_complete(napi);
        // Re-enable interrupts
        iowrite32(1, bar + MASK_REG);
    }
    
    return work_done;
}
```

## DMA Descriptor Ring

```c
/* Allocate Ring */
ring = dma_alloc_coherent(&pdev->dev,
                          count * sizeof(*desc),
                          &ring_dma,
                          GFP_KERNEL);

/* Setup Descriptor */
desc[i].addr = buffer_dma;
desc[i].length = size;
desc[i].flags = DESC_OWN;

/* Process Completed */
while (desc[tail].flags & DESC_DONE) {
    process_buffer(tail);
    tail = (tail + 1) % count;
}
```

## Multi-Queue Setup

```c
/* Set queue count */
netif_set_real_num_tx_queues(netdev, num_queues);
netif_set_real_num_rx_queues(netdev, num_queues);

/* Get queue */
txq = netdev_get_tx_queue(netdev, queue_id);

/* Per-queue processing */
for (i = 0; i < num_queues; i++) {
    process_queue(i);
}
```

## Debugging

```bash
# View PCI devices
lspci -vv

# Check interrupts
cat /proc/interrupts | grep eth0

# Network stats
ethtool -S eth0
ip -s link show eth0

# Module info
modinfo pcie_dma_driver.ko

# Dynamic debug
echo 'module pcie_dma_driver +p' > /sys/kernel/debug/dynamic_debug/control
```

## Common Issues

### Driver not loading
```bash
# Check dependencies
modinfo driver.ko

# Check dmesg
dmesg | tail -50
```

### No interrupts
```bash
# Check /proc/interrupts
cat /proc/interrupts

# Verify MSI enabled
lspci -vv | grep MSI
```

### DMA not working
```bash
# Check DMA mask
dmesg | grep DMA

# Verify bus mastering
lspci -vv | grep "Bus Master"
```

### Network not working
```bash
# Check interface
ip link show

# Bring up
ip link set eth0 up

# Check stats
ethtool -S eth0
```

## Performance Tuning

```bash
# Interrupt coalescing
ethtool -C eth0 rx-usecs 100

# Ring size
ethtool -G eth0 rx 4096 tx 4096

# Queue count
ethtool -L eth0 combined 8

# CPU affinity
echo f > /proc/irq/50/smp_affinity
```

## Memory Barriers

```c
wmb();  // Write memory barrier
rmb();  // Read memory barrier
mb();   // Full memory barrier
smp_wmb();  // SMP write barrier
```

## PCI Config Space

```c
/* Read config */
pci_read_config_word(pdev, PCI_VENDOR_ID, &vendor);
pci_read_config_word(pdev, PCI_DEVICE_ID, &device);
pci_read_config_byte(pdev, PCI_REVISION_ID, &rev);

/* Write config */
pci_write_config_word(pdev, offset, value);
```

## File Locations

```
/sys/bus/pci/devices/     - PCI device info
/sys/class/net/           - Network devices
/proc/interrupts          - Interrupt counters
/dev/pcie_*               - Character devices
```

## Testing

```bash
# Packet generation
ping -f 192.168.1.1
iperf3 -c server -t 60

# Stress test
while true; do ifconfig eth0 down; ifconfig eth0 up; done
```