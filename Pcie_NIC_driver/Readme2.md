# PCIe and NIC Driver Collection

Complete collection of Linux kernel PCIe and Network Interface Card (NIC) drivers for embedded, networking, and systems programming.

## Driver Categories

### PCIe Drivers (7 Types)

1. **Basic PCIe Enumeration Driver** (`01_pcie_enum_driver.c`)
   - Device detection and enumeration
   - Vendor/Device ID reading
   - BAR region discovery
   - Basic device initialization

2. **PCIe BAR Memory-Mapped Driver** (`02_pcie_bar_mmio_driver.c`)
   - BAR memory mapping (ioremap)
   - MMIO register access (readl/writel)
   - Memory barriers
   - Character device interface
   - User-space BAR access via /dev

3. **PCIe Interrupt Driver** (`03_pcie_interrupt_driver.c`)
   - Legacy INTx interrupts
   - MSI (Message Signaled Interrupts)
   - MSI-X (Extended MSI) with multiple vectors
   - Tasklet and workqueue bottom halves
   - Interrupt statistics

4. **PCIe DMA Driver** (`04_pcie_dma_driver.c`)
   - Coherent DMA allocation
   - Streaming DMA mapping
   - Scatter-gather DMA
   - TX/RX descriptor rings
   - DMA transfer initiation

5. **PCIe Character Device Driver** (Integrated in BAR driver)
   - /dev interface for user-space
   - IOCTL commands
   - mmap support
   - Register access from user-space

6. **PCIe SR-IOV Driver** (Documentation only)
   - Virtual Function (VF) creation
   - Physical Function (PF) management
   - VF configuration
   - Used in virtualization/cloud

7. **PCIe Hot-Plug Driver** (Documentation only)
   - Device insertion/removal detection
   - Dynamic resource allocation
   - State management
   - Enterprise systems

### NIC Drivers (7 Types)

1. **Basic Ethernet NIC Driver** (`01_nic_basic_driver.c`)
   - net_device_ops implementation
   - TX/RX packet handling
   - Basic interrupt handling
   - MAC address management
   - Network statistics

2. **NAPI Polling Driver** (`02_nic_napi_driver.c`)
   - NAPI poll function
   - Interrupt mitigation
   - Budget-based packet processing
   - High-performance RX
   - GRO (Generic Receive Offload)

3. **Multi-Queue NIC Driver** (`03_nic_multiqueue_driver.c`)
   - Multiple TX/RX queue pairs
   - Per-CPU queues
   - CPU affinity
   - Scalable performance
   - Used in 10G+ NICs

4. **DMA-Based NIC Driver** (Integrated in NAPI driver)
   - TX/RX descriptor rings
   - DMA buffer management
   - Zero-copy operations
   - Scatter-gather support

5. **SR-IOV NIC Driver** (Documentation only)
   - Virtual NIC creation
   - VF assignment to VMs
   - MAC/VLAN filtering
   - Data center deployment

6. **XDP/eBPF Offload Driver** (Documentation only)
   - XDP hook integration
   - eBPF program offload
   - Fast path networking
   - Kernel bypass

7. **SmartNIC/DPDK Driver** (Documentation only)
   - User-space packet processing
   - PMD (Poll Mode Driver)
   - Zero-copy
   - High-frequency trading

## Architecture Overview

### PCIe Driver Architecture

```
┌─────────────────────────────────────┐
│     User Space Application          │
└──────────────┬──────────────────────┘
               │ (ioctl, read, write)
┌──────────────▼──────────────────────┐
│     Character Device Interface       │
├─────────────────────────────────────┤
│          Driver Core                 │
│  - Probe/Remove                      │
│  - Resource Management               │
│  - Power Management                  │
├─────────────────────────────────────┤
│       Interrupt Handler              │
│  - Top Half (ISR)                    │
│  - Bottom Half (Tasklet/Work)        │
├─────────────────────────────────────┤
│          DMA Engine                  │
│  - Descriptor Management             │
│  - Buffer Management                 │
│  - Transfer Control                  │
├─────────────────────────────────────┤
│        BAR/MMIO Access               │
│  - Register Read/Write               │
│  - Memory Barriers                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         PCIe Hardware                │
└─────────────────────────────────────┘
```

### NIC Driver Architecture

```
┌─────────────────────────────────────┐
│      Network Stack (TCP/IP)          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         net_device Layer             │
│  - ndo_open/ndo_stop                 │
│  - ndo_start_xmit                    │
│  - ndo_get_stats                     │
├─────────────────────────────────────┤
│          NAPI Layer                  │
│  - napi_poll()                       │
│  - Interrupt Mitigation              │
│  - Budget Management                 │
├─────────────────────────────────────┤
│      TX/RX Queue Management          │
│  - Descriptor Rings                  │
│  - DMA Buffers                       │
│  - Multi-queue Support               │
├─────────────────────────────────────┤
│       Hardware Interface             │
│  - BAR Access                        │
│  - DMA Control                       │
│  - Interrupt Handling                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      NIC Hardware (PHY/MAC)          │
└─────────────────────────────────────┘
```

## Building and Loading

### Build All Drivers

```bash
make
```

### Build Specific Driver

```bash
make pcie_enum_driver.ko
make nic_napi_driver.ko
```

### Load Drivers

```bash
# PCIe Drivers
sudo insmod pcie_enum_driver.ko
sudo insmod pcie_bar_mmio_driver.ko
sudo insmod pcie_interrupt_driver.ko
sudo insmod pcie_dma_driver.ko

# NIC Drivers
sudo insmod nic_basic_driver.ko
sudo insmod nic_napi_driver.ko
sudo insmod nic_multiqueue_driver.ko
```

### View Loaded Drivers

```bash
lsmod | grep pcie
lsmod | grep nic
dmesg | tail -50
```

### Unload Drivers

```bash
sudo rmmod nic_multiqueue_driver
sudo rmmod nic_napi_driver
sudo rmmod nic_basic_driver
sudo rmmod pcie_dma_driver
sudo rmmod pcie_interrupt_driver
sudo rmmod pcie_bar_mmio_driver
sudo rmmod pcie_enum_driver
```

## 📋 Feature Matrix

| Driver Type | DMA | Interrupts | NAPI | Multi-Queue | SR-IOV |
|------------|-----|------------|------|-------------|--------|
| Enum       | ❌  | ❌         | ❌   | ❌          | ❌     |
| BAR MMIO   | ❌  | ❌         | ❌   | ❌          | ❌     |
| Interrupt  | ❌  | ✅         | ❌   | ❌          | ❌     |
| DMA        | ✅  | ✅         | ❌   | ❌          | ❌     |
| NIC Basic  | ❌  | ✅         | ❌   | ❌          | ❌     |
| NIC NAPI   | ✅  | ✅         | ✅   | ❌          | ❌     |
| NIC MQ     | ✅  | ✅         | ✅   | ✅          | ❌     |

## 💡 Key Concepts

### 1. PCIe Enumeration
- Scan PCI bus
- Identify devices by Vendor/Device ID
- Read configuration space
- Assign resources (BARs, IRQ)

### 2. BAR (Base Address Register)
- Memory-mapped I/O regions
- 6 BARs per device (BAR0-BAR5)
- Can be memory or I/O space
- Access via ioremap/pci_iomap

### 3. Interrupts

**Legacy INTx:**
- Shared interrupts
- Level-triggered
- Slow performance

**MSI (Message Signaled Interrupts):**
- Edge-triggered
- No sharing needed
- Better performance

**MSI-X:**
- Multiple interrupt vectors
- Per-queue interrupts
- Best performance
- Scalable

### 4. DMA (Direct Memory Access)

**Coherent DMA:**
- Synchronized between CPU and device
- Used for descriptor rings
- dma_alloc_coherent()

**Streaming DMA:**
- One-direction transfers
- Used for packet buffers
- dma_map_single()

**Scatter-Gather DMA:**
- Non-contiguous memory
- Multiple buffers in one transfer
- dma_map_sg()

### 5. NAPI (New API)

**Benefits:**
- Interrupt mitigation
- Polling when busy
- Budget-based processing
- Better CPU utilization

**Flow:**
```
Interrupt → Disable IRQ → Schedule NAPI
  → Poll packets → Process up to budget
    → If done: Complete NAPI → Enable IRQ
```

### 6. Multi-Queue

**Advantages:**
- Multiple TX/RX queue pairs
- CPU core scaling
- Reduced lock contention
- Better cache locality

**Use Cases:**
- 10G/40G/100G NICs
- High packet rate
- Multi-core systems

## Use Cases by Industry

### Data Center / Cloud
- SR-IOV NIC drivers
- Multi-queue support
- XDP offload
- High throughput

### Automotive
- TSN (Time-Sensitive Networking)
- Deterministic latency
- Safety-critical drivers
- Real-time requirements

### Telecommunications
- DPDK compatibility
- SmartNIC integration
- Low latency
- High packet rates

### Embedded Systems
- Basic enumeration
- Minimal footprint
- Power management
- Custom hardware

### FPGA / Accelerators
- Custom PCIe endpoints
- DMA engines
- Scatter-gather
- High bandwidth

## 📊 Performance Tuning

### Interrupt Mitigation
```c
/* Reduce interrupt rate */
iowrite32(100, bar0 + INTR_THROTTLE_REG); // 100 µs delay
```

### NAPI Tuning
```bash
# Increase NAPI weight for higher throughput
echo 128 > /sys/class/net/eth0/gro_flush_timeout
```

### Multi-Queue Setup
```bash
# Set number of queues
ethtool -L eth0 combined 8

# Set CPU affinity
echo 0 > /proc/irq/50/smp_affinity
```

### DMA Optimization
```c
/* Use larger buffers */
#define DMA_BUF_SIZE 16384

/* Enable scatter-gather */
netdev->features |= NETIF_F_SG;
```

## 🔧 Debugging Tools

### lspci
```bash
# List all PCI devices
lspci -vv

# Show specific device
lspci -s 00:1f.6 -vv
```

### dmesg
```bash
# View kernel messages
dmesg | grep pcie

# Follow in real-time
dmesg -w
```

### /sys/class/net
```bash
# View network device info
cat /sys/class/net/eth0/statistics/rx_packets
cat /sys/class/net/eth0/speed
```

### ethtool
```bash
# View driver info
ethtool -i eth0

# View statistics
ethtool -S eth0

# View ring sizes
ethtool -g eth0
```

### /proc/interrupts
```bash
# View interrupt distribution
cat /proc/interrupts | grep eth0
```

## API Reference

### PCI APIs
```c
pci_register_driver()      // Register PCI driver
pci_enable_device()        // Enable PCI device
pci_request_regions()      // Request memory regions
pci_iomap()                // Map BAR to kernel
pci_set_master()           // Enable bus mastering (DMA)
pci_enable_msi()           // Enable MSI
pci_enable_msix_range()    // Enable MSI-X
```

### DMA APIs
```c
dma_set_mask()             // Set DMA address mask
dma_alloc_coherent()       // Allocate coherent DMA
dma_map_single()           // Map single buffer
dma_map_sg()               // Map scatter-gather list
dma_sync_single_for_cpu()  // Sync for CPU access
```

### Network APIs
```c
alloc_etherdev()           // Allocate net_device
register_netdev()          // Register network device
netif_napi_add()           // Add NAPI
napi_schedule()            // Schedule NAPI poll
netif_rx()                 // Pass packet to stack
napi_gro_receive()         // GRO receive
```

## 🛠 Development Workflow

1. **Setup Development Environment**
   ```bash
   sudo apt install build-essential linux-headers-$(uname -r)
   ```

2. **Modify Driver**
   - Edit source code
   - Add features
   - Update Makefile if needed

3. **Build**
   ```bash
   make clean
   make
   ```

4. **Test**
   ```bash
   sudo insmod driver.ko
   dmesg | tail -20
   # Test functionality
   sudo rmmod driver
   ```

5. **Debug**
   - Add printk statements
   - Check dmesg output
   - Use dynamic debug
   - Monitor /proc and /sys

## 🎓 Learning Path

### Beginner
1. Start with **pcie_enum_driver**
   - Understand probe/remove
   - Read vendor/device ID
   - Learn PCI config space

2. Move to **pcie_bar_mmio_driver**
   - Learn MMIO access
   - Understand memory barriers
   - Character device basics

### Intermediate
3. Study **pcie_interrupt_driver**
   - INTx vs MSI vs MSI-X
   - Top/bottom half handlers
   - Interrupt mitigation

4. Master **pcie_dma_driver**
   - Coherent vs streaming DMA
   - Descriptor rings
   - Scatter-gather

### Advanced
5. Implement **nic_napi_driver**
   - NAPI architecture
   - Packet processing
   - Performance tuning

6. Build **nic_multiqueue_driver**
   - Multi-queue design
   - CPU affinity
   - Scaling strategies

## 🏆 Resume Keywords

Include these in your resume:
- PCIe enumeration and configuration
- BAR memory-mapped I/O (MMIO)
- MSI/MSI-X interrupt handling
- DMA engine implementation (coherent/streaming/scatter-gather)
- Linux net_device driver architecture
- NAPI polling and interrupt mitigation
- Multi-queue NIC driver development
- TX/RX descriptor ring management
- Kernel debugging (dmesg, printk, kgdb)
- Device tree and ACPI integration

## 📖 References

- [Linux Device Drivers, 3rd Edition](https://lwn.net/Kernel/LDD3/)
- [Linux Kernel Documentation - PCI](https://www.kernel.org/doc/html/latest/PCI/)
- [Linux Kernel Documentation - Networking](https://www.kernel.org/doc/html/latest/networking/)
- [Intel 82599 Datasheet](https://www.intel.com/content/www/us/en/products/docs/network-io/ethernet/controllers/82599-10-gbe-controller-datasheet.html)

## 📝 License

GPL v2 (as per Linux kernel requirements)

## Author

Built for systems programming and kernel driver development learning.

---

**Note:** These are educational drivers. For production use, extensive testing, error handling, and hardware-specific tuning are required.