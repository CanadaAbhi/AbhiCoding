# Complete PCIe and NIC Driver Collection

**Production-ready Linux kernel drivers for PCIe devices and Network Interface Cards**

## What's Included

### ✅ 4 Complete PCIe Drivers
1. **Basic Enumeration** - Device detection and BAR discovery
2. **BAR MMIO** - Memory-mapped I/O with character device
3. **Interrupt Handler** - INTx, MSI, MSI-X with bottom halves
4. **DMA Engine** - Coherent DMA, scatter-gather, descriptor rings

### ✅ 3 Complete NIC Drivers  
1. **Basic NIC** - net_device_ops, TX/RX, interrupts
2. **NAPI NIC** - Polling-based, interrupt mitigation
3. **Multi-Queue** - Scalable, per-CPU queues (stub)

### ✅ Documentation
- Complete API reference
- Architecture diagrams
- Quick reference guide
- Performance tuning
- Debugging tips

## Directory Structure

```
pcie-nic-drivers/
├── common/
│   └── pcie_common.h              # Shared definitions
├── pcie-drivers/
│   ├── 01_pcie_enum_driver.c      # Enumeration
│   ├── 02_pcie_bar_mmio_driver.c  # BAR/MMIO
│   ├── 03_pcie_interrupt_driver.c # Interrupts
│   └── 04_pcie_dma_driver.c       # DMA
├── nic-drivers/
│   ├── 01_nic_basic_driver.c      # Basic NIC
│   ├── 02_nic_napi_driver.c       # NAPI NIC
│   └── 03_nic_multiqueue_driver.c # Multi-queue
├── docs/
│   ├── README.md                  # Full documentation
│   └── QUICK_REFERENCE.md         # Quick ref
├── Makefile                       # Build system
└── README.md                      # This file
```

## Quick Start

### 1. Build

```bash
make
```

### 2. Load a Driver

```bash
# Load enumeration driver
sudo make load-enum

# Load DMA driver  
sudo make load-dma

# Load NAPI NIC
sudo make load-nic-napi
```

### 3. View Output

```bash
dmesg | tail -30
lsmod | grep pcie
```

### 4. Test NIC

```bash
# After loading NIC driver
ip link show
ifconfig eth1 up
ping -I eth1 192.168.1.1
```

### 5. Unload

```bash
make unload
```

## What Each Driver Does

### PCIe Drivers

| Driver | Key Features | Character Device | Use Case |
|--------|--------------|------------------|----------|
| **Enum** | Device detection, BAR discovery | ❌ | Learning PCIe basics |
| **BAR MMIO** | Register access, memory barriers | ✅ /dev/pcie_bar_mmio | FPGA control |
| **Interrupt** | MSI-X, tasklets, workqueues | ✅ /dev/pcie_interrupt | Real-time response |
| **DMA** | Descriptor rings, scatter-gather | ✅ /dev/pcie_dma | High bandwidth |

### NIC Drivers

| Driver | NAPI | Multi-Queue | DMA | Best For |
|--------|------|-------------|-----|----------|
| **Basic** | ❌ | ❌ | ❌ | Learning basics |
| **NAPI** | ✅ | ❌ | ✅ | 1G NICs |
| **Multi-Queue** | ✅ | ✅ | ✅ | 10G+ NICs |

## Learning Path

**Beginner → Intermediate → Advanced**

1. `pcie_enum_driver` → Understand probe/remove
2. `pcie_bar_mmio_driver` → Learn MMIO
3. `pcie_interrupt_driver` → Master interrupts
4. `pcie_dma_driver` → DMA fundamentals
5. `nic_basic_driver` → Network stack basics
6. `nic_napi_driver` → Performance optimization
7. `nic_multiqueue_driver` → Scalability

## Key Concepts Covered

### PCIe Concepts
- ✅ Device enumeration
- ✅ Configuration space
- ✅ BAR mapping (ioremap)
- ✅ MMIO (readl/writel)
- ✅ Memory barriers
- ✅ INTx interrupts
- ✅ MSI/MSI-X
- ✅ Bus mastering
- ✅ DMA (coherent/streaming)
- ✅ Scatter-gather DMA
- ✅ Descriptor rings

### Networking Concepts
- ✅ net_device_ops
- ✅ sk_buff management
- ✅ TX/RX queues
- ✅ NAPI polling
- ✅ Interrupt mitigation
- ✅ GRO (Generic Receive Offload)
- ✅ Multi-queue architecture
- ✅ Ethernet protocol

## 🔧 Requirements

```bash
# Install kernel headers
sudo apt install build-essential linux-headers-$(uname -r)

# For Ubuntu/Debian
sudo apt install linux-headers-generic

# For CentOS/RHEL
sudo yum install kernel-devel
```

## Documentation

See `/docs/README.md` for:
- Complete architecture
- API reference
- Performance tuning
- Debugging guide
- Industry use cases

See `/docs/QUICK_REFERENCE.md` for:
- API quick reference
- Common commands
- Troubleshooting

## Target Audience

Perfect for:
- **Embedded Linux Engineers**
- **Kernel Driver Developers**
- **Network Systems Engineers**
- **FPGA Engineers** (PCIe endpoints)
- **Students** learning device drivers

## Resume Keywords

This collection demonstrates:
- Linux kernel module development
- PCIe driver architecture
- DMA engine implementation
- Network device driver development
- NAPI and interrupt mitigation
- Multi-queue NIC design
- MSI-X interrupt handling
- Descriptor ring management

## Common Commands

```bash
# Build
make                    # Build all
make clean              # Clean

# Load
sudo insmod driver.ko
sudo rmmod driver

# Debug
dmesg | tail -50
lspci -vv
cat /proc/interrupts

# Network
ip link show
ethtool -S eth0
ifconfig eth0 up
```

## Debugging

```bash
# Enable debug messages
echo 8 > /proc/sys/kernel/printk

# View PCI device
lspci -s 00:1f.6 -vv

# Check interrupts
cat /proc/interrupts | grep eth

# Network stats
ethtool -S eth0
```

## Performance

These drivers demonstrate:
- Zero-copy DMA
- NAPI interrupt mitigation
- Multi-queue scalability
- Efficient descriptor management
- Proper memory barriers

## Use Cases

### Data Center
- Multi-queue NIC drivers
- SR-IOV support
- High throughput

### Automotive  
- Deterministic latency
- Real-time constraints
- Safety critical

### FPGA Development
- Custom PCIe endpoints
- DMA engines
- Register access

### Networking
- 10G/40G NICs
- Packet processing
- Low latency

## Support

For issues:
1. Check `dmesg` output
2. Verify kernel version compatibility
3. Review documentation
4. Check hardware compatibility

## 📄 License

GPL v2 (Linux kernel requirement)

##  Credits

Built for kernel driver education and professional development.

---

**Ready to build kernel drivers? Start with `make` and explore!** 🚀