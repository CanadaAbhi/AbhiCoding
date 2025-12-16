# Makefile for PCIe and NIC Drivers Collection

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Common flags
ccflags-y := -I$(PWD)/common

# PCIe Drivers
obj-m += pcie_enum_driver.o
obj-m += pcie_bar_mmio_driver.o
obj-m += pcie_interrupt_driver.o
obj-m += pcie_dma_driver.o

# NIC Drivers
obj-m += nic_basic_driver.o
obj-m += nic_napi_driver.o
obj-m += nic_multiqueue_driver.o

# Source files for each driver
pcie_enum_driver-objs := pcie-drivers/01_pcie_enum_driver.o
pcie_bar_mmio_driver-objs := pcie-drivers/02_pcie_bar_mmio_driver.o
pcie_interrupt_driver-objs := pcie-drivers/03_pcie_interrupt_driver.o
pcie_dma_driver-objs := pcie-drivers/04_pcie_dma_driver.o

nic_basic_driver-objs := nic-drivers/01_nic_basic_driver.o
nic_napi_driver-objs := nic-drivers/02_nic_napi_driver.o
nic_multiqueue_driver-objs := nic-drivers/03_nic_multiqueue_driver.o

PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.ko *.o *.mod.* *.symvers *.order

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

# Load specific driver
load-enum:
	sudo insmod pcie_enum_driver.ko
	dmesg | tail -20

load-bar:
	sudo insmod pcie_bar_mmio_driver.ko
	dmesg | tail -20

load-irq:
	sudo insmod pcie_interrupt_driver.ko
	dmesg | tail -20

load-dma:
	sudo insmod pcie_dma_driver.ko
	dmesg | tail -20

load-nic-basic:
	sudo insmod nic_basic_driver.ko
	dmesg | tail -20

load-nic-napi:
	sudo insmod nic_napi_driver.ko
	dmesg | tail -20

load-nic-mq:
	sudo insmod nic_multiqueue_driver.ko
	dmesg | tail -20

# Unload all drivers
unload:
	-sudo rmmod nic_multiqueue_driver 2>/dev/null
	-sudo rmmod nic_napi_driver 2>/dev/null
	-sudo rmmod nic_basic_driver 2>/dev/null
	-sudo rmmod pcie_dma_driver 2>/dev/null
	-sudo rmmod pcie_interrupt_driver 2>/dev/null
	-sudo rmmod pcie_bar_mmio_driver 2>/dev/null
	-sudo rmmod pcie_enum_driver 2>/dev/null

# Show loaded drivers
show:
	lsmod | grep -E "(pcie|nic)_"

# Build and test
test: all
	@echo "Drivers built successfully"
	@echo "Available drivers:"
	@ls -1 *.ko 2>/dev/null || echo "No drivers built"

help:
	@echo "PCIe and NIC Drivers Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all drivers"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install drivers to system"
	@echo "  load-*       - Load specific driver"
	@echo "  unload       - Unload all drivers"
	@echo "  show         - Show loaded drivers"
	@echo "  test         - Build and verify"
	@echo ""
	@echo "Available drivers:"
	@echo "  - pcie_enum_driver        (PCIe Enumeration)"
	@echo "  - pcie_bar_mmio_driver    (BAR MMIO)"
	@echo "  - pcie_interrupt_driver   (INTx/MSI/MSI-X)"
	@echo "  - pcie_dma_driver         (DMA with SG)"
	@echo "  - nic_basic_driver        (Basic NIC)"
	@echo "  - nic_napi_driver         (NAPI Polling)"
	@echo "  - nic_multiqueue_driver   (Multi-queue NIC)"

.PHONY: all clean install test help show unload