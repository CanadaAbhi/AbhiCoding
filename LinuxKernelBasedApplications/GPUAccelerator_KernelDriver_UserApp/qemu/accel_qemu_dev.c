// accel_qemu_dev.c -- QOM skeleton for `qemu-system-x86_64 -device accel-test`.
// Vendor 0xfffe (continuing the fake-vendor convention from pcie_dma_drv),
// device 0x0002. BAR0 registers mirror struct accel_cmdq_desc + status/irq
// fields 1:1 so a real MMIO driver variant of accel_core.c is a near drop-in.
#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/irq.h"
#include "qemu/timer.h"

#define REG_DOORBELL   0x00  /* write descriptor index to kick engine   */
#define REG_STATUS     0x04  /* last completed tag (low 32 bits)        */
#define REG_STATUS_HI  0x08  /* last completed tag (high 32 bits)       */
#define REG_IRQ_STATUS 0x0c  /* bit0 = completion pending               */
#define REG_IRQ_MASK   0x10
#define REG_DESC_BASE  0x100 /* ACCEL_CMDQ_DEPTH descriptors follow      */

typedef struct AccelTestState {
	PCIDevice parent_obj;
	MemoryRegion mmio;
	QEMUTimer *compute_timer;
	uint64_t regs[0x40];
	uint8_t *dma_ram; /* mapped via pci_dma_read/write in real impl */
} AccelTestState;

static void accel_mmio_write(void *opaque, hwaddr addr, uint64_t val, unsigned size)
{
	AccelTestState *s = opaque;
	if (addr == REG_DOORBELL) {
		/* schedule a timer to model compute latency, then raise IRQ --
		 * exactly what accel_hw_sim.c's workqueue does in software */
		timer_mod(s->compute_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + 2000);
	} else {
		s->regs[addr / 8] = val;
	}
}

static uint64_t accel_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
	AccelTestState *s = opaque;
	return s->regs[addr / 8];
}

static const MemoryRegionOps accel_mmio_ops = {
	.read = accel_mmio_read, .write = accel_mmio_write,
	.endianness = DEVICE_LITTLE_ENDIAN,
};

static void accel_timer_cb(void *opaque)
{
	AccelTestState *s = opaque;
	s->regs[REG_IRQ_STATUS / 8] |= 1;
	pci_set_irq(PCI_DEVICE(s), 1); /* legacy INTx; MSI-X variant analogous to pcie_dma_drv */
}

static void accel_realize(PCIDevice *pdev, Error **errp)
{
	AccelTestState *s = ACCEL_TEST(pdev);
	memory_region_init_io(&s->mmio, OBJECT(s), &accel_mmio_ops, s, "accel-test-mmio", 0x1000);
	pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
	s->compute_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, accel_timer_cb, s);
}
/* full class_init / TypeInfo boilerplate omitted for brevity -- follows the
 * same pattern as QEMU's real 'edu' device, which pcie_dma_drv already
 * models its BAR0 layout after. */
