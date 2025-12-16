// accel_irq.c -- threaded IRQ registered against the simulator's virq. Top
// half wakes the thread; thread reads the "status register" (last completed
// tag + error) via accel_hw_sim_read_status(), retires that job's command
// queue slot, signals its fence, and acks the status register -- the same
// ack-then-signal-then-wake sequence used in fake_imu_drv.c and pcie_dma_drv.c.
#include "accel_priv.h"
#include <linux/interrupt.h>

extern int accel_hw_sim_read_status(struct accel_hw_sim *hw, u64 *tag, int *error);
extern void accel_hw_sim_ack(struct accel_hw_sim *hw);
extern void accel_cmdq_pop(struct accel_dev *dev);

static irqreturn_t accel_irq_thread(int irq, void *data)
{
	struct accel_dev *dev = data;
	u64 tag;
	int error;

	if (accel_hw_sim_read_status(dev->hw, &tag, &error) == 0) {
		accel_cmdq_pop(dev);
		accel_fence_signal_seqno(dev, tag, error);
		accel_hw_sim_ack(dev->hw);
	}
	return IRQ_HANDLED;
}

int accel_irq_init(struct accel_dev *dev)
{
	extern int accel_hw_sim_get_virq(struct accel_hw_sim *hw);
	dev->irq = accel_hw_sim_get_virq(dev->hw);
	return request_threaded_irq(dev->irq, NULL, accel_irq_thread,
				     IRQF_ONESHOT, "accel_drv", dev);
}

void accel_irq_fini(struct accel_dev *dev)
{
	free_irq(dev->irq, dev);
}
