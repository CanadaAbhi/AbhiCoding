// accel_hw_sim.c -- software model of the accelerator's hardware. Registers a
// platform_device ("accel-sim") that accel_drv.ko probes against, exposes a
// fake IRQ via irq_domain + generic_handle_irq() (same trick as hw_event_gen.ko
// in irq_defer_lab and fake_imu_hw_sim.c's gpiochip harness), and actually
// performs the requested compute (memcpy/vecadd/scale/checksum) on the DMA
// buffer bytes so results are real, not faked.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/irqdomain.h>
#include <linux/irq.h>
#include <linux/workqueue.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/io.h>
#include "../drv/accel_priv.h" /* for enum accel_opcode etc via accel_uapi.h */

struct sim_job {
	struct list_head node;
	u64 tag;
	u32 opcode;
	s32 scalar;
	void *in_kv, *out_kv; /* kernel virtual addrs of the DMA buffers */
	u64 in_len, out_len;
	struct work_struct work;
};

struct accel_hw_sim {
	struct irq_domain *domain;
	int virq;
	struct workqueue_struct *wq;
	spinlock_t status_lock;
	u64 last_completed_tag;
	int last_error;
	bool irq_pending;
};

static struct platform_device *g_pdev;

/* ---- fake IRQ domain plumbing (mirrors hw_event_gen.ko) ---- */
static int sim_irq_map(struct irq_domain *d, unsigned int irq, irq_hw_number_t hw)
{
	irq_set_chip_and_handler(irq, &dummy_irq_chip, handle_simple_irq);
	return 0;
}
static const struct irq_domain_ops sim_domain_ops = { .map = sim_irq_map };

/* ---- the actual "compute engine" ---- */
static void sim_do_work(struct work_struct *w)
{
	struct sim_job *j = container_of(w, struct sim_job, work);
	struct accel_hw_sim *hw = platform_get_drvdata(g_pdev);
	u8 *in = j->in_kv, *out = j->out_kv;
	int error = 0;

	/* simulate compute latency proportional to data size, like real HW */
	udelay(min_t(u64, j->in_len / 4, 5000));

	switch (j->opcode) {
	case ACCEL_OP_MEMCPY:
		memcpy(out, in, min(j->in_len, j->out_len));
		break;
	case ACCEL_OP_VEC_ADD_SCAL:
		for (u64 i = 0; i < min(j->in_len, j->out_len); i++)
			out[i] = (u8)(in[i] + j->scalar);
		break;
	case ACCEL_OP_VEC_SCALE:
		for (u64 i = 0; i < min(j->in_len, j->out_len); i++)
			out[i] = (u8)(in[i] * j->scalar);
		break;
	case ACCEL_OP_CHECKSUM: {
		u64 sum = 0;
		for (u64 i = 0; i < j->in_len; i++) sum += in[i];
		if (j->out_len >= 8) memcpy(out, &sum, 8);
		break;
	}
	default:
		error = -EINVAL;
	}

	spin_lock(&hw->status_lock);
	hw->last_completed_tag = j->tag;
	hw->last_error = error;
	hw->irq_pending = true;
	spin_unlock(&hw->status_lock);

	/* fire the "completion interrupt" -- exactly what real hardware does
	 * after a DMA engine finishes and writes its status register */
	generic_handle_irq(hw->virq);
	kfree(j);
}

/* ---- exported "MMIO" API used by accel_drv.c ---- */
int accel_hw_sim_push_job(struct accel_hw_sim *hw, u64 tag, u32 opcode, s32 scalar,
			   void *in_kv, u64 in_len, void *out_kv, u64 out_len)
{
	struct sim_job *j = kzalloc(sizeof(*j), GFP_ATOMIC);
	if (!j)
		return -ENOMEM;
	j->tag = tag; j->opcode = opcode; j->scalar = scalar;
	j->in_kv = in_kv; j->in_len = in_len; j->out_kv = out_kv; j->out_len = out_len;
	INIT_WORK(&j->work, sim_do_work);
	queue_work(hw->wq, &j->work); /* == "doorbell write triggers DMA fetch" */
	return 0;
}
EXPORT_SYMBOL_GPL(accel_hw_sim_push_job);

int accel_hw_sim_read_status(struct accel_hw_sim *hw, u64 *tag, int *error)
{
	int pending;
	spin_lock(&hw->status_lock);
	pending = hw->irq_pending;
	*tag = hw->last_completed_tag;
	*error = hw->last_error;
	spin_unlock(&hw->status_lock);
	return pending ? 0 : -ENODATA;
}
EXPORT_SYMBOL_GPL(accel_hw_sim_read_status);

void accel_hw_sim_ack(struct accel_hw_sim *hw)
{
	spin_lock(&hw->status_lock);
	hw->irq_pending = false;
	spin_unlock(&hw->status_lock);
}
EXPORT_SYMBOL_GPL(accel_hw_sim_ack);

int accel_hw_sim_get_virq(struct accel_hw_sim *hw) { return hw->virq; }
EXPORT_SYMBOL_GPL(accel_hw_sim_get_virq);

static int __init accel_hw_sim_init(void)
{
	struct accel_hw_sim *hw = kzalloc(sizeof(*hw), GFP_KERNEL);
	if (!hw) return -ENOMEM;

	spin_lock_init(&hw->status_lock);
	hw->wq = alloc_workqueue("accel_hw_sim", WQ_UNBOUND, 4);

	hw->domain = irq_domain_add_linear(NULL, 1, &sim_domain_ops, NULL);
	hw->virq = irq_create_mapping(hw->domain, 0);

	g_pdev = platform_device_register_simple("accel-sim", -1, NULL, 0);
	platform_set_drvdata(g_pdev, hw);
	dev_set_platform_data(&g_pdev->dev, hw); /* driver reads this to get *hw */

	pr_info("accel_hw_sim: registered accel-sim, virq=%d\n", hw->virq);
	return 0;
}

static void __exit accel_hw_sim_exit(void)
{
	struct accel_hw_sim *hw = platform_get_drvdata(g_pdev);
	destroy_workqueue(hw->wq);
	irq_dispose_mapping(hw->virq);
	irq_domain_remove(hw->domain);
	platform_device_unregister(g_pdev);
	kfree(hw);
}

module_init(accel_hw_sim_init);
module_exit(accel_hw_sim_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Software simulator for accel_drv accelerator");
