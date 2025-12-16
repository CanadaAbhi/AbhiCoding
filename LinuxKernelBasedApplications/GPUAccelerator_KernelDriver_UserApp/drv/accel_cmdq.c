// accel_cmdq.c -- host-visible ring of job descriptors. accel_cmdq_push()
// writes a descriptor into the ring (what would be a normal memory write to
// a DMA-visible ring on real hardware) and then "rings the doorbell" by
// calling into the simulator/MMIO push function -- the same head/tail/depth
// bookkeeping a PCIe accelerator's ring would use, just software-backed.
#include "accel_priv.h"

extern int accel_hw_sim_push_job(struct accel_hw_sim *hw, u64 tag, u32 opcode, s32 scalar,
				  void *in_kv, u64 in_len, void *out_kv, u64 out_len);

void accel_cmdq_init(struct accel_dev *dev)
{
	spin_lock_init(&dev->cmdq.lock);
	init_waitqueue_head(&dev->cmdq.space_wq);
	dev->cmdq.head = dev->cmdq.tail = 0;
}

static bool cmdq_has_space(struct accel_cmdq *q)
{
	return ((q->tail + 1) % ACCEL_CMDQ_DEPTH) != q->head;
}

u64 accel_cmdq_depth(struct accel_dev *dev)
{
	struct accel_cmdq *q = &dev->cmdq;
	u64 d;
	spin_lock(&q->lock);
	d = (q->tail - q->head + ACCEL_CMDQ_DEPTH) % ACCEL_CMDQ_DEPTH;
	spin_unlock(&q->lock);
	return d;
}

/* consumer side: called from the IRQ path once hw_sim reports a tag done */
static void cmdq_pop(struct accel_dev *dev)
{
	struct accel_cmdq *q = &dev->cmdq;
	spin_lock(&q->lock);
	if (q->head != q->tail)
		q->head = (q->head + 1) % ACCEL_CMDQ_DEPTH;
	spin_unlock(&q->lock);
	wake_up(&q->space_wq);
}
void accel_cmdq_pop(struct accel_dev *dev) { cmdq_pop(dev); }

int accel_cmdq_push(struct accel_dev *dev, struct accel_job *job)
{
	struct accel_cmdq *q = &dev->cmdq;
	struct accel_cmdq_desc *d;
	int ret;

	ret = wait_event_interruptible(q->space_wq, cmdq_has_space(q));
	if (ret) return ret;

	spin_lock(&q->lock);
	d = &q->ring[q->tail];
	d->tag = job->seqno;
	d->opcode = job->opcode;
	d->scalar = job->scalar;
	d->in_dma = job->inbuf->dma_addr + job->in_off;
	d->out_dma = job->outbuf->dma_addr + job->out_off;
	d->in_len = job->in_len;
	d->out_len = job->out_len;
	q->tail = (q->tail + 1) % ACCEL_CMDQ_DEPTH;
	if (accel_cmdq_depth(dev) > q->depth_max)
		q->depth_max = accel_cmdq_depth(dev);
	spin_unlock(&q->lock);

	/* "doorbell write": kick the simulator with the descriptor contents */
	return accel_hw_sim_push_job(dev->hw, d->tag, d->opcode, d->scalar,
				     job->inbuf->cpu_addr + job->in_off, d->in_len,
				     job->outbuf->cpu_addr + job->out_off, d->out_len);
}
