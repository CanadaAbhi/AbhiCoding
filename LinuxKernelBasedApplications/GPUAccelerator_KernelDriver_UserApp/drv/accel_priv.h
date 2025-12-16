#ifndef ACCEL_PRIV_H
#define ACCEL_PRIV_H
#include <linux/idr.h>
#include <linux/kthread.h>
#include <linux/dma-fence.h>
#include <linux/dma-mapping.h>
#include <linux/miscdevice.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/kref.h>
#include "../include/accel_uapi.h"

#define ACCEL_CMDQ_DEPTH 64

struct accel_buffer {
	u32 handle;
	size_t size;
	dma_addr_t dma_addr;
	void *cpu_addr;
	struct kref refcount;
	struct accel_dev *dev;
};

struct accel_job {
	struct list_head node;
	u64 seqno;
	u32 opcode;
	s32 scalar;
	u32 priority;
	struct accel_buffer *inbuf, *outbuf;
	u64 in_off, in_len, out_off, out_len;
	struct dma_fence *fence;
	ktime_t submit_ts;
	struct accel_dev *dev;
};

/* one descriptor slot as it would live in a real MMIO-visible ring */
struct accel_cmdq_desc {
	u64 tag;      /* == job seqno, echoed back on completion */
	u32 opcode;
	s32 scalar;
	dma_addr_t in_dma, out_dma;
	u64 in_len, out_len;
};

struct accel_cmdq {
	struct accel_cmdq_desc ring[ACCEL_CMDQ_DEPTH];
	u32 head, tail;      /* producer/consumer indices */
	u64 depth_max;
	spinlock_t lock;
	wait_queue_head_t space_wq;
};

struct accel_dev {
	struct platform_device *pdev;
	struct miscdevice miscdev;

	/* Buffer Manager */
	struct idr buf_idr;
	spinlock_t buf_lock;

	/* Command Queue */
	struct accel_cmdq cmdq;

	/* Scheduler */
	struct task_struct *sched_thread;
	struct list_head pending[3]; /* indexed by accel_priority */
	spinlock_t sched_lock;
	wait_queue_head_t sched_wq;
	bool stop;

	/* Fence */
	u64 fence_context;
	atomic64_t seqno_counter;
	struct idr inflight_idr;     /* seqno -> struct accel_job*, for IRQ lookup */
	spinlock_t inflight_lock;

	/* Interrupt */
	int irq;
	struct accel_hw_sim *hw;     /* opaque handle into the simulator */

	/* Stats */
	struct accel_stats stats;
	spinlock_t stats_lock;
};

/* accel_bufmgr.c */
struct accel_buffer *accel_buf_alloc(struct accel_dev *dev, u64 size, u32 *out_handle);
void accel_buf_put(struct accel_buffer *buf);
struct accel_buffer *accel_buf_get_by_handle(struct accel_dev *dev, u32 handle);
int accel_buf_free_handle(struct accel_dev *dev, u32 handle);
int accel_buf_mmap(struct accel_dev *dev, struct vm_area_struct *vma);

/* accel_cmdq.c */
void accel_cmdq_init(struct accel_dev *dev);
int accel_cmdq_push(struct accel_dev *dev, struct accel_job *job); /* blocks if full */
u64 accel_cmdq_depth(struct accel_dev *dev);

/* accel_sched.c */
int accel_sched_init(struct accel_dev *dev);
void accel_sched_fini(struct accel_dev *dev);
u64 accel_sched_submit(struct accel_dev *dev, struct accel_job *job);

/* accel_fence.c */
int accel_fence_init(struct accel_dev *dev);
struct dma_fence *accel_fence_create(struct accel_dev *dev, u64 *out_seqno);
void accel_fence_signal_seqno(struct accel_dev *dev, u64 seqno, int error);
int accel_fence_wait(struct accel_dev *dev, u64 seqno, u64 timeout_ns);

/* accel_irq.c */
int accel_irq_init(struct accel_dev *dev);
void accel_irq_fini(struct accel_dev *dev);

#endif
