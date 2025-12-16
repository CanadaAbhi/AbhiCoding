// accel_sched.c -- single-engine priority scheduler: a kthread drains three
// FIFO priority lists (HIGH/NORMAL/LOW), always preferring higher priority,
// and pushes the winning job into the command queue. (This is a deliberately
// simplified cousin of the multi-engine, aging/EDF drm_sched-style scheduler
// project -- here scoped to one accelerator engine per this task's diagram.)
#include "accel_priv.h"
#include <linux/kthread.h>

static struct accel_job *sched_pick_next(struct accel_dev *dev)
{
	struct accel_job *job = NULL;
	int prio;

	spin_lock(&dev->sched_lock);
	for (prio = ACCEL_PRIO_HIGH; prio <= ACCEL_PRIO_LOW; prio++) {
		if (!list_empty(&dev->pending[prio])) {
			job = list_first_entry(&dev->pending[prio], struct accel_job, node);
			list_del(&job->node);
			break;
		}
	}
	spin_unlock(&dev->sched_lock);
	return job;
}

static bool sched_has_work(struct accel_dev *dev)
{
	int prio;
	bool any = false;
	spin_lock(&dev->sched_lock);
	for (prio = ACCEL_PRIO_HIGH; prio <= ACCEL_PRIO_LOW; prio++)
		any |= !list_empty(&dev->pending[prio]);
	spin_unlock(&dev->sched_lock);
	return any;
}

static int accel_sched_fn(void *data)
{
	struct accel_dev *dev = data;

	while (!kthread_should_stop()) {
		struct accel_job *job;

		wait_event_interruptible(dev->sched_wq,
			sched_has_work(dev) || kthread_should_stop());
		if (kthread_should_stop()) break;

		job = sched_pick_next(dev);
		if (!job) continue;

		if (accel_cmdq_push(dev, job)) {
			spin_lock(&dev->stats_lock);
			dev->stats.jobs_failed++;
			spin_unlock(&dev->stats_lock);
			accel_fence_signal_seqno(dev, job->seqno, -EIO);
		}
		/* job struct itself stays alive via the inflight_idr until the
		 * IRQ handler retires it; scheduler's list ownership ends here */
	}
	return 0;
}

int accel_sched_init(struct accel_dev *dev)
{
	int i;
	for (i = 0; i < 3; i++) INIT_LIST_HEAD(&dev->pending[i]);
	spin_lock_init(&dev->sched_lock);
	init_waitqueue_head(&dev->sched_wq);
	dev->sched_thread = kthread_run(accel_sched_fn, dev, "accel_sched");
	return PTR_ERR_OR_ZERO(dev->sched_thread);
}

void accel_sched_fini(struct accel_dev *dev)
{
	kthread_stop(dev->sched_thread);
}

u64 accel_sched_submit(struct accel_dev *dev, struct accel_job *job)
{
	spin_lock(&dev->sched_lock);
	list_add_tail(&job->node, &dev->pending[job->priority]);
	spin_unlock(&dev->sched_lock);
	wake_up(&dev->sched_wq);
	return job->seqno;
}
