// irq_defer_lab.c -- services the fake hw_event_gen IRQ line via one of three
// selectable deferred-work mechanisms (module param defer_mode), instrumenting
// every stage so workqueue vs tasklet vs threaded-IRQ can be A/B compared with
// identical hardware timing.
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <linux/kfifo.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/poll.h>
#include <linux/uaccess.h>
#include <linux/debugfs.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include "hw_event_gen.h"
#include "irq_defer_uapi.h"

static unsigned int defer_mode = IRQ_DEFER_MODE_WORKQUEUE;
module_param(defer_mode, uint, 0444);
MODULE_PARM_DESC(defer_mode, "0=workqueue 1=tasklet 2=threaded_irq (see irq_defer_uapi.h)");

static unsigned int busy_us = 200;
module_param(busy_us, uint, 0644);
MODULE_PARM_DESC(busy_us, "simulated heavy-processing duration in microseconds");

#define SCRATCH_LEN   4096
#define FIFO_DEPTH    256

struct irq_defer_state {
	int irq;
	u8 *scratch;

	/* per-event in-flight timestamps (single-IRQ-line lab, so one slot is fine;
	 * a multi-queue design would index this by a per-job id instead) */
	ktime_t ts_hw;
	ktime_t ts_top;

	struct work_struct  work;      /* MODE_WORKQUEUE and MODE_THREADED_IRQ's 2nd hop */
	struct tasklet_struct tasklet; /* MODE_TASKLET */

	struct kfifo fifo;
	wait_queue_head_t readq;
	spinlock_t fifo_lock;
	u64 seq;

	/* running stats, exposed via debugfs */
	atomic64_t count;
	atomic64_t sum_total_ns, min_total_ns, max_total_ns;

	struct cdev cdev;
	dev_t devt;
	struct class *class;
};

static struct irq_defer_state *st;

/* ================= shared "heavy processing" simulation ================= */
/* Represents e.g. checksum/decode work a real driver would do off the fast
 * path: touches a scratch buffer for busy_us worth of time. Safe to call from
 * process context (workqueue/thread) OR softirq context (tasklet), but doing
 * this in softirq context for anything beyond a few hundred us is exactly the
 * anti-pattern this lab is built to demonstrate. */
static void heavy_processing(void)
{
	ktime_t end = ktime_add_us(ktime_get(), busy_us);
	unsigned i;
	u8 acc = 0;

	while (ktime_before(ktime_get(), end)) {
		for (i = 0; i < SCRATCH_LEN; i++)
			acc += ++st->scratch[i];
	}
}

static void push_record(ktime_t ts_dispatch, ktime_t ts_done)
{
	struct irq_defer_record rec;
	u64 total_ns = ktime_to_ns(ktime_sub(ts_done, st->ts_hw));

	rec.seq = st->seq++;
	rec.mode = defer_mode;
	rec.dispatch_ns = ktime_to_ns(ktime_sub(ts_dispatch, st->ts_top));
	rec.proc_ns = ktime_to_ns(ktime_sub(ts_done, ts_dispatch));
	rec.total_ns = total_ns;

	spin_lock(&st->fifo_lock);
	if (kfifo_avail(&st->fifo) < sizeof(rec)) {
		struct irq_defer_record discard;
		kfifo_out(&st->fifo, &discard, sizeof(discard));
	}
	kfifo_in(&st->fifo, &rec, sizeof(rec));
	spin_unlock(&st->fifo_lock);

	atomic64_inc(&st->count);
	atomic64_add(total_ns, &st->sum_total_ns);
	if (total_ns < atomic64_read(&st->min_total_ns) || atomic64_read(&st->count) == 1)
		atomic64_set(&st->min_total_ns, total_ns);
	if (total_ns > atomic64_read(&st->max_total_ns))
		atomic64_set(&st->max_total_ns, total_ns);

	wake_up_interruptible(&st->readq);
}

/* ================= MODE_WORKQUEUE: hardirq -> workqueue -> heavy work ===== */
static void wq_deferred_fn(struct work_struct *w)
{
	ktime_t ts_dispatch = ktime_get();
	heavy_processing();                 /* workqueue context: fine to sleep too */
	push_record(ts_dispatch, ktime_get());
}

static irqreturn_t irq_top_half_wq(int irq, void *dev_id)
{
	st->ts_hw = hweg_last_event_ts();
	st->ts_top = ktime_get();
	hweg_ack();                          /* "acknowledge hardware" */
	queue_work(system_highpri_wq, &st->work);
	return IRQ_HANDLED;
}

/* ================= MODE_TASKLET: hardirq -> tasklet(softirq) -> heavy work */
static void tasklet_deferred_fn(struct tasklet_struct *t)
{
	ktime_t ts_dispatch = ktime_get();
	heavy_processing();                 /* softirq context: MUST NOT sleep, bounded! */
	push_record(ts_dispatch, ktime_get());
}

static irqreturn_t irq_top_half_tasklet(int irq, void *dev_id)
{
	st->ts_hw = hweg_last_event_ts();
	st->ts_top = ktime_get();
	hweg_ack();
	tasklet_schedule(&st->tasklet);
	return IRQ_HANDLED;
}

/* ================= MODE_THREADED_IRQ: hardirq(primary) -> irq thread ->
 * workqueue -> heavy work. The thread itself stays cheap so it's immediately
 * available for the next interrupt; genuinely heavy/variable-length work is
 * still pushed to a workqueue, matching the requested pipeline diagram. */
static irqreturn_t irq_primary_threaded(int irq, void *dev_id)
{
	st->ts_hw = hweg_last_event_ts();
	st->ts_top = ktime_get();
	hweg_ack();
	return IRQ_WAKE_THREAD;
}

static irqreturn_t irq_thread_fn(int irq, void *dev_id)
{
	/* minimal thread-side dispatch latency; hand heavy lifting to workqueue */
	queue_work(system_highpri_wq, &st->work);
	return IRQ_HANDLED;
}
/* wq_deferred_fn (already defined above) serves as the heavy-processing hop
 * for this mode too -- same function, reused. */

/* ================= chardev: read/poll/ioctl ================= */
static ssize_t irq_defer_read(struct file *f, char __user *buf, size_t count, loff_t *pos)
{
	struct irq_defer_record rec;
	int ret;

	if (count < sizeof(rec))
		return -EINVAL;
	if (kfifo_is_empty(&st->fifo)) {
		if (f->f_flags & O_NONBLOCK)
			return -EAGAIN;
		ret = wait_event_interruptible(st->readq, !kfifo_is_empty(&st->fifo));
		if (ret)
			return ret;
	}
	spin_lock(&st->fifo_lock);
	ret = kfifo_out(&st->fifo, &rec, sizeof(rec));
	spin_unlock(&st->fifo_lock);
	if (ret != sizeof(rec))
		return -EIO;
	return copy_to_user(buf, &rec, sizeof(rec)) ? -EFAULT : sizeof(rec);
}

static __poll_t irq_defer_poll(struct file *f, poll_table *wait)
{
	poll_wait(f, &st->readq, wait);
	return kfifo_is_empty(&st->fifo) ? 0 : (EPOLLIN | EPOLLRDNORM);
}

static long irq_defer_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	u32 mode;
	switch (cmd) {
	case IRQ_DEFER_IOC_GET_MODE:
		mode = defer_mode;
		return copy_to_user((void __user *)arg, &mode, sizeof(mode)) ? -EFAULT : 0;
	case IRQ_DEFER_IOC_RESET_STATS:
		atomic64_set(&st->count, 0);
		atomic64_set(&st->sum_total_ns, 0);
		atomic64_set(&st->min_total_ns, 0);
		atomic64_set(&st->max_total_ns, 0);
		return 0;
	default:
		return -ENOTTY;
	}
}

static const struct file_operations irq_defer_fops = {
	.owner = THIS_MODULE,
	.read = irq_defer_read,
	.poll = irq_defer_poll,
	.unlocked_ioctl = irq_defer_ioctl,
};

/* ================= debugfs quick stats ================= */
static int stats_show(struct seq_file *s, void *unused)
{
	u64 n = atomic64_read(&st->count);
	static const char *names[] = { "workqueue", "tasklet", "threaded_irq" };

	seq_printf(s, "mode        : %s\n", names[defer_mode]);
	seq_printf(s, "busy_us     : %u\n", busy_us);
	seq_printf(s, "count       : %llu\n", n);
	if (n) {
		seq_printf(s, "avg_total_ns: %llu\n", atomic64_read(&st->sum_total_ns) / n);
		seq_printf(s, "min_total_ns: %llu\n", atomic64_read(&st->min_total_ns));
		seq_printf(s, "max_total_ns: %llu\n", atomic64_read(&st->max_total_ns));
	}
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(stats);
static struct dentry *debugfs_root;

/* ================= module init/exit ================= */
static int __init irq_defer_lab_init(void)
{
	int ret;

	st = kzalloc(sizeof(*st), GFP_KERNEL);
	if (!st)
		return -ENOMEM;
	st->scratch = kzalloc(SCRATCH_LEN, GFP_KERNEL);
	if (!st->scratch) {
		ret = -ENOMEM;
		goto err_st;
	}

	spin_lock_init(&st->fifo_lock);
	init_waitqueue_head(&st->readq);
	ret = kfifo_alloc(&st->fifo, FIFO_DEPTH * sizeof(struct irq_defer_record), GFP_KERNEL);
	if (ret)
		goto err_scratch;
	INIT_WORK(&st->work, wq_deferred_fn);
	tasklet_setup(&st->tasklet, tasklet_deferred_fn);
	atomic64_set(&st->min_total_ns, 0);

	st->irq = hweg_get_irq();
	if (st->irq < 0) {
		pr_err("irq_defer_lab: hw_event_gen not loaded?\n");
		ret = st->irq;
		goto err_fifo;
	}

	switch (defer_mode) {
	case IRQ_DEFER_MODE_WORKQUEUE:
		ret = request_irq(st->irq, irq_top_half_wq, 0, "irq_defer_lab", NULL);
		break;
	case IRQ_DEFER_MODE_TASKLET:
		ret = request_irq(st->irq, irq_top_half_tasklet, 0, "irq_defer_lab", NULL);
		break;
	case IRQ_DEFER_MODE_THREADED_IRQ:
		ret = request_threaded_irq(st->irq, irq_primary_threaded, irq_thread_fn,
					    IRQF_ONESHOT, "irq_defer_lab", NULL);
		break;
	default:
		ret = -EINVAL;
	}
	if (ret)
		goto err_fifo;

	/* chardev */
	st->class = class_create("irq_defer_lab");
	ret = alloc_chrdev_region(&st->devt, 0, 1, "irq_defer_lab");
	if (ret)
		goto err_irq;
	cdev_init(&st->cdev, &irq_defer_fops);
	ret = cdev_add(&st->cdev, st->devt, 1);
	if (ret)
		goto err_chrdev;
	device_create(st->class, NULL, st->devt, NULL, "irq_defer_lab0");

	debugfs_root = debugfs_create_dir("irq_defer_lab", NULL);
	debugfs_create_file("stats", 0444, debugfs_root, NULL, &stats_fops);

	pr_info("irq_defer_lab: mode=%u irq=%d busy_us=%u\n", defer_mode, st->irq, busy_us);
	return 0;

err_chrdev:
	unregister_chrdev_region(st->devt, 1);
err_irq:
	free_irq(st->irq, NULL);
err_fifo:
	kfifo_free(&st->fifo);
err_scratch:
	kfree(st->scratch);
err_st:
	kfree(st);
	return ret;
}

static void __exit irq_defer_lab_exit(void)
{
	debugfs_remove_recursive(debugfs_root);
	device_destroy(st->class, st->devt);
	cdev_del(&st->cdev);
	unregister_chrdev_region(st->devt, 1);
	class_destroy(st->class);
	free_irq(st->irq, NULL);
	tasklet_kill(&st->tasklet);
	cancel_work_sync(&st->work);
	kfifo_free(&st->fifo);
	kfree(st->scratch);
	kfree(st);
}
module_init(irq_defer_lab_init);
module_exit(irq_defer_lab_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("IRQ top-half + deferred-work (workqueue/tasklet/threaded-irq) comparison lab");
