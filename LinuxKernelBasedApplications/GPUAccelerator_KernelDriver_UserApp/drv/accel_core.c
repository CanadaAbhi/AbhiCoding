// accel_core.c -- platform_driver that binds to "accel-sim" (or, with the
// QEMU/FPGA backend swapped in, a PCI/AXI device exposing the same register
// contract), wires up bufmgr/cmdq/scheduler/fence/irq, and exposes the UAPI
// chardev used by user/app.c.
#include "accel_priv.h"
#include <linux/platform_device.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/module.h>

static int accel_open(struct inode *i, struct file *f) { return 0; }
static int accel_release(struct inode *i, struct file *f) { return 0; }

static int accel_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct accel_dev *dev = container_of(f->private_data ? f->private_data :
					      (void *)f->f_inode->i_cdev, struct accel_dev, miscdev);
	/* simpler: fetch via miscdevice's parent container */
	struct miscdevice *m = f->private_data;
	dev = container_of(m, struct accel_dev, miscdev);
	return accel_buf_mmap(dev, vma);
}

static long accel_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct miscdevice *m = f->private_data;
	struct accel_dev *dev = container_of(m, struct accel_dev, miscdev);
	void __user *uarg = (void __user *)arg;

	switch (cmd) {
	case ACCEL_IOC_BUFFER_ALLOC: {
		struct accel_buffer_alloc a;
		struct accel_buffer *buf;
		if (copy_from_user(&a, uarg, sizeof(a))) return -EFAULT;
		buf = accel_buf_alloc(dev, a.size, &a.handle);
		if (!buf) return -ENOMEM;
		a.mmap_offset = (u64)a.handle * ACCEL_MMAP_UNIT;
		return copy_to_user(uarg, &a, sizeof(a)) ? -EFAULT : 0;
	}
	case ACCEL_IOC_BUFFER_FREE: {
		struct accel_buffer_free a;
		if (copy_from_user(&a, uarg, sizeof(a))) return -EFAULT;
		return accel_buf_free_handle(dev, a.handle);
	}
	case ACCEL_IOC_SUBMIT: {
		struct accel_submit s;
		struct accel_job *job;
		struct accel_buffer *in, *out;

		if (copy_from_user(&s, uarg, sizeof(s))) return -EFAULT;
		in = accel_buf_get_by_handle(dev, s.input_handle);
		out = accel_buf_get_by_handle(dev, s.output_handle);
		if (!in || !out) { if (in) accel_buf_put(in); if (out) accel_buf_put(out); return -EINVAL; }

		job = kzalloc(sizeof(*job), GFP_KERNEL);
		if (!job) { accel_buf_put(in); accel_buf_put(out); return -ENOMEM; }

		job->dev = dev; job->opcode = s.opcode; job->scalar = s.scalar;
		job->priority = s.priority; job->inbuf = in; job->outbuf = out;
		job->in_off = s.input_offset; job->in_len = s.input_len;
		job->out_off = s.output_offset; job->out_len = s.output_len;
		job->submit_ts = ktime_get();
		job->fence = accel_fence_create(dev, &job->seqno);
		if (!job->fence) { kfree(job); accel_buf_put(in); accel_buf_put(out); return -ENOMEM; }

		spin_lock(&dev->inflight_lock);
		idr_alloc(&dev->inflight_idr, job, job->seqno, job->seqno + 1, GFP_ATOMIC);
		spin_unlock(&dev->inflight_lock);

		spin_lock(&dev->stats_lock);
		dev->stats.jobs_submitted++;
		spin_unlock(&dev->stats_lock);

		s.out_seqno = accel_sched_submit(dev, job);
		return copy_to_user(uarg, &s, sizeof(s)) ? -EFAULT : 0;
	}
	case ACCEL_IOC_WAIT: {
		struct accel_wait w;
		int ret;
		if (copy_from_user(&w, uarg, sizeof(w))) return -EFAULT;
		ret = accel_fence_wait(dev, w.seqno, w.timeout_ns);
		w.out_status = (ret == 0) ? 0 : (ret == -ETIME ? 1 : 2);
		return copy_to_user(uarg, &w, sizeof(w)) ? -EFAULT : 0;
	}
	case ACCEL_IOC_GET_STATS: {
		struct accel_stats st;
		spin_lock(&dev->stats_lock);
		st = dev->stats;
		spin_unlock(&dev->stats_lock);
		st.cmdq_depth_cur = accel_cmdq_depth(dev);
		st.cmdq_depth_max = dev->cmdq.depth_max;
		return copy_to_user(uarg, &st, sizeof(st)) ? -EFAULT : 0;
	}
	default:
		return -ENOTTY;
	}
}

static const struct file_operations accel_fops = {
	.owner = THIS_MODULE, .open = accel_open, .release = accel_release,
	.unlocked_ioctl = accel_ioctl, .mmap = accel_mmap,
};

static int accel_probe(struct platform_device *pdev)
{
	struct accel_dev *dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	int ret;

	if (!dev) return -ENOMEM;
	dev->pdev = pdev;
	dev->hw = dev_get_platdata(&pdev->dev);
	idr_init(&dev->buf_idr);
	spin_lock_init(&dev->buf_lock);
	spin_lock_init(&dev->stats_lock);

	accel_cmdq_init(dev);
	accel_fence_init(dev);

	ret = accel_sched_init(dev);
	if (ret) goto err_free;
	ret = accel_irq_init(dev);
	if (ret) goto err_sched;

	dev->miscdev.minor = MISC_DYNAMIC_MINOR;
	dev->miscdev.name = "accel0";
	dev->miscdev.fops = &accel_fops;
	ret = misc_register(&dev->miscdev);
	if (ret) goto err_irq;

	platform_set_drvdata(pdev, dev);
	dev_info(&pdev->dev, "accel_drv: /dev/accel0 ready (irq=%d)\n", dev->irq);
	return 0;

err_irq: accel_irq_fini(dev);
err_sched: accel_sched_fini(dev);
err_free: kfree(dev);
	return ret;
}

static int accel_remove(struct platform_device *pdev)
{
	struct accel_dev *dev = platform_get_drvdata(pdev);
	misc_deregister(&dev->miscdev);
	accel_irq_fini(dev);
	accel_sched_fini(dev);
	idr_destroy(&dev->buf_idr);
	idr_destroy(&dev->inflight_idr);
	kfree(dev);
	return 0;
}

static struct platform_driver accel_pdrv = {
	.probe = accel_probe,
	.remove = accel_remove,
	.driver = { .name = "accel-sim" },
};
module_platform_driver(accel_pdrv);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Simplified accelerator driver: bufmgr+cmdq+scheduler+fence+irq");
