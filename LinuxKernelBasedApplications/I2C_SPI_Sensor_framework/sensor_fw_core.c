// sensor_fw_core.c -- shared class/chardev/sysfs/kfifo plumbing for
// heterogeneous bus-attached sensors (I2C temp, SPI IMU, ...). Bus drivers
// only implement register-level I/O + IRQ handling; everything a userspace
// daemon touches (open/read/poll/ioctl/sysfs) is uniform across sensor types.
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/idr.h>
#include "sensor_fw_core.h"

#define SENSOR_FW_MAX_DEVICES 16

static struct class *sensor_fw_class;
static dev_t sensor_fw_base_devt;
static DEFINE_IDA(sensor_fw_ida);

/* ================= chardev fops ================= */
static int sensor_fw_open(struct inode *inode, struct file *filp)
{
	struct sensor_fw_dev *sfd = container_of(inode->i_cdev, struct sensor_fw_dev, cdev);
	filp->private_data = sfd;
	return 0;
}

static ssize_t sensor_fw_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos)
{
	struct sensor_fw_dev *sfd = filp->private_data;
	struct sensor_fw_sample sample;
	int ret;

	if (count < sizeof(sample))
		return -EINVAL;

	if (kfifo_is_empty(&sfd->fifo)) {
		if (filp->f_flags & O_NONBLOCK)
			return -EAGAIN;
		ret = wait_event_interruptible(sfd->wq, !kfifo_is_empty(&sfd->fifo));
		if (ret)
			return ret;
	}

	mutex_lock(&sfd->lock);
	ret = kfifo_out(&sfd->fifo, &sample, sizeof(sample));
	mutex_unlock(&sfd->lock);
	if (ret != sizeof(sample))
		return -EIO;

	if (copy_to_user(buf, &sample, sizeof(sample)))
		return -EFAULT;
	return sizeof(sample);
}

static __poll_t sensor_fw_poll(struct file *filp, poll_table *wait)
{
	struct sensor_fw_dev *sfd = filp->private_data;
	poll_wait(filp, &sfd->wq, wait);
	return kfifo_is_empty(&sfd->fifo) ? 0 : (EPOLLIN | EPOLLRDNORM);
}

static long sensor_fw_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct sensor_fw_dev *sfd = filp->private_data;
	struct sensor_fw_info info;
	u32 hz;

	switch (cmd) {
	case SENSOR_FW_IOC_GET_INFO:
		memset(&info, 0, sizeof(info));
		info.type = sfd->type;
		info.num_channels = sfd->num_channels;
		info.scale_milli = sfd->scale_milli;
		strscpy(info.name, sfd->name, sizeof(info.name));
		return copy_to_user((void __user *)arg, &info, sizeof(info)) ? -EFAULT : 0;

	case SENSOR_FW_IOC_SET_ODR:
		if (!sfd->set_odr)
			return -ENOTSUPP;
		if (copy_from_user(&hz, (void __user *)arg, sizeof(hz)))
			return -EFAULT;
		return sfd->set_odr(sfd, hz);

	default:
		return -ENOTTY;
	}
}

static const struct file_operations sensor_fw_fops = {
	.owner = THIS_MODULE,
	.open = sensor_fw_open,
	.read = sensor_fw_read,
	.poll = sensor_fw_poll,
	.unlocked_ioctl = sensor_fw_ioctl,
};

/* ================= sysfs ================= */
static ssize_t name_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct sensor_fw_dev *sfd = dev_get_drvdata(dev);
	return sysfs_emit(buf, "%s\n", sfd->name);
}
static DEVICE_ATTR_RO(name);

static ssize_t type_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct sensor_fw_dev *sfd = dev_get_drvdata(dev);
	return sysfs_emit(buf, "%u\n", sfd->type);
}
static DEVICE_ATTR_RO(type);

static ssize_t sample_count_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct sensor_fw_dev *sfd = dev_get_drvdata(dev);
	return sysfs_emit(buf, "%llu\n", (unsigned long long)atomic64_read(&sfd->sample_count));
}
static DEVICE_ATTR_RO(sample_count);

static ssize_t overrun_count_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct sensor_fw_dev *sfd = dev_get_drvdata(dev);
	return sysfs_emit(buf, "%d\n", atomic_read(&sfd->overrun_count));
}
static DEVICE_ATTR_RO(overrun_count);

static ssize_t value_latest_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct sensor_fw_dev *sfd = dev_get_drvdata(dev);
	int i, n = 0;

	mutex_lock(&sfd->lock);
	if (!sfd->have_last) {
		mutex_unlock(&sfd->lock);
		return sysfs_emit(buf, "no data\n");
	}
	for (i = 0; i < sfd->num_channels; i++)
		n += sysfs_emit_at(buf, n, "%d%s", sfd->last_sample.chan[i],
				    (i == sfd->num_channels - 1) ? "\n" : " ");
	mutex_unlock(&sfd->lock);
	return n;
}
static DEVICE_ATTR_RO(value_latest);

static struct attribute *sensor_fw_attrs[] = {
	&dev_attr_name.attr,
	&dev_attr_type.attr,
	&dev_attr_sample_count.attr,
	&dev_attr_overrun_count.attr,
	&dev_attr_value_latest.attr,
	NULL,
};
static const struct attribute_group sensor_fw_group = {
	.attrs = sensor_fw_attrs,
};

/* ================= public API ================= */
int sensor_fw_register(struct sensor_fw_dev *sfd)
{
	int ret, minor;

	if (!sfd->name || !sfd->parent || sfd->num_channels > SENSOR_FW_MAX_CHANNELS)
		return -EINVAL;

	minor = ida_alloc_max(&sensor_fw_ida, SENSOR_FW_MAX_DEVICES - 1, GFP_KERNEL);
	if (minor < 0)
		return minor;
	sfd->minor = minor;
	sfd->devt = MKDEV(MAJOR(sensor_fw_base_devt), minor);

	mutex_init(&sfd->lock);
	init_waitqueue_head(&sfd->wq);
	atomic_set(&sfd->overrun_count, 0);
	atomic64_set(&sfd->sample_count, 0);
	sfd->have_last = false;

	ret = kfifo_alloc(&sfd->fifo, SENSOR_FW_FIFO_DEPTH * sizeof(struct sensor_fw_sample),
			   GFP_KERNEL);
	if (ret)
		goto err_ida;

	cdev_init(&sfd->cdev, &sensor_fw_fops);
	ret = cdev_add(&sfd->cdev, sfd->devt, 1);
	if (ret)
		goto err_fifo;

	sfd->dev = device_create(sensor_fw_class, sfd->parent, sfd->devt, sfd, "%s", sfd->name);
	if (IS_ERR(sfd->dev)) {
		ret = PTR_ERR(sfd->dev);
		goto err_cdev;
	}

	ret = sysfs_create_group(&sfd->dev->kobj, &sensor_fw_group);
	if (ret)
		goto err_device;

	dev_info(sfd->parent, "sensor_fw: registered '%s' as /dev/%s (minor %d)\n",
		 sfd->name, sfd->name, minor);
	return 0;

err_device:
	device_destroy(sensor_fw_class, sfd->devt);
err_cdev:
	cdev_del(&sfd->cdev);
err_fifo:
	kfifo_free(&sfd->fifo);
err_ida:
	ida_free(&sensor_fw_ida, minor);
	return ret;
}
EXPORT_SYMBOL_GPL(sensor_fw_register);

void sensor_fw_unregister(struct sensor_fw_dev *sfd)
{
	sysfs_remove_group(&sfd->dev->kobj, &sensor_fw_group);
	device_destroy(sensor_fw_class, sfd->devt);
	cdev_del(&sfd->cdev);
	kfifo_free(&sfd->fifo);
	ida_free(&sensor_fw_ida, sfd->minor);
}
EXPORT_SYMBOL_GPL(sensor_fw_unregister);

/* Called from bus-driver IRQ-thread/workqueue context (may sleep). */
int sensor_fw_push_sample(struct sensor_fw_dev *sfd, const s32 *chan, u32 flags)
{
	struct sensor_fw_sample sample;
	int i, ret;

	sample.ts_ns = ktime_get_ns();
	sample.flags = flags;
	for (i = 0; i < sfd->num_channels; i++)
		sample.chan[i] = chan[i];
	for (; i < SENSOR_FW_MAX_CHANNELS; i++)
		sample.chan[i] = 0;

	mutex_lock(&sfd->lock);
	if (kfifo_avail(&sfd->fifo) < sizeof(sample)) {
		/* drop oldest to make room -- bounded FIFO, never blocks IRQ thread */
		struct sensor_fw_sample discard;
		kfifo_out(&sfd->fifo, &discard, sizeof(discard));
		atomic_inc(&sfd->overrun_count);
		sample.flags |= SENSOR_FW_FLAG_OVERRUN;
	}
	ret = kfifo_in(&sfd->fifo, &sample, sizeof(sample));
	sfd->last_sample = sample;
	sfd->have_last = true;
	mutex_unlock(&sfd->lock);

	atomic64_inc(&sfd->sample_count);
	wake_up_interruptible(&sfd->wq);
	return ret == sizeof(sample) ? 0 : -EIO;
}
EXPORT_SYMBOL_GPL(sensor_fw_push_sample);

/* ================= module init/exit ================= */
static int __init sensor_fw_core_init(void)
{
	int ret;

	sensor_fw_class = class_create("sensor_fw");
	if (IS_ERR(sensor_fw_class))
		return PTR_ERR(sensor_fw_class);

	ret = alloc_chrdev_region(&sensor_fw_base_devt, 0, SENSOR_FW_MAX_DEVICES, "sensor_fw");
	if (ret) {
		class_destroy(sensor_fw_class);
		return ret;
	}
	pr_info("sensor_fw_core: loaded, major=%d\n", MAJOR(sensor_fw_base_devt));
	return 0;
}

static void __exit sensor_fw_core_exit(void)
{
	unregister_chrdev_region(sensor_fw_base_devt, SENSOR_FW_MAX_DEVICES);
	class_destroy(sensor_fw_class);
}
module_init(sensor_fw_core_init);
module_exit(sensor_fw_core_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Sensor framework core: common chardev/sysfs/kfifo ABI for I2C/SPI sensor drivers");
