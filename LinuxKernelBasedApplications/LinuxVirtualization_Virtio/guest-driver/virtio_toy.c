// virtio_toy.c -- Guest Linux driver for the custom virtio-toy PCI device.
// Mirrors gpu_drv.c's ring-buffer/fence pattern but the "ring buffer" here
// IS the virtio split-ring (desc/avail/used) managed entirely by virtio
// core (virtqueue_add_sgs/virtqueue_kick) instead of a hand-rolled ring --
// this is the real production pattern your gpu_drv.c ring simulates.
#include <linux/module.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/miscdevice.h>
#include <linux/completion.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include "virtio_toy.h"

struct toy_job {
    struct virtio_toy_req req;
    void *in_buf, *out_buf;
    struct completion done;
};

struct virtio_toy_dev {
    struct virtio_device *vdev;
    struct virtqueue *vq;
    struct miscdevice miscdev;
    spinlock_t lock;
};

/* virtqueue completion callback -- fires from the guest's MSI-X interrupt
 * handler (virtio-pci's per-vq ISR calls this via vring_interrupt()).
 * This IS the "interrupt" leg of the pipeline on the guest side: hardware
 * (virtio-pci MMIO ISR status register) -> vring_interrupt() -> here. */
static void toy_vq_callback(struct virtqueue *vq)
{
    struct toy_job *job;
    unsigned int len;

    while ((job = virtqueue_get_buf(vq, &len)) != NULL)
        complete(&job->done);
}

static int toy_submit_and_wait(struct virtio_toy_dev *tdev, struct toy_job *job)
{
    struct scatterlist sg_req, sg_in, sg_out, *sgs[3];
    int ret;

    sg_init_one(&sg_req, &job->req, sizeof(job->req));
    sg_init_one(&sg_in, job->in_buf, job->req.nelem * sizeof(u32));
    sg_init_one(&sg_out, job->out_buf, job->req.nelem * sizeof(u32));
    sgs[0] = &sg_req; sgs[1] = &sg_in; sgs[2] = &sg_out;

    init_completion(&job->done);

    spin_lock(&tdev->lock);
    /* 2 out_sgs (device reads: header+input), 1 in_sg (device writes: output) */
    ret = virtqueue_add_sgs(tdev->vq, sgs, 2, 1, job, GFP_ATOMIC);
    if (ret == 0)
        virtqueue_kick(tdev->vq); /* guest "doorbell": ioeventfd-trapped in KVM,
                                    * never reaches QEMU's vCPU-blocking main loop
                                    * when ioeventfd is wired up */
    spin_unlock(&tdev->lock);
    if (ret)
        return ret;

    if (!wait_for_completion_timeout(&job->done, msecs_to_jiffies(2000)))
        return -ETIMEDOUT;
    return job->req.status;
}

static long toy_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct virtio_toy_dev *tdev = filp->private_data;
    struct toy_ioc_submit u;
    struct toy_job *job;
    int ret;

    if (cmd != TOY_IOC_SUBMIT)
        return -ENOTTY;
    if (copy_from_user(&u, (void __user *)arg, sizeof(u)))
        return -EFAULT;

    job = kzalloc(sizeof(*job), GFP_KERNEL);
    if (!job)
        return -ENOMEM;

    job->req.op = u.op;
    job->req.scalar = u.scalar;
    job->req.nelem = u.nelem;
    job->in_buf = kmalloc(u.nelem * sizeof(u32), GFP_KERNEL);
    job->out_buf = kmalloc(u.nelem * sizeof(u32), GFP_KERNEL);
    if (!job->in_buf || !job->out_buf) { ret = -ENOMEM; goto out; }

    if (copy_from_user(job->in_buf, (void __user *)u.in_uaddr,
                        u.nelem * sizeof(u32))) { ret = -EFAULT; goto out; }

    ret = toy_submit_and_wait(tdev, job);
    if (ret == 0 && copy_to_user((void __user *)u.out_uaddr, job->out_buf,
                                 u.nelem * sizeof(u32)))
        ret = -EFAULT;
out:
    kfree(job->in_buf);
    kfree(job->out_buf);
    kfree(job);
    return ret;
}

static int toy_open(struct inode *i, struct file *filp)
{
    filp->private_data = container_of(filp->private_data, struct virtio_toy_dev, miscdev);
    return 0;
}

static const struct file_operations toy_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = toy_ioctl,
    .open = toy_open,
};

static int virtio_toy_probe(struct virtio_device *vdev)
{
    struct virtio_toy_dev *tdev;
    vq_callback_t *callbacks[] = { toy_vq_callback };
    const char *names[] = { "toy_vq" };
    int ret;

    tdev = kzalloc(sizeof(*tdev), GFP_KERNEL);
    if (!tdev)
        return -ENOMEM;
    tdev->vdev = vdev;
    spin_lock_init(&tdev->lock);
    vdev->priv = tdev;

    /* find_vqs negotiates MSI-X/INTx and wires the callback to the
     * per-vq interrupt -- this is where PCIe MSI-X vectors get bound. */
    ret = virtio_find_vqs(vdev, 1, &tdev->vq, callbacks, names, NULL);
    if (ret)
        goto err_free;

    virtio_device_ready(vdev); /* DRIVER_OK -- guest tells host it's ready */

    tdev->miscdev.minor = MISC_DYNAMIC_MINOR;
    tdev->miscdev.name = "virtio_toy0";
    tdev->miscdev.fops = &toy_fops;
    tdev->miscdev.parent = &vdev->dev;
    ret = misc_register(&tdev->miscdev);
    if (ret)
        goto err_vqs;
    return 0;

err_vqs:
    vdev->config->del_vqs(vdev);
err_free:
    kfree(tdev);
    return ret;
}

static void virtio_toy_remove(struct virtio_device *vdev)
{
    struct virtio_toy_dev *tdev = vdev->priv;

    misc_deregister(&tdev->miscdev);
    virtio_reset_device(vdev);
    vdev->config->del_vqs(vdev);
    kfree(tdev);
}

static struct virtio_device_id id_table[] = {
    { VIRTIO_ID_TOY, VIRTIO_DEV_ANY_ID },
    { 0 },
};

static struct virtio_driver virtio_toy_driver = {
    .driver.name = "virtio_toy",
    .driver.owner = THIS_MODULE,
    .id_table = id_table,
    .probe = virtio_toy_probe,
    .remove = virtio_toy_remove,
};

module_virtio_driver(virtio_toy_driver);
MODULE_DEVICE_TABLE(virtio, id_table);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Guest driver for custom virtio-toy device");
