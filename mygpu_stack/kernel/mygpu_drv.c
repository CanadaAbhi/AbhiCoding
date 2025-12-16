// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/fs.h>

#include <drm/drm_drv.h>
#include <drm/drm_device.h>
#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_ioctl.h>
#include <drm/drm_managed.h>

#include "mygpu_gem.h"
#include "mygpu_sched.h"

/* ------------------------------------------------------------------ */
/* Fake GPU thread (Stage 3)                                           */
/* ------------------------------------------------------------------ */

static struct task_struct *gpu_thread;

static int mygpu_gpu_thread(void *data)
{
    while (!kthread_should_stop()) {
        pr_info("mygpu: fake GPU executing command buffer\n");
        msleep(16); /* ~60 FPS */
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* File operations                                                     */
/* ------------------------------------------------------------------ */

static ssize_t mygpu_write(struct file *filp,
    const char __user *buf,
    size_t count,
    loff_t *off)
{
mygpu_sched_submit(count);
return count;
}


static const struct file_operations mygpu_fops = {
    .owner = THIS_MODULE,
    .write = mygpu_write,
};

/* ------------------------------------------------------------------ */
/* DRM driver                                                          */
/* ------------------------------------------------------------------ */

static const struct drm_driver mygpu_drm_driver = {
    .driver_features = DRIVER_GEM | DRIVER_RENDER,
    .name            = "mygpu",
    .desc            = "Educational GPU Driver",
    .date            = "2025",
    .major           = 1,
    .minor           = 0,
    .fops            = &mygpu_fops,
};

/* ------------------------------------------------------------------ */
/* Platform driver                                                     */
/* ------------------------------------------------------------------ */

static int mygpu_probe(struct platform_device *pdev)
{
    struct drm_device *drm;
    int ret;

    drm = drm_dev_alloc(&mygpu_drm_driver, &pdev->dev);
    if (IS_ERR(drm))
        return PTR_ERR(drm);

    ret = drm_dev_register(drm, 0);
    if (ret) {
        drm_dev_put(drm);
        return ret;
    }

    gpu_thread = kthread_run(mygpu_gpu_thread, NULL, "mygpu-gpu");
    if (IS_ERR(gpu_thread)) {
        drm_dev_unregister(drm);
        drm_dev_put(drm);
        return PTR_ERR(gpu_thread);
    }

    platform_set_drvdata(pdev, drm);
    pr_info("mygpu: DRM device registered\n");

    ret = mygpu_sched_start();
    if (ret)
      return ret;

    return 0;
}

static int mygpu_remove(struct platform_device *pdev)
{
    struct drm_device *drm = platform_get_drvdata(pdev);
    mygpu_sched_stop();
    if (gpu_thread)
        kthread_stop(gpu_thread);

    drm_dev_unregister(drm);
    drm_dev_put(drm);
    pr_info("mygpu: device removed\n");
    return 0;
}

static struct platform_driver mygpu_platform_driver = {
    .probe  = mygpu_probe,
    .mmap = mygpu_mmap,
    .remove = mygpu_remove,
    .driver = {
        .name = "mygpu",
    },
};

/* ------------------------------------------------------------------ */
/* Module init / exit                                                  */
/* ------------------------------------------------------------------ */

static struct platform_device *mygpu_pdev;

static int __init mygpu_init(void)
{
    int ret;

    ret = platform_driver_register(&mygpu_platform_driver);
    if (ret)
        return ret;

    mygpu_pdev = platform_device_register_simple("mygpu", -1, NULL, 0);
    if (IS_ERR(mygpu_pdev)) {
        platform_driver_unregister(&mygpu_platform_driver);
        return PTR_ERR(mygpu_pdev);
    }

    pr_info("mygpu: module loaded\n");
    return 0;
}

static int mygpu_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct drm_gem_object *obj = vma->vm_private_data;
    struct mygpu_gem *mgem = container_of(obj, struct mygpu_gem, base);

    return remap_vmalloc_range(vma, mgem->vaddr, 0);
}

static void __exit mygpu_exit(void)
{
    platform_device_unregister(mygpu_pdev);
    platform_driver_unregister(&mygpu_platform_driver);
    pr_info("mygpu: module unloaded\n");
}

module_init(mygpu_init);
module_exit(mygpu_exit);

MODULE_AUTHOR("You");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Minimal End-to-End GPU Driver");



