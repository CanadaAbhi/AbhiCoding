// SPDX-License-Identifier: GPL-2.0
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/list.h>
#include <linux/printk.h>

/*
 * Educational GPU scheduler
 * - Command queue
 * - GPU worker thread
 * - No DRM scheduler yet (next stage)
 */

/* ------------------------------------------------------------------ */
/* Command object                                                      */
/* ------------------------------------------------------------------ */

struct mygpu_cmd {
    struct list_head node;
    size_t size;
};

/* ------------------------------------------------------------------ */
/* Scheduler state                                                     */
/* ------------------------------------------------------------------ */

static struct task_struct *gpu_thread;
static LIST_HEAD(cmd_queue);
static DEFINE_MUTEX(queue_lock);
static bool running;

/* ------------------------------------------------------------------ */
/* GPU thread                                                         */
/* ------------------------------------------------------------------ */

static int mygpu_gpu_thread(void *data)
{
    struct mygpu_cmd *cmd;

    pr_info("mygpu_sched: GPU thread started\n");

    while (!kthread_should_stop()) {

        mutex_lock(&queue_lock);

        if (list_empty(&cmd_queue)) {
            mutex_unlock(&queue_lock);
            msleep(10);
            continue;
        }

        cmd = list_first_entry(&cmd_queue, struct mygpu_cmd, node);
        list_del(&cmd->node);

        mutex_unlock(&queue_lock);

        /* Simulate GPU execution */
        pr_info("mygpu_sched: executing command (%zu bytes)\n", cmd->size);
        msleep(16); /* ~60 FPS */

        kfree(cmd);
    }

    pr_info("mygpu_sched: GPU thread stopping\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API (used by mygpu_drv.c)                                    */
/* ------------------------------------------------------------------ */

int mygpu_sched_start(void)
{
    if (running)
        return 0;

    gpu_thread = kthread_run(mygpu_gpu_thread, NULL, "mygpu-gpu");
    if (IS_ERR(gpu_thread))
        return PTR_ERR(gpu_thread);

    running = true;
    return 0;
}

void mygpu_sched_stop(void)
{
    if (!running)
        return;

    kthread_stop(gpu_thread);
    running = false;
}

void mygpu_sched_submit(size_t size)
{
    struct mygpu_cmd *cmd;

    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd)
        return;

    cmd->size = size;

    mutex_lock(&queue_lock);
    list_add_tail(&cmd->node, &cmd_queue);
    mutex_unlock(&queue_lock);

    pr_info("mygpu_sched: command queued (%zu bytes)\n", size);
}
