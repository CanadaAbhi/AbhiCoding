#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/delay.h>
#include <asm/current.h>
#include <linux/sched.h>

MODULE_LICENSE("GPL");

struct workqueue_struct *my_queue = NULL;
typedef struct my_work{
        struct work_struct work;
        char data[20];
}my_work;

my_work deferred_work;

static void work_fn(struct work_struct *work)
{
	my_work *defer_work = (my_work *)container_of(work, my_work, work);
	pr_info("starting deferred work execution\n");
	pr_info("Data:%s\n", defer_work->data);
	pr_info("current pid : %d , current process : %s\n",current->pid, current->comm);
	pr_info("processor id:%d\n", smp_processor_id());
	pr_info("ending deferred work execution\n");
}


static int test_tasklet_init(void)
{
        pr_info("%s: In init processorid:%d\n", __func__, smp_processor_id());
	INIT_WORK(&deferred_work.work, work_fn);
	strcpy(deferred_work.data, "Linux is easy");
	my_queue = alloc_workqueue("my_queue", WQ_HIGHPRI, 1);
	queue_work(my_queue, &deferred_work.work);
	flush_workqueue(my_queue);
	pr_info("%s: Init complete processor id:%d\n", __func__, smp_processor_id());
	return 0;
}

static void test_tasklet_exit(void)
{
        pr_info("%s: In exit\n", __func__);
	destroy_workqueue(my_queue);
}

module_init(test_tasklet_init);
module_exit(test_tasklet_exit);
