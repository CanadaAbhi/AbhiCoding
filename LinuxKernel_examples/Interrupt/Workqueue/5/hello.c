#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/interrupt.h>
#include <linux/delay.h>
#include <asm/current.h>
#include <linux/sched.h>

MODULE_LICENSE("GPL");

struct workqueue_struct *my_queue1 = NULL;
struct workqueue_struct *my_queue2 = NULL;

typedef struct my_work{
        struct work_struct work;
        char data[20];
}my_work;

my_work deferred_work1, deferred_work2;

static void work_fn1(struct work_struct *work)
{
	my_work *defer_work = (my_work *)container_of(work, my_work, work);
	pr_info("starting deferred work1 execution on processor id:%d\n", smp_processor_id());
	pr_info("%s:Data:%s\n", __func__, defer_work->data);
	mdelay(4000);
	pr_info("%s:current pid : %d , current process : %s\n", __func__,current->pid, current->comm);
	mdelay(4000);
	pr_info("%s:processor id:%d\n", __func__, smp_processor_id());
	mdelay(4000);
	pr_info("ending deferred work 1 execution on processor:%d\n", smp_processor_id());
}

static void work_fn2(struct work_struct *work)
{
	my_work *defer_work = (my_work *)container_of(work, my_work, work);
	pr_info("starting deferred work2 execution on processor id:%d\n", smp_processor_id());
	pr_info("%s:Data:%s\n", __func__, defer_work->data);
	mdelay(4000);
	pr_info("%s:current pid : %d , current process : %s\n", __func__,current->pid, current->comm);
	mdelay(4000);
	pr_info("%s:processor id:%d\n", __func__, smp_processor_id());
	mdelay(4000);
	pr_info("ending deferred work2 execution on processor:%d\n", smp_processor_id());
}


static int test_tasklet_init(void)
{
        pr_info("%s: In init processorid:%d\n", __func__, smp_processor_id());

	INIT_WORK(&deferred_work1.work, work_fn1);
	INIT_WORK(&deferred_work2.work, work_fn2);

	strcpy(deferred_work1.data, "Linux is easy");
	strcpy(deferred_work2.data, "Workqueues are easy");

	my_queue1 = alloc_ordered_workqueue("my_queue1", 0);
	queue_work(my_queue1, &deferred_work1.work);
	queue_work(my_queue1, &deferred_work2.work);
	pr_info("%s: Init complete processor id:%d\n", __func__, smp_processor_id());
	return 0;
}

static void test_tasklet_exit(void)
{
        pr_info("%s: In exit\n", __func__);
	destroy_workqueue(my_queue1);
}

module_init(test_tasklet_init);
module_exit(test_tasklet_exit);
