#ifndef IRQ_DEFER_UAPI_H
#define IRQ_DEFER_UAPI_H
#include <linux/types.h>
#include <linux/ioctl.h>

enum irq_defer_mode {
	IRQ_DEFER_MODE_WORKQUEUE    = 0,  /* hardirq -> workqueue -> heavy work           */
	IRQ_DEFER_MODE_TASKLET      = 1,  /* hardirq -> tasklet (softirq) -> heavy work   */
	IRQ_DEFER_MODE_THREADED_IRQ = 2,  /* hardirq -> irq thread -> workqueue -> heavy  */
};

struct irq_defer_record {
	__u64 seq;
	__u32 mode;
	__u64 dispatch_ns;  /* ack -> deferred-handler-start (scheduling latency) */
	__u64 proc_ns;      /* deferred-handler heavy-processing duration */
	__u64 total_ns;     /* hw-event -> wake (end-to-end) */
};

#define IRQ_DEFER_MAGIC 0xE6
#define IRQ_DEFER_IOC_GET_MODE     _IOR(IRQ_DEFER_MAGIC, 1, __u32)
#define IRQ_DEFER_IOC_RESET_STATS  _IO(IRQ_DEFER_MAGIC, 2)

#endif
