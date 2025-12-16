#ifndef HW_EVENT_GEN_H
#define HW_EVENT_GEN_H
#include <linux/ktime.h>

/* Exported by hw_event_gen.ko for irq_defer_lab.ko to consume. */
int hweg_get_irq(void);          /* returns the fake Linux IRQ number, or -ENODEV */
void hweg_ack(void);             /* simulates writing the device's IRQ-ack register */
ktime_t hweg_last_event_ts(void); /* ktime the hrtimer last raised the line */

#endif
