// hw_event_gen.c -- simulates a hardware event source and its interrupt line
// using a real Linux irq_domain + hrtimer, so downstream drivers exercise the
// genuine request_irq()/request_threaded_irq()/handle_simple_irq() code paths
// without needing physical silicon.
#include <linux/module.h>
#include <linux/irq.h>
#include <linux/irqdomain.h>
#include <linux/hrtimer.h>
#include <linux/spinlock.h>
#include <linux/device.h>
#include <linux/slab.h>
#include "hw_event_gen.h"

#define NR_FAKE_IRQS 1
#define FAKE_HWIRQ   0

struct hweg_dev {
	struct irq_domain *domain;
	int virq;

	spinlock_t reg_lock;      /* protects the fake register file */
	u32 status_reg;           /* bit0 = IRQ_PENDING, like a real device */
	ktime_t last_event_ts;
	u64 events_generated;

	struct hrtimer timer;
	u32 event_hz;
	bool enabled;

	struct device *dev;
};

#define STATUS_IRQ_PENDING BIT(0)

static struct hweg_dev *g_hweg;

/* ---- minimal irq_chip: no real masking hardware, so mask/unmask are
 * bookkeeping no-ops. This is what handle_simple_irq() calls into. ---- */
static void hweg_irq_mask(struct irq_data *d) { }
static void hweg_irq_unmask(struct irq_data *d) { }

static struct irq_chip hweg_irq_chip = {
	.name = "hweg",
	.irq_mask = hweg_irq_mask,
	.irq_unmask = hweg_irq_unmask,
};

static int hweg_domain_map(struct irq_domain *d, unsigned int virq, irq_hw_number_t hwirq)
{
	irq_set_chip_and_handler(virq, &hweg_irq_chip, handle_simple_irq);
	irq_set_chip_data(virq, d->host_data);
	return 0;
}

static const struct irq_domain_ops hweg_domain_ops = {
	.map = hweg_domain_map,
	.xlate = irq_domain_xlate_onecell,
};

/* hrtimer callback = the "hardware" raising its interrupt line. Runs in
 * hardirq context, exactly like a real parent-IRQ demux handler would. */
static enum hrtimer_restart hweg_timer_fn(struct hrtimer *t)
{
	struct hweg_dev *hw = container_of(t, struct hweg_dev, timer);
	unsigned long flags;

	spin_lock_irqsave(&hw->reg_lock, flags);
	hw->status_reg |= STATUS_IRQ_PENDING;
	hw->last_event_ts = ktime_get();
	hw->events_generated++;
	spin_unlock_irqrestore(&hw->reg_lock, flags);

	generic_handle_irq(hw->virq);   /* deliver the fake IRQ to whoever requested it */

	if (hw->enabled)
		hrtimer_forward_now(t, ns_to_ktime(NSEC_PER_SEC / hw->event_hz));
	return hw->enabled ? HRTIMER_RESTART : HRTIMER_NORESTART;
}

int hweg_get_irq(void)
{
	return g_hweg ? g_hweg->virq : -ENODEV;
}
EXPORT_SYMBOL_GPL(hweg_get_irq);

/* models a real MMIO write to a status/ack register: clears IRQ_PENDING */
void hweg_ack(void)
{
	unsigned long flags;

	if (!g_hweg)
		return;
	spin_lock_irqsave(&g_hweg->reg_lock, flags);
	g_hweg->status_reg &= ~STATUS_IRQ_PENDING;
	spin_unlock_irqrestore(&g_hweg->reg_lock, flags);
}
EXPORT_SYMBOL_GPL(hweg_ack);

ktime_t hweg_last_event_ts(void)
{
	return g_hweg ? g_hweg->last_event_ts : 0;
}
EXPORT_SYMBOL_GPL(hweg_last_event_ts);

/* ---- sysfs runtime control ---- */
static ssize_t event_hz_show(struct device *dev, struct device_attribute *a, char *buf)
{
	return sysfs_emit(buf, "%u\n", g_hweg->event_hz);
}
static ssize_t event_hz_store(struct device *dev, struct device_attribute *a,
			       const char *buf, size_t count)
{
	u32 hz;
	int ret = kstrtou32(buf, 10, &hz);
	if (ret || hz == 0 || hz > 100000)
		return -EINVAL;
	g_hweg->event_hz = hz;
	return count;
}
static DEVICE_ATTR_RW(event_hz);

static ssize_t enable_show(struct device *dev, struct device_attribute *a, char *buf)
{
	return sysfs_emit(buf, "%d\n", g_hweg->enabled);
}
static ssize_t enable_store(struct device *dev, struct device_attribute *a,
			     const char *buf, size_t count)
{
	bool on;
	int ret = kstrtobool(buf, &on);
	if (ret)
		return ret;

	if (on && !g_hweg->enabled) {
		g_hweg->enabled = true;
		hrtimer_start(&g_hweg->timer, ns_to_ktime(NSEC_PER_SEC / g_hweg->event_hz),
			      HRTIMER_MODE_REL);
	} else if (!on && g_hweg->enabled) {
		g_hweg->enabled = false;
		hrtimer_cancel(&g_hweg->timer);
	}
	return count;
}
static DEVICE_ATTR_RW(enable);

static ssize_t events_generated_show(struct device *dev, struct device_attribute *a, char *buf)
{
	return sysfs_emit(buf, "%llu\n", g_hweg->events_generated);
}
static DEVICE_ATTR_RO(events_generated);

static struct attribute *hweg_attrs[] = {
	&dev_attr_event_hz.attr, &dev_attr_enable.attr, &dev_attr_events_generated.attr, NULL,
};
ATTRIBUTE_GROUPS(hweg);

static struct class *hweg_class;

static int __init hw_event_gen_init(void)
{
	struct hweg_dev *hw;
	int ret;

	hw = kzalloc(sizeof(*hw), GFP_KERNEL);
	if (!hw)
		return -ENOMEM;
	spin_lock_init(&hw->reg_lock);
	hw->event_hz = 1000;   /* 1kHz default fake event rate */
	hw->enabled = true;

	hw->domain = irq_domain_add_linear(NULL, NR_FAKE_IRQS, &hweg_domain_ops, hw);
	if (!hw->domain) {
		ret = -ENOMEM;
		goto err_free;
	}
	hw->virq = irq_create_mapping(hw->domain, FAKE_HWIRQ);
	if (!hw->virq) {
		ret = -ENOMEM;
		goto err_domain;
	}

	hweg_class = class_create("hweg");
	if (IS_ERR(hweg_class)) {
		ret = PTR_ERR(hweg_class);
		goto err_domain;
	}
	hw->dev = device_create_with_groups(hweg_class, NULL, MKDEV(0, 0), hw,
					     hweg_groups, "hweg0");
	if (IS_ERR(hw->dev)) {
		ret = PTR_ERR(hw->dev);
		goto err_class;
	}

	hrtimer_init(&hw->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	hw->timer.function = hweg_timer_fn;

	g_hweg = hw;
	hrtimer_start(&hw->timer, ns_to_ktime(NSEC_PER_SEC / hw->event_hz), HRTIMER_MODE_REL);

	pr_info("hw_event_gen: virq=%d, event_hz=%u\n", hw->virq, hw->event_hz);
	return 0;

err_class:
	class_destroy(hweg_class);
err_domain:
	irq_domain_remove(hw->domain);
err_free:
	kfree(hw);
	return ret;
}

static void __exit hw_event_gen_exit(void)
{
	hrtimer_cancel(&g_hweg->timer);
	device_destroy(hweg_class, MKDEV(0, 0));
	class_destroy(hweg_class);
	irq_dispose_mapping(g_hweg->virq);
	irq_domain_remove(g_hweg->domain);
	kfree(g_hweg);
}
module_init(hw_event_gen_init);
module_exit(hw_event_gen_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Virtual hardware event generator + fake IRQ line for irq_defer_lab");
