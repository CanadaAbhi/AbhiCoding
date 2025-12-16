0x00 REG_ID          RO   magic 0x54
0x01 REG_TEMP_MSB    RO   \_ signed 16-bit, centi-degrees C (e.g. 2543 = 25.43C)
0x02 REG_TEMP_LSB    RO   /
0x03 REG_CONFIG      RW   bit0=ENABLE
0x04 REG_THRESH_MSB  RW   \_ signed 16-bit high-threshold, centi-degrees
0x05 REG_THRESH_LSB  RW   /
0x06 REG_INT_STATUS  RO/W1C bit0=THRESH_HIGH
0x07 REG_INT_ENABLE  RW   bit0=THRESH_HIGH


// fake_temp_i2c.c -- I2C temperature sensor driver registering with sensor_fw
#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/interrupt.h>
#include <linux/gpio/consumer.h>
#include <linux/workqueue.h>
#include <linux/pm_runtime.h>
#include <linux/of.h>
#include <linux/regulator/consumer.h>
#include <linux/slab.h>
#include "sensor_fw_core.h"

#define REG_ID          0x00
#define REG_TEMP_MSB    0x01
#define REG_TEMP_LSB    0x02
#define REG_CONFIG      0x03
#define REG_THRESH_MSB  0x04
#define REG_THRESH_LSB  0x05
#define REG_INT_STATUS  0x06
#define REG_INT_ENABLE  0x07

#define ID_MAGIC        0x54
#define CONFIG_ENABLE   BIT(0)
#define INT_THRESH_HIGH BIT(0)

struct fake_temp {
	struct i2c_client *client;
	struct sensor_fw_dev sfd;
	struct regulator *vdd;         /* stub supply, models RPMh-voted rail */
	struct work_struct slow_work;  /* deferred drift/overrun bookkeeping */
	atomic_t drift_events;
};

static int temp_read_s16(struct i2c_client *c, u8 msb_reg)
{
	int msb = i2c_smbus_read_byte_data(c, msb_reg);
	int lsb = i2c_smbus_read_byte_data(c, msb_reg + 1);
	if (msb < 0 || lsb < 0)
		return -EIO;
	return (s16)((msb << 8) | lsb);
}

static int temp_write_s16(struct i2c_client *c, u8 msb_reg, s16 val)
{
	int ret = i2c_smbus_write_byte_data(c, msb_reg, (val >> 8) & 0xff);
	if (ret)
		return ret;
	return i2c_smbus_write_byte_data(c, msb_reg + 1, val & 0xff);
}

/* slow-path: runs in process context off the workqueue, updates stats that
 * don't need to be on the IRQ-thread critical path */
static void fake_temp_slow_work(struct work_struct *w)
{
	struct fake_temp *ft = container_of(w, struct fake_temp, slow_work);
	dev_dbg(&ft->client->dev, "slow-path: drift_events=%d overrun=%d\n",
		atomic_read(&ft->drift_events), atomic_read(&ft->sfd.overrun_count));
}

/* threaded IRQ: primary handler is NULL (IRQF_ONESHOT) because the very
 * first thing we need to do is an I2C transfer, which sleeps -- can't run
 * in hard-irq context, unlike the PCIe MMIO ISR in pcie_dma_drv. */
static irqreturn_t fake_temp_irq_thread(int irq, void *data)
{
	struct fake_temp *ft = data;
	struct i2c_client *c = ft->client;
	int status, temp;
	s32 chan[1];

	pm_runtime_get_sync(&c->dev);

	status = i2c_smbus_read_byte_data(c, REG_INT_STATUS);
	if (status < 0)
		goto out;

	if (status & INT_THRESH_HIGH) {
		temp = temp_read_s16(c, REG_TEMP_MSB);
		if (temp >= 0 || temp < 0) { /* always true; keep for clarity */
			chan[0] = temp;
			sensor_fw_push_sample(&ft->sfd, chan, 0);
		}
		i2c_smbus_write_byte_data(c, REG_INT_STATUS, INT_THRESH_HIGH); /* W1C */
		atomic_inc(&ft->drift_events);
		queue_work(system_freezable_wq, &ft->slow_work);
	}
out:
	pm_runtime_mark_last_busy(&c->dev);
	pm_runtime_put_autosuspend(&c->dev);
	return IRQ_HANDLED;
}

/* ---- sysfs: driver-specific extras beyond the framework's common attrs ---- */
static ssize_t threshold_mC_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct fake_temp *ft = dev_get_drvdata(dev->parent);
	int val;

	pm_runtime_get_sync(&ft->client->dev);
	val = temp_read_s16(ft->client, REG_THRESH_MSB);
	pm_runtime_put_autosuspend(&ft->client->dev);
	return sysfs_emit(buf, "%d\n", val * 10); /* centi-C -> milli-C */
}
static ssize_t threshold_mC_store(struct device *dev, struct device_attribute *a,
				   const char *buf, size_t count)
{
	struct fake_temp *ft = dev_get_drvdata(dev->parent);
	long milli_c;
	int ret = kstrtol(buf, 10, &milli_c);
	if (ret)
		return ret;

	pm_runtime_get_sync(&ft->client->dev);
	ret = temp_write_s16(ft->client, REG_THRESH_MSB, (s16)(milli_c / 10));
	pm_runtime_put_autosuspend(&ft->client->dev);
	return ret ? ret : count;
}
static DEVICE_ATTR_RW(threshold_mC);

static struct attribute *fake_temp_attrs[] = { &dev_attr_threshold_mC.attr, NULL };
static const struct attribute_group fake_temp_group = { .attrs = fake_temp_attrs };

/* ---- runtime PM ---- */
static int fake_temp_runtime_suspend(struct device *dev)
{
	struct fake_temp *ft = dev_get_drvdata(dev);
	i2c_smbus_write_byte_data(ft->client, REG_CONFIG, 0); /* disable */
	if (ft->vdd)
		regulator_disable(ft->vdd);
	return 0;
}
static int fake_temp_runtime_resume(struct device *dev)
{
	struct fake_temp *ft = dev_get_drvdata(dev);
	int ret;
	if (ft->vdd) {
		ret = regulator_enable(ft->vdd);
		if (ret)
			return ret;
	}
	i2c_smbus_write_byte_data(ft->client, REG_CONFIG, CONFIG_ENABLE);
	return 0;
}
static DEFINE_RUNTIME_DEV_PM_OPS(fake_temp_pm_ops,
				  fake_temp_runtime_suspend, fake_temp_runtime_resume, NULL);

static int fake_temp_probe(struct i2c_client *client)
{
	struct fake_temp *ft;
	int id, ret;

	ft = devm_kzalloc(&client->dev, sizeof(*ft), GFP_KERNEL);
	if (!ft)
		return -ENOMEM;
	ft->client = client;
	i2c_set_clientdata(client, ft);
	INIT_WORK(&ft->slow_work, fake_temp_slow_work);

	ft->vdd = devm_regulator_get_optional(&client->dev, "vdd");
	if (IS_ERR(ft->vdd))
		ft->vdd = NULL;
	if (ft->vdd)
		regulator_enable(ft->vdd);

	id = i2c_smbus_read_byte_data(client, REG_ID);
	if (id != ID_MAGIC) {
		dev_err(&client->dev, "bad ID 0x%02x, want 0x%02x\n", id, ID_MAGIC);
		return -ENODEV;
	}
	i2c_smbus_write_byte_data(client, REG_CONFIG, CONFIG_ENABLE);
	i2c_smbus_write_byte_data(client, REG_INT_ENABLE, INT_THRESH_HIGH);

	ft->sfd.name = "temp0";
	ft->sfd.parent = &client->dev;
	ft->sfd.type = SENSOR_FW_TYPE_TEMP;
	ft->sfd.num_channels = 1;
	ft->sfd.scale_milli = 10; /* chan[0] is centi-C; real_mC = chan[0] * 10 */
	ft->sfd.set_odr = NULL;   /* fixed conversion rate on this fake part */
	ft->sfd.drvdata = ft;

	ret = sensor_fw_register(&ft->sfd);
	if (ret)
		return ret;

	ret = sysfs_create_group(&ft->sfd.dev->kobj, &fake_temp_group);
	if (ret)
		goto err_unreg;

	/* client->irq is auto-populated from the DT "interrupts" property via
	 * of_i2c_register_devices(), same mechanism as fake_imu_drv. */
	ret = devm_request_threaded_irq(&client->dev, client->irq, NULL,
					 fake_temp_irq_thread,
					 IRQF_ONESHOT | IRQF_TRIGGER_RISING,
					 "fake-temp", ft);
	if (ret)
		goto err_sysfs;

	pm_runtime_set_autosuspend_delay(&client->dev, 200);
	pm_runtime_use_autosuspend(&client->dev);
	pm_runtime_enable(&client->dev);

	dev_info(&client->dev, "fake-temp probed, irq=%d\n", client->irq);
	return 0;

err_sysfs:
	sysfs_remove_group(&ft->sfd.dev->kobj, &fake_temp_group);
err_unreg:
	sensor_fw_unregister(&ft->sfd);
	return ret;
}

static void fake_temp_remove(struct i2c_client *client)
{
	struct fake_temp *ft = i2c_get_clientdata(client);

	pm_runtime_disable(&client->dev);
	sysfs_remove_group(&ft->sfd.dev->kobj, &fake_temp_group);
	sensor_fw_unregister(&ft->sfd);
	if (ft->vdd)
		regulator_disable(ft->vdd);
}

static const struct of_device_id fake_temp_of_match[] = {
	{ .compatible = "acme,fake-temp" },
	{},
};
MODULE_DEVICE_TABLE(of, fake_temp_of_match);

static struct i2c_driver fake_temp_driver = {
	.driver = {
		.name = "fake_temp_i2c",
		.of_match_table = fake_temp_of_match,
		.pm = pm_ptr(&fake_temp_pm_ops),
	},
	.probe = fake_temp_probe,
	.remove = fake_temp_remove,
};
module_i2c_driver(fake_temp_driver);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("I2C temperature sensor driver on sensor_fw");
