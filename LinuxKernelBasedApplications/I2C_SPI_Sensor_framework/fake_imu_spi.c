0x00 REG_ID        RO  magic 0xA9
0x01 REG_CTRL      RW  bit0=ENABLE
0x02 REG_ODR       RW  requested ODR in Hz (fake part accepts any value)
0x03 REG_INT_EN    RW  bit0=DRDY
0x04 REG_INT_STAT  RO/W1C bit0=DRDY
0x08..0x13 REG_AX..REG_GZ  RO  6 x signed 16-bit (ax,ay,az,gx,gy,gz), milli-g / milli-dps


// fake_imu_spi.c -- SPI 6-axis IMU driver registering with sensor_fw
#include <linux/module.h>
#include <linux/spi/spi.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <linux/pm_runtime.h>
#include <linux/of.h>
#include <linux/regulator/consumer.h>
#include <linux/slab.h>
#include "sensor_fw_core.h"

#define REG_ID        0x00
#define REG_CTRL      0x01
#define REG_ODR       0x02
#define REG_INT_EN    0x03
#define REG_INT_STAT  0x04
#define REG_AX        0x08  /* 6 consecutive s16 regs: ax,ay,az,gx,gy,gz */

#define ID_MAGIC      0xA9
#define CTRL_ENABLE   BIT(0)
#define INT_DRDY      BIT(0)
#define SPI_READ_BIT  0x80

struct fake_imu {
	struct spi_device *spi;
	struct sensor_fw_dev sfd;
	struct regulator *vdd;
	struct work_struct slow_work;
	atomic_t overrun_events;
	u32 odr_hz;
};

static int imu_reg_read(struct spi_device *spi, u8 reg, u8 *buf, size_t len)
{
	u8 cmd = reg | SPI_READ_BIT;
	return spi_write_then_read(spi, &cmd, 1, buf, len);
}
static int imu_reg_write(struct spi_device *spi, u8 reg, u8 val)
{
	u8 tx[2] = { reg, val };
	return spi_write(spi, tx, sizeof(tx));
}

static void fake_imu_slow_work(struct work_struct *w)
{
	struct fake_imu *imu = container_of(w, struct fake_imu, slow_work);
	dev_dbg(&imu->spi->dev, "slow-path: overrun_events=%d\n",
		atomic_read(&imu->overrun_events));
}

/* threaded IRQ: SPI transfers sleep, so like the I2C temp driver we run
 * entirely in thread context; nothing to do in a hard-irq primary handler. */
static irqreturn_t fake_imu_irq_thread(int irq, void *data)
{
	struct fake_imu *imu = data;
	struct spi_device *spi = imu->spi;
	u8 status;
	__be16 raw[6];
	s32 chan[6];
	int i, ret;

	pm_runtime_get_sync(&spi->dev);

	ret = imu_reg_read(spi, REG_INT_STAT, &status, 1);
	if (ret || !(status & INT_DRDY))
		goto out;

	ret = imu_reg_read(spi, REG_AX, (u8 *)raw, sizeof(raw));
	if (ret)
		goto out;

	for (i = 0; i < 6; i++)
		chan[i] = (s16)be16_to_cpu(raw[i]);

	if (sensor_fw_push_sample(&imu->sfd, chan, 0))
		atomic_inc(&imu->overrun_events);

	imu_reg_write(spi, REG_INT_STAT, INT_DRDY); /* W1C */
	queue_work(system_freezable_wq, &imu->slow_work);
out:
	pm_runtime_mark_last_busy(&spi->dev);
	pm_runtime_put_autosuspend(&spi->dev);
	return IRQ_HANDLED;
}

/* framework callback: userspace can retune sample rate via
 * ioctl(SENSOR_FW_IOC_SET_ODR) without any driver-specific ioctl. */
static int fake_imu_set_odr(struct sensor_fw_dev *sfd, u32 hz)
{
	struct fake_imu *imu = sfd->drvdata;
	int ret;

	pm_runtime_get_sync(&imu->spi->dev);
	ret = imu_reg_write(imu->spi, REG_ODR, (u8)min_t(u32, hz, 255));
	pm_runtime_put_autosuspend(&imu->spi->dev);
	if (!ret)
		imu->odr_hz = hz;
	return ret;
}

static ssize_t odr_hz_show(struct device *dev, struct device_attribute *a, char *buf)
{
	struct fake_imu *imu = dev_get_drvdata(dev->parent);
	return sysfs_emit(buf, "%u\n", imu->odr_hz);
}
static ssize_t odr_hz_store(struct device *dev, struct device_attribute *a,
			     const char *buf, size_t count)
{
	struct fake_imu *imu = dev_get_drvdata(dev->parent);
	u32 hz;
	int ret = kstrtou32(buf, 10, &hz);
	if (ret)
		return ret;
	ret = fake_imu_set_odr(&imu->sfd, hz);
	return ret ? ret : count;
}
static DEVICE_ATTR_RW(odr_hz);
static struct attribute *fake_imu_attrs[] = { &dev_attr_odr_hz.attr, NULL };
static const struct attribute_group fake_imu_group = { .attrs = fake_imu_attrs };

static int fake_imu_runtime_suspend(struct device *dev)
{
	struct fake_imu *imu = dev_get_drvdata(dev);
	imu_reg_write(imu->spi, REG_CTRL, 0);
	if (imu->vdd)
		regulator_disable(imu->vdd);
	return 0;
}
static int fake_imu_runtime_resume(struct device *dev)
{
	struct fake_imu *imu = dev_get_drvdata(dev);
	int ret;
	if (imu->vdd) {
		ret = regulator_enable(imu->vdd);
		if (ret)
			return ret;
	}
	imu_reg_write(imu->spi, REG_CTRL, CTRL_ENABLE);
	return 0;
}
static DEFINE_RUNTIME_DEV_PM_OPS(fake_imu_pm_ops,
				  fake_imu_runtime_suspend, fake_imu_runtime_resume, NULL);

static int fake_imu_probe(struct spi_device *spi)
{
	struct fake_imu *imu;
	u8 id;
	int ret;

	imu = devm_kzalloc(&spi->dev, sizeof(*imu), GFP_KERNEL);
	if (!imu)
		return -ENOMEM;
	imu->spi = spi;
	spi_set_drvdata(spi, imu);
	INIT_WORK(&imu->slow_work, fake_imu_slow_work);
	imu->odr_hz = 100;

	spi->mode = SPI_MODE_0;
	spi->bits_per_word = 8;
	ret = spi_setup(spi);
	if (ret)
		return ret;

	imu->vdd = devm_regulator_get_optional(&spi->dev, "vdd");
	if (IS_ERR(imu->vdd))
		imu->vdd = NULL;
	if (imu->vdd)
		regulator_enable(imu->vdd);

	ret = imu_reg_read(spi, REG_ID, &id, 1);
	if (ret || id != ID_MAGIC) {
		dev_err(&spi->dev, "bad ID 0x%02x\n", id);
		return -ENODEV;
	}
	imu_reg_write(spi, REG_CTRL, CTRL_ENABLE);
	imu_reg_write(spi, REG_ODR, imu->odr_hz);
	imu_reg_write(spi, REG_INT_EN, INT_DRDY);

	imu->sfd.name = "imu_spi0";
	imu->sfd.parent = &spi->dev;
	imu->sfd.type = SENSOR_FW_TYPE_IMU;
	imu->sfd.num_channels = 6;
	imu->sfd.scale_milli = 1000; /* chan[i] already milli-units */
	imu->sfd.set_odr = fake_imu_set_odr;
	imu->sfd.drvdata = imu;

	ret = sensor_fw_register(&imu->sfd);
	if (ret)
		return ret;

	ret = sysfs_create_group(&imu->sfd.dev->kobj, &fake_imu_group);
	if (ret)
		goto err_unreg;

	/* GPIO data-ready line, spi->irq populated from DT "interrupts" the
	 * same way client->irq is for I2C devices. */
	ret = devm_request_threaded_irq(&spi->dev, spi->irq, NULL,
					 fake_imu_irq_thread,
					 IRQF_ONESHOT | IRQF_TRIGGER_RISING,
					 "fake-imu-spi", imu);
	if (ret)
		goto err_sysfs;

	pm_runtime_set_autosuspend_delay(&spi->dev, 200);
	pm_runtime_use_autosuspend(&spi->dev);
	pm_runtime_enable(&spi->dev);

	dev_info(&spi->dev, "fake-imu-spi probed, irq=%d\n", spi->irq);
	return 0;

err_sysfs:
	sysfs_remove_group(&imu->sfd.dev->kobj, &fake_imu_group);
err_unreg:
	sensor_fw_unregister(&imu->sfd);
	return ret;
}

static void fake_imu_remove(struct spi_device *spi)
{
	struct fake_imu *imu = spi_get_drvdata(spi);

	pm_runtime_disable(&spi->dev);
	sysfs_remove_group(&imu->sfd.dev->kobj, &fake_imu_group);
	sensor_fw_unregister(&imu->sfd);
	if (imu->vdd)
		regulator_disable(imu->vdd);
}

static const struct of_device_id fake_imu_of_match[] = {
	{ .compatible = "acme,fake-imu-spi" },
	{},
};
MODULE_DEVICE_TABLE(of, fake_imu_of_match);

static struct spi_driver fake_imu_driver = {
	.driver = {
		.name = "fake_imu_spi",
		.of_match_table = fake_imu_of_match,
		.pm = pm_ptr(&fake_imu_pm_ops),
	},
	.probe = fake_imu_probe,
	.remove = fake_imu_remove,
};
module_spi_driver(fake_imu_driver);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SPI 6-axis IMU driver on sensor_fw");
