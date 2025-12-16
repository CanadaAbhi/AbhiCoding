#ifndef SENSOR_FW_CORE_H
#define SENSOR_FW_CORE_H
#include <linux/kfifo.h>
#include <linux/cdev.h>
#include <linux/wait.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/device.h>
#include "sensor_fw_uapi.h"

#define SENSOR_FW_FIFO_DEPTH 64   /* samples buffered per device */

/*
 * One struct per registered sensor. Low-level bus drivers (I2C/SPI) fill in
 * the "input" fields and call sensor_fw_register(); everything below the
 * comment line is framework-owned after that call succeeds.
 */
struct sensor_fw_dev {
	/* ---- filled in by the bus driver before registering ---- */
	const char *name;              /* "temp0", "imu_spi0", ... */
	struct device *parent;         /* &i2c_client->dev / &spi_device->dev */
	enum sensor_fw_type type;
	u32 num_channels;
	s32 scale_milli;
	int (*set_odr)(struct sensor_fw_dev *sfd, u32 hz); /* optional, NULL if fixed-rate */
	void *drvdata;

	/* ---- framework-owned ---- */
	struct device *dev;
	struct cdev cdev;
	dev_t devt;
	int minor;
	struct kfifo fifo;              /* byte fifo of struct sensor_fw_sample records */
	wait_queue_head_t wq;
	struct mutex lock;
	atomic_t overrun_count;
	atomic64_t sample_count;
	struct sensor_fw_sample last_sample;
	bool have_last;
};

int sensor_fw_register(struct sensor_fw_dev *sfd);
void sensor_fw_unregister(struct sensor_fw_dev *sfd);
int sensor_fw_push_sample(struct sensor_fw_dev *sfd, const s32 *chan, u32 flags);

#endif
