#ifndef SENSOR_FW_UAPI_H
#define SENSOR_FW_UAPI_H
#include <linux/types.h>
#include <linux/ioctl.h>

#define SENSOR_FW_MAX_CHANNELS 6   /* enough for 3-axis accel + 3-axis gyro */

struct sensor_fw_sample {
	__u64 ts_ns;
	__s32 chan[SENSOR_FW_MAX_CHANNELS]; /* driver-defined fixed-point units */
	__u32 flags;
};
#define SENSOR_FW_FLAG_OVERRUN 0x1

enum sensor_fw_type {
	SENSOR_FW_TYPE_TEMP = 1,
	SENSOR_FW_TYPE_IMU  = 2,
};

struct sensor_fw_info {
	__u32 type;
	__u32 num_channels;
	__s32 scale_milli;   /* real_value = chan / scale_milli (see driver docs) */
	char  name[32];
};

#define SENSOR_FW_MAGIC 0xE5
#define SENSOR_FW_IOC_GET_INFO  _IOR(SENSOR_FW_MAGIC, 1, struct sensor_fw_info)
#define SENSOR_FW_IOC_SET_ODR   _IOW(SENSOR_FW_MAGIC, 2, __u32)

#endif
