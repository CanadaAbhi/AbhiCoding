#ifndef ACCEL_UAPI_H
#define ACCEL_UAPI_H
#include <linux/types.h>
#include <linux/ioctl.h>

enum accel_opcode {
	ACCEL_OP_MEMCPY       = 0,
	ACCEL_OP_VEC_ADD_SCAL = 1, /* output[i] = input[i] + scalar (u8 arrays) */
	ACCEL_OP_VEC_SCALE    = 2, /* output[i] = input[i] * scalar             */
	ACCEL_OP_CHECKSUM     = 3, /* output[0..7] = 64-bit sum of input bytes  */
};

enum accel_priority { ACCEL_PRIO_HIGH = 0, ACCEL_PRIO_NORMAL = 1, ACCEL_PRIO_LOW = 2 };

struct accel_buffer_alloc {
	__u64 size;
	__u32 flags;
	__u32 handle;       /* out */
	__u64 mmap_offset;  /* out: pass to mmap() as offset */
};

struct accel_buffer_free {
	__u32 handle;
};

struct accel_submit {
	__u32 input_handle;
	__u64 input_offset;
	__u64 input_len;
	__u32 output_handle;
	__u64 output_offset;
	__u64 output_len;
	__u32 opcode;
	__s32 scalar;
	__u32 priority;
	__u64 out_seqno;     /* out: fence sequence number */
};

struct accel_wait {
	__u64 seqno;
	__u64 timeout_ns;
	__u32 out_status;    /* out: 0=ok, 1=timeout, 2=error */
};

struct accel_stats {
	__u64 jobs_submitted;
	__u64 jobs_completed;
	__u64 jobs_failed;
	__u64 cmdq_depth_cur;
	__u64 cmdq_depth_max;
	__u64 latency_ns_min;
	__u64 latency_ns_max;
	__u64 latency_ns_sum;
	__u64 latency_ns_count;
};

#define ACCEL_MMAP_UNIT (1ULL << 24) /* handle N maps to mmap offset N * ACCEL_MMAP_UNIT */

#define ACCEL_IOC_MAGIC 0xAC
#define ACCEL_IOC_BUFFER_ALLOC _IOWR(ACCEL_IOC_MAGIC, 1, struct accel_buffer_alloc)
#define ACCEL_IOC_BUFFER_FREE  _IOW (ACCEL_IOC_MAGIC, 2, struct accel_buffer_free)
#define ACCEL_IOC_SUBMIT       _IOWR(ACCEL_IOC_MAGIC, 3, struct accel_submit)
#define ACCEL_IOC_WAIT         _IOWR(ACCEL_IOC_MAGIC, 4, struct accel_wait)
#define ACCEL_IOC_GET_STATS    _IOR (ACCEL_IOC_MAGIC, 5, struct accel_stats)

#endif
