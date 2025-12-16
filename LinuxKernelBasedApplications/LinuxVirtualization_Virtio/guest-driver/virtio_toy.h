#ifndef VIRTIO_TOY_H
#define VIRTIO_TOY_H
#include <linux/types.h>
#include <linux/ioctl.h>

#define VIRTIO_ID_TOY 63

struct virtio_toy_req {
    __u32 op;
    __u32 scalar;
    __u32 nelem;
    __u32 status;
};

enum { TOY_OP_MEMCPY = 0, TOY_OP_ADD_SCALAR = 1, TOY_OP_SQUARE = 2 };

struct toy_ioc_submit {
    __u32 op, scalar, nelem;
    __u64 in_uaddr, out_uaddr;   /* userspace buffer pointers */
};

#define TOY_IOC_MAGIC 0xC0
#define TOY_IOC_SUBMIT _IOWR(TOY_IOC_MAGIC, 1, struct toy_ioc_submit)

#endif
