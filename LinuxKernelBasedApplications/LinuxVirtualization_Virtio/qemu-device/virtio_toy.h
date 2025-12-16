// virtio-toy.h -- QEMU device model header. This is the "Host Driver" /
// device-model half of the custom VirtIO device: it owns the virtqueue,
// pops guest descriptors, does a toy transform, and notifies completion.
// Structurally identical to how QEMU's real virtio-net/virtio-blk device
// models work -- same VirtIODevice base, same virtqueue API.
#ifndef QEMU_VIRTIO_TOY_H
#define QEMU_VIRTIO_TOY_H

#include "hw/virtio/virtio.h"
#include "qom/object.h"

#define TYPE_VIRTIO_TOY "virtio-toy-device"
OBJECT_DECLARE_SIMPLE_TYPE(VirtIOToy, VIRTIO_TOY)

/* Must match guest driver's uapi definition exactly (shared "ABI") */
struct virtio_toy_req {
    uint32_t op;      /* 0=memcpy 1=add_scalar 2=square */
    uint32_t scalar;
    uint32_t nelem;
    uint32_t status;  /* filled by device: 0 = OK */
};

enum {
    VIRTIO_TOY_OP_MEMCPY = 0,
    VIRTIO_TOY_OP_ADD_SCALAR = 1,
    VIRTIO_TOY_OP_SQUARE = 2,
};

struct VirtIOToy {
    VirtIODevice parent_obj;
    VirtQueue *vq;
};

#endif
