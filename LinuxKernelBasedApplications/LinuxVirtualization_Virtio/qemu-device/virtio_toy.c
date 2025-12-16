// virtio-toy.c -- QEMU device implementation. Pops descriptor chains from
// the single virtqueue: [req header][input buf][output buf]. Performs the
// compute directly on the mapped guest memory (QEMU has the guest RAM
// mmap'd already -- this IS the "shared memory" path: no copy into a
// separate host buffer is required, virtqueue_pop() hands us iovecs that
// point straight into guest physical RAM via the VM's memory region).
#include "qemu/osdep.h"
#include "virtio-toy.h"
#include "hw/virtio/virtio-access.h"
#include "qemu/iov.h"
#include "qapi/error.h"

static void virtio_toy_do_op(struct virtio_toy_req *req,
                              const uint8_t *in, uint8_t *out)
{
    uint32_t *in32 = (uint32_t *)in;
    uint32_t *out32 = (uint32_t *)out;
    uint32_t i;

    switch (req->op) {
    case VIRTIO_TOY_OP_ADD_SCALAR:
        for (i = 0; i < req->nelem; i++)
            out32[i] = in32[i] + req->scalar;
        break;
    case VIRTIO_TOY_OP_SQUARE:
        for (i = 0; i < req->nelem; i++)
            out32[i] = in32[i] * in32[i];
        break;
    default: /* MEMCPY */
        memcpy(out, in, req->nelem * sizeof(uint32_t));
        break;
    }
    req->status = 0;
}

/* Called on guest kick (guest's virtqueue_kick() -> vmexit or ioeventfd
 * trap -> this handler runs in a QEMU AioContext, NOT the vCPU thread when
 * ioeventfd is wired -- that's the actual performance story of virtio). */
static void virtio_toy_handle_vq(VirtIODevice *vdev, VirtQueue *vq)
{
    VirtQueueElement *elem;

    while ((elem = virtqueue_pop(vq, sizeof(VirtQueueElement)))) {
        struct virtio_toy_req req;
        uint8_t *inbuf, *outbuf;
        size_t inlen, outlen;

        if (elem->out_num < 2 || elem->in_num < 1) {
            virtqueue_detach_element(vq, elem, 0);
            g_free(elem);
            continue;
        }

        /* out_sg[0] = req header (device reads), out_sg[1] = input data
         * in_sg[0] = output data (device writes) -- split-ring desc chain */
        iov_to_buf(elem->out_sg, 1, 0, &req, sizeof(req));
        inlen = elem->out_sg[1].iov_len;
        inbuf = elem->out_sg[1].iov_base;   /* direct pointer into guest RAM */
        outlen = elem->in_sg[0].iov_len;
        outbuf = elem->in_sg[0].iov_base;

        if (req.nelem * sizeof(uint32_t) <= inlen &&
            req.nelem * sizeof(uint32_t) <= outlen) {
            virtio_toy_do_op(&req, inbuf, outbuf);
        } else {
            req.status = -1;
        }

        /* write status back into the *input* side the guest also reads --
         * append a small status word after the output payload region, or
         * simpler: reuse req header space mapped read/write. Here we just
         * push length == outlen back to the used ring; guest reads status
         * via the same header buffer it mapped writable. */
        virtqueue_push(vq, elem, outlen);
        g_free(elem);
    }
    /* Injects the completion interrupt: MSI-X via irqfd if wired, else a
     * normal virtio_notify_config()-style vmexit-based injection. */
    virtio_notify(vdev, vq);
}

static void virtio_toy_device_realize(DeviceState *dev, Error **errp)
{
    VirtIODevice *vdev = VIRTIO_DEVICE(dev);
    VirtIOToy *toy = VIRTIO_TOY(dev);

    virtio_init(vdev, VIRTIO_ID_TOY, 0 /* no device-specific config space */);
    toy->vq = virtio_add_queue(vdev, 256, virtio_toy_handle_vq);
}

static void virtio_toy_device_unrealize(DeviceState *dev)
{
    VirtIODevice *vdev = VIRTIO_DEVICE(dev);
    virtio_del_queue(vdev, 0);
    virtio_cleanup(vdev);
}

static uint64_t virtio_toy_get_features(VirtIODevice *vdev, uint64_t f, Error **errp)
{
    return f; /* no extra feature bits for this toy device */
}

static const VMStateDescription vmstate_virtio_toy = {
    .name = "virtio-toy",
    .minimum_version_id = 1,
    .version_id = 1,
    .fields = (VMStateField[]) {
        VMSTATE_VIRTIO_DEVICE,
        VMSTATE_END_OF_LIST()
    },
};

static void virtio_toy_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->vmsd = &vmstate_virtio_toy;
    vdc->realize = virtio_toy_device_realize;
    vdc->unrealize = virtio_toy_device_unrealize;
    vdc->get_features = virtio_toy_get_features;
}

static const TypeInfo virtio_toy_info = {
    .name = TYPE_VIRTIO_TOY,
    .parent = TYPE_VIRTIO_DEVICE,
    .instance_size = sizeof(VirtIOToy),
    .class_init = virtio_toy_class_init,
};

static void virtio_toy_register_types(void)
{
    type_register_static(&virtio_toy_info);
}
type_init(virtio_toy_register_types)
