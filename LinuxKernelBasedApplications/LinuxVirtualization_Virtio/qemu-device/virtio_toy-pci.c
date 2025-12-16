// virtio-toy-pci.c -- wraps VirtIOToy in the virtio-pci transport, exactly
// like virtio-net-pci/virtio-blk-pci do for the stock devices. This is what
// makes the device show up on the guest's PCI bus, get a PCI BAR for
// virtio-pci-modern (MMIO notify/common-cfg regions), and get an MSI-X
// vector for interrupt injection -- the "PCIe" layer of the stack.
#include "qemu/osdep.h"
#include "virtio-toy.h"
#include "hw/virtio/virtio-pci.h"

typedef struct VirtIOToyPCI {
    VirtIOPCIProxy parent_obj;
    VirtIOToy vdev;
} VirtIOToyPCI;

#define TYPE_VIRTIO_TOY_PCI "virtio-toy-pci"
DECLARE_INSTANCE_CHECKER(VirtIOToyPCI, VIRTIO_TOY_PCI, TYPE_VIRTIO_TOY_PCI)

static void virtio_toy_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)
{
    VirtIOToyPCI *dev = VIRTIO_TOY_PCI(vpci_dev);
    DeviceState *vdev = DEVICE(&dev->vdev);

    vpci_dev->class_code = PCI_CLASS_OTHERS;
    qdev_realize(vdev, BUS(&vpci_dev->bus), errp);
}

static void virtio_toy_pci_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    VirtioPCIClass *k = VIRTIO_PCI_CLASS(klass);
    PCIDeviceClass *pcidev_k = PCI_DEVICE_CLASS(klass);

    k->realize = virtio_toy_pci_realize;
    pcidev_k->vendor_id = 0x1AF4; /* Red Hat / virtio vendor ID */
    pcidev_k->device_id = 0x10F0; /* unused id in the virtio range -- toy device */
    pcidev_k->revision = VIRTIO_PCI_ABI_VERSION;
    pcidev_k->class_id = PCI_CLASS_OTHERS;
    dc->hotpluggable = false;
}

static void virtio_toy_pci_instance_init(Object *obj)
{
    VirtIOToyPCI *dev = VIRTIO_TOY_PCI(obj);
    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev), TYPE_VIRTIO_TOY);
}

static const VirtioPCIDeviceTypeInfo virtio_toy_pci_info = {
    .base_name = TYPE_VIRTIO_TOY_PCI,
    .generic_name = "virtio-toy-pci",
    .instance_size = sizeof(VirtIOToyPCI),
    .instance_init = virtio_toy_pci_instance_init,
    .class_init = virtio_toy_pci_class_init,
};

static void virtio_toy_pci_register(void)
{
    virtio_pci_types_register(&virtio_toy_pci_info);
}
type_init(virtio_toy_pci_register)
