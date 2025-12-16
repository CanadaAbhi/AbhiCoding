// vfio_dma_demo.c -- minimal VFIO group/container setup + DMA map, showing
// the path virtio bypasses entirely: here the device is REAL (or a QEMU
// vfio-pci-assigned device), mapped directly into the guest's IOVA space.
#include <linux/vfio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    int container = open("/dev/vfio/vfio", O_RDWR);
    int group = open("/dev/vfio/1", O_RDWR); /* IOMMU group of target device */

    ioctl(group, VFIO_GROUP_SET_CONTAINER, &container);
    ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU);

    int device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, "0000:00:05.0");

    /* Map a chunk of process memory into the device's IOVA space --
     * conceptually identical to your pcie_dma_drv's dma_map_single, but
     * done from userspace and backed by the real IOMMU hardware instead
     * of dma_alloc_coherent. */
    void *buf = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    struct vfio_iommu_type1_dma_map map = {
        .argsz = sizeof(map),
        .vaddr = (__u64)buf,
        .iova = 0x100000,
        .size = 4096,
        .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
    };
    ioctl(container, VFIO_IOMMU_MAP_DMA, &map);

    /* BAR0 mmap for direct MMIO access -- near-native register latency,
     * bypassing QEMU's virtio device-model emulation entirely. */
    struct vfio_region_info reg = { .argsz = sizeof(reg), .index = 0 };
    ioctl(device, VFIO_DEVICE_GET_REGION_INFO, &reg);
    void *bar0 = mmap(NULL, reg.size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, device, reg.offset);

    printf("BAR0 mapped at %p, size=%llu, IOVA 0x%llx -> HVA %p\n",
           bar0, reg.size, map.iova, buf);
    return 0;
}
