// kmem_lab_drv.c -- experimental kernel memory-manager driver. Allocates via
// four different kernel allocator paths and mmaps each one into userspace
// through a single misc device, using vma->vm_pgoff to carry an opaque
// buffer "handle" the way DRM GEM / V4L2 mmap offset schemes do. This lets
// one /dev/kmem_lab file support many independently-mmapped buffers instead
// of needing one fd per allocation.
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/idr.h>
#include <linux/uaccess.h>
#include <linux/spinlock.h>
#include "kmem_lab_uapi.h"

#define MAX_ORDER_PAGES 10  /* alloc_pages cap: 2^10 * PAGE_SIZE = 4MB */

struct kmem_buf {
    enum kmem_type type;
    u32 flags;
    size_t size;         /* requested */
    size_t actual_size;  /* rounded/backing size */
    void *kaddr;          /* kmalloc/vmalloc/dma cpu pointer */
    struct page *pages;   /* alloc_pages head page (PAGES type) */
    unsigned int order;   /* PAGES type order */
    dma_addr_t dma_handle; /* DMA type */
};

static struct idr kmem_idr;
static DEFINE_SPINLOCK(kmem_lock);
static struct platform_device *kmem_pdev; /* fake device for dma_alloc_coherent */

/* ---- allocation backends ---- */

static int kmem_alloc_kmalloc(struct kmem_buf *b)
{
    /* kmalloc rounds up to the nearest slab size class (kmalloc-8 ... -8192,
     * then page-order for larger) -- this rounding IS the internal
     * fragmentation the benchmark measures via actual_size vs size. */
    b->kaddr = kmalloc(b->size, GFP_KERNEL);
    if (!b->kaddr)
        return -ENOMEM;
    b->actual_size = kmalloc_size_roundup(b->size);
    return 0;
}

static int kmem_alloc_pages(struct kmem_buf *b)
{
    unsigned int order = get_order(b->size);
    if (order > MAX_ORDER_PAGES)
        return -EINVAL;
    b->pages = alloc_pages(GFP_KERNEL | __GFP_ZERO, order);
    if (!b->pages)
        return -ENOMEM;
    b->order = order;
    b->actual_size = (1UL << order) * PAGE_SIZE;
    b->kaddr = page_address(b->pages);
    return 0;
}

static int kmem_alloc_vmalloc(struct kmem_buf *b)
{
    /* vmalloc_user() zeroes and marks pages VM_USERMAP so
     * remap_vmalloc_range() is legal later -- same helper used in
     * shm_ipc_lab's shm_ring.h. Physically scattered pages stitched into
     * one virtually-contiguous kernel mapping via the kernel page tables. */
    b->kaddr = vmalloc_user(b->size);
    if (!b->kaddr)
        return -ENOMEM;
    b->actual_size = PAGE_ALIGN(b->size);
    return 0;
}

static int kmem_alloc_dma(struct kmem_buf *b)
{
    b->kaddr = dma_alloc_coherent(&kmem_pdev->dev, b->size, &b->dma_handle,
                                   GFP_KERNEL);
    if (!b->kaddr)
        return -ENOMEM;
    b->actual_size = PAGE_ALIGN(b->size);
    return 0;
}

static void kmem_free_buf(struct kmem_buf *b)
{
    switch (b->type) {
    case KMEM_TYPE_KMALLOC:
        kfree(b->kaddr);
        break;
    case KMEM_TYPE_PAGES:
        __free_pages(b->pages, b->order);
        break;
    case KMEM_TYPE_VMALLOC:
        vfree(b->kaddr);
        break;
    case KMEM_TYPE_DMA:
        dma_free_coherent(&kmem_pdev->dev, b->actual_size, b->kaddr,
                           b->dma_handle);
        break;
    }
    kfree(b);
}

/* ---- ioctl ---- */

static long kmem_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    void __user *uarg = (void __user *)arg;

    if (cmd == KMEM_IOC_ALLOC) {
        struct kmem_alloc_req req;
        struct kmem_buf *b;
        int handle, ret;

        if (copy_from_user(&req, uarg, sizeof(req)))
            return -EFAULT;
        if (req.size == 0 || req.size > (16UL << 20))
            return -EINVAL;

        b = kzalloc(sizeof(*b), GFP_KERNEL);
        if (!b)
            return -ENOMEM;
        b->type = req.type;
        b->flags = req.flags;
        b->size = req.size;

        switch (req.type) {
        case KMEM_TYPE_KMALLOC: ret = kmem_alloc_kmalloc(b); break;
        case KMEM_TYPE_PAGES:   ret = kmem_alloc_pages(b);   break;
        case KMEM_TYPE_VMALLOC: ret = kmem_alloc_vmalloc(b); break;
        case KMEM_TYPE_DMA:     ret = kmem_alloc_dma(b);     break;
        default: ret = -EINVAL;
        }
        if (ret) { kfree(b); return ret; }

        spin_lock(&kmem_lock);
        handle = idr_alloc(&kmem_idr, b, 1, 0, GFP_ATOMIC);
        spin_unlock(&kmem_lock);
        if (handle < 0) { kmem_free_buf(b); return handle; }

        req.handle = handle;
        req.actual_size = b->actual_size;
        if (copy_to_user(uarg, &req, sizeof(req)))
            return -EFAULT;
        return 0;

    } else if (cmd == KMEM_IOC_FREE) {
        struct kmem_free_req req;
        struct kmem_buf *b;

        if (copy_from_user(&req, uarg, sizeof(req)))
            return -EFAULT;

        spin_lock(&kmem_lock);
        b = idr_remove(&kmem_idr, (int)req.handle);
        spin_unlock(&kmem_lock);
        if (!b)
            return -ENOENT;
        kmem_free_buf(b);
        return 0;

    } else if (cmd == KMEM_IOC_GETINFO) {
        struct kmem_info_req req;
        struct kmem_buf *b;

        if (copy_from_user(&req, uarg, sizeof(req)))
            return -EFAULT;

        spin_lock(&kmem_lock);
        b = idr_find(&kmem_idr, (int)req.handle);
        spin_unlock(&kmem_lock);
        if (!b)
            return -ENOENT;

        req.type = b->type;
        req.size = b->size;
        req.actual_size = b->actual_size;
        req.phys_addr = (b->type == KMEM_TYPE_PAGES) ?
                         page_to_phys(b->pages) :
                         (b->type == KMEM_TYPE_KMALLOC) ?
                         virt_to_phys(b->kaddr) : 0;
        req.dma_addr = (b->type == KMEM_TYPE_DMA) ? b->dma_handle : 0;

        if (copy_to_user(uarg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }
    return -ENOTTY;
}

/* ---- mmap: dispatch by handle encoded in vm_pgoff ---- */

static int kmem_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long handle = vma->vm_pgoff; /* mmap offset param, shifted by caller */
    struct kmem_buf *b;
    size_t len = vma->vm_end - vma->vm_start;
    int ret;

    spin_lock(&kmem_lock);
    b = idr_find(&kmem_idr, (int)handle);
    spin_unlock(&kmem_lock);
    if (!b)
        return -ENOENT;
    if (len > b->actual_size)
        return -EINVAL;

    vma->vm_pgoff = 0; /* real APIs below expect 0-based offset */

    switch (b->type) {
    case KMEM_TYPE_KMALLOC:
    case KMEM_TYPE_PAGES: {
        unsigned long pfn = (b->type == KMEM_TYPE_PAGES) ?
                             page_to_pfn(b->pages) :
                             virt_to_phys(b->kaddr) >> PAGE_SHIFT;
        pgprot_t prot = vma->vm_page_prot;

        if (b->flags & KMEM_FLAG_UNCACHED)
            prot = pgprot_noncached(prot); /* cache-behavior knob: forces every
                                             * access to bypass CPU cache and hit
                                             * memory directly -- the throughput
                                             * delta vs cached PAGES is the
                                             * "cache performance" benchmark. */
        vma->vm_page_prot = prot;
        ret = remap_pfn_range(vma, vma->vm_start, pfn, len, prot);
        break;
    }
    case KMEM_TYPE_VMALLOC:
        /* stitches the vmalloc'd (physically scattered) pages into the
         * process's page tables one PFN at a time under the hood. */
        ret = remap_vmalloc_range(vma, b->kaddr, 0);
        break;
    case KMEM_TYPE_DMA:
        ret = dma_mmap_coherent(&kmem_pdev->dev, vma, b->kaddr,
                                 b->dma_handle, len);
        break;
    default:
        ret = -EINVAL;
    }
    return ret;
}

static const struct file_operations kmem_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = kmem_ioctl,
    .mmap = kmem_mmap,
};

static struct miscdevice kmem_miscdev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "kmem_lab",
    .fops = &kmem_fops,
};

static int __init kmem_lab_init(void)
{
    int ret;

    idr_init(&kmem_idr);

    /* Fake platform_device purely to have a struct device with a DMA mask
     * for dma_alloc_coherent -- same pattern as accel_lab/pcie_dma_drv. */
    kmem_pdev = platform_device_register_simple("kmem_lab_dma", -1, NULL, 0);
    if (IS_ERR(kmem_pdev))
        return PTR_ERR(kmem_pdev);
    dma_set_mask_and_coherent(&kmem_pdev->dev, DMA_BIT_MASK(32));

    ret = misc_register(&kmem_miscdev);
    if (ret)
        platform_device_unregister(kmem_pdev);
    return ret;
}

static void __exit kmem_lab_exit(void)
{
    misc_deregister(&kmem_miscdev);
    platform_device_unregister(kmem_pdev);
    idr_destroy(&kmem_idr);
}

module_init(kmem_lab_init);
module_exit(kmem_lab_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Experimental kernel memory manager: kmalloc/pages/vmalloc/DMA + mmap");
