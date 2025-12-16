// accel_bufmgr.c -- allocates DMA-coherent buffers, tracks them in an idr
// keyed by an opaque handle returned to userspace, and services mmap() so
// the app gets a zero-copy CPU pointer onto the same memory the simulator's
// "hardware" reads/writes directly.
#include "accel_priv.h"
#include <linux/slab.h>
#include <linux/mm.h>

static void accel_buf_release(struct kref *kref)
{
	struct accel_buffer *buf = container_of(kref, struct accel_buffer, refcount);
	dma_free_coherent(&buf->dev->pdev->dev, buf->size, buf->cpu_addr, buf->dma_addr);
	kfree(buf);
}

void accel_buf_put(struct accel_buffer *buf)
{
	kref_put(&buf->refcount, accel_buf_release);
}

struct accel_buffer *accel_buf_alloc(struct accel_dev *dev, u64 size, u32 *out_handle)
{
	struct accel_buffer *buf = kzalloc(sizeof(*buf), GFP_KERNEL);
	int id;

	if (!buf) return NULL;
	buf->cpu_addr = dma_alloc_coherent(&dev->pdev->dev, size, &buf->dma_addr, GFP_KERNEL);
	if (!buf->cpu_addr) { kfree(buf); return NULL; }

	buf->size = size;
	buf->dev = dev;
	kref_init(&buf->refcount);

	spin_lock(&dev->buf_lock);
	id = idr_alloc(&dev->buf_idr, buf, 1, 0, GFP_ATOMIC);
	spin_unlock(&dev->buf_lock);
	if (id < 0) {
		dma_free_coherent(&dev->pdev->dev, size, buf->cpu_addr, buf->dma_addr);
		kfree(buf);
		return NULL;
	}
	buf->handle = id;
	*out_handle = id;
	return buf;
}

struct accel_buffer *accel_buf_get_by_handle(struct accel_dev *dev, u32 handle)
{
	struct accel_buffer *buf;
	spin_lock(&dev->buf_lock);
	buf = idr_find(&dev->buf_idr, handle);
	if (buf) kref_get(&buf->refcount);
	spin_unlock(&dev->buf_lock);
	return buf;
}

int accel_buf_free_handle(struct accel_dev *dev, u32 handle)
{
	struct accel_buffer *buf;
	spin_lock(&dev->buf_lock);
	buf = idr_remove(&dev->buf_idr, handle);
	spin_unlock(&dev->buf_lock);
	if (!buf) return -ENOENT;
	accel_buf_put(buf); /* drop the idr's reference */
	return 0;
}

int accel_buf_mmap(struct accel_dev *dev, struct vm_area_struct *vma)
{
	u64 offset = vma->vm_pgoff << PAGE_SHIFT;
	u32 handle = offset / ACCEL_MMAP_UNIT;
	struct accel_buffer *buf = accel_buf_get_by_handle(dev, handle);
	int ret;

	if (!buf) return -EINVAL;
	ret = dma_mmap_coherent(&dev->pdev->dev, vma, buf->cpu_addr, buf->dma_addr, buf->size);
	accel_buf_put(buf); /* mapping doesn't need to hold the extra ref */
	return ret;
}
