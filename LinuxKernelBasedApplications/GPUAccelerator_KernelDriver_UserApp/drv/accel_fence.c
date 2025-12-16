// accel_fence.c -- per-job dma_fence tracking. Each submitted job gets a
// unique, monotonically increasing seqno and a dma_fence; the job struct is
// tracked in inflight_idr keyed by seqno so the IRQ handler can find and
// signal the right fence when the simulator reports completion.
#include "accel_priv.h"

static const char *accel_fence_get_driver_name(struct dma_fence *f) { return "accel_drv"; }
static const char *accel_fence_get_timeline_name(struct dma_fence *f) { return "accel-engine0"; }

static const struct dma_fence_ops accel_fence_ops = {
	.get_driver_name = accel_fence_get_driver_name,
	.get_timeline_name = accel_fence_get_timeline_name,
};

int accel_fence_init(struct accel_dev *dev)
{
	dev->fence_context = dma_fence_context_alloc(1);
	atomic64_set(&dev->seqno_counter, 0);
	idr_init(&dev->inflight_idr);
	spin_lock_init(&dev->inflight_lock);
	return 0;
}

struct dma_fence *accel_fence_create(struct accel_dev *dev, u64 *out_seqno)
{
	struct dma_fence *f = kzalloc(sizeof(*f), GFP_KERNEL);
	u64 seqno = atomic64_inc_return(&dev->seqno_counter);

	if (!f) return NULL;
	dma_fence_init(f, &accel_fence_ops, &dev->inflight_lock, dev->fence_context, seqno);
	*out_seqno = seqno;
	return f;
}

void accel_fence_signal_seqno(struct accel_dev *dev, u64 seqno, int error)
{
	struct accel_job *job;

	spin_lock(&dev->inflight_lock);
	job = idr_remove(&dev->inflight_idr, seqno);
	spin_unlock(&dev->inflight_lock);
	if (!job) return;

	if (error) dma_fence_set_error(job->fence, error);
	dma_fence_signal(job->fence);

	spin_lock(&dev->stats_lock);
	if (!error) {
		u64 lat = ktime_to_ns(ktime_sub(ktime_get(), job->submit_ts));
		dev->stats.jobs_completed++;
		dev->stats.latency_ns_sum += lat;
		dev->stats.latency_ns_count++;
		if (!dev->stats.latency_ns_min || lat < dev->stats.latency_ns_min)
			dev->stats.latency_ns_min = lat;
		if (lat > dev->stats.latency_ns_max)
			dev->stats.latency_ns_max = lat;
	}
	spin_unlock(&dev->stats_lock);

	accel_buf_put(job->inbuf);
	accel_buf_put(job->outbuf);
	dma_fence_put(job->fence);
	kfree(job);
}

int accel_fence_wait(struct accel_dev *dev, u64 seqno, u64 timeout_ns)
{
	/* find by scanning is O(n) but inflight depth is bounded by cmdq depth;
	 * acceptable for a lab. A production driver would keep a small hash. */
	struct accel_job *job;
	struct dma_fence *f = NULL;
	long ret;

	spin_lock(&dev->inflight_lock);
	job = idr_find(&dev->inflight_idr, seqno);
	if (job) f = dma_fence_get(job->fence);
	spin_unlock(&dev->inflight_lock);

	if (!f) return 0; /* already signaled and reaped */

	ret = dma_fence_wait_timeout(f, true, nsecs_to_jiffies(timeout_ns));
	dma_fence_put(f);
	if (ret == 0) return -ETIME;
	if (ret < 0) return ret;
	return 0;
}
