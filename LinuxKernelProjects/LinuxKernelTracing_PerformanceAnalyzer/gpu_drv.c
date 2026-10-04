+#include "gpu_drv_trace.h"

 static long gpu_ioctl_submit(struct gpu_ctx *ctx, struct gpu_submit __user *uarg)
 {
     ...
     job->id = atomic64_fetch_inc(&g_next_job_id);
+    trace_gpu_ioctl_submit_enter(ctx->id, job->id, args.ring_idx);

     if (validate_cmd_stream(job) < 0) return -EINVAL;   /* SET_CTX/FENCE injection guard */

     ring_slot = ring_reserve(&ctx->ring, job);
+    trace_gpu_ring_push(job->id, ring_wptr(&ctx->ring));
     wake_up(&ctx->ring.wq);
     ...
 }

 /* fake GPU consumer kthread */
 static int gpu_consumer_thread(void *arg)
 {
     ...
     job = ring_pop(&engine->ring);
+    trace_gpu_job_start(job->id, engine->id);
     usleep_range(job->duration_us, job->duration_us + 50);
+    trace_gpu_job_complete(job->id, engine->id, job->duration_us);
     assert_fake_irq(engine);   /* generic_handle_irq() on our fake IRQ line */
     ...
 }

 /* IH tasklet, softirq context */
 static void gpu_ih_tasklet_fn(unsigned long data)
 {
     struct gpu_job *job = ...;
+    trace_gpu_ih_tasklet(job->id);
     dma_fence_signal(&job->base);
+    trace_gpu_fence_signal(job->id, job->base.error);
 }
