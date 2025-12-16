struct job_trace {
    u64 job_id; bool valid; u32 waiter_tid, ctx_id, engine_id; int fence_error;
    u64 t_syscall_enter, t_submit_enter, t_ring_push, t_gpu_start, t_gpu_complete,
        t_irq_entry, t_softirq_entry, t_ih_tasklet, t_fence_signal,
        t_wakeup, t_sched_in, t_syscall_exit, t_dma_start, t_dma_complete;
};

/* Small fixed-size open-addressed table keyed by job_id -- a full hash
 * map library is overkill at the job counts this demo runs; called out
 * as the first thing to swap for something like uthash/khash if this
 * were tracing a production workload with sustained high submission
 * rates. */
#define JOB_TABLE_SIZE 4096
static struct job_trace g_jobs[JOB_TABLE_SIZE];
static u32 g_pending_tid_to_slot[65536];   /* tid -> in-flight job slot, bridges
                                               EVT_SYSCALL_ENTER (job_id unknown yet)
                                               to EVT_SUBMIT_ENTER (job_id known) */

static struct job_trace *job_slot(u64 job_id, bool create)
{
    u32 h = job_id % JOB_TABLE_SIZE;
    for (u32 i = 0; i < JOB_TABLE_SIZE; i++) {
        u32 idx = (h + i) % JOB_TABLE_SIZE;
        if (g_jobs[idx].valid && g_jobs[idx].job_id == job_id) return &g_jobs[idx];
        if (!g_jobs[idx].valid && create) { g_jobs[idx].valid = true; g_jobs[idx].job_id = job_id; return &g_jobs[idx]; }
    }
    return NULL;
}

static void handle_event(struct trace_event *e)
{
    switch (e->type) {
    case EVT_SYSCALL_ENTER:
        g_pending_tid_to_slot[e->tid] = e->ts_ns;   /* stash ts; job_id not yet known */
        break;
    case EVT_SUBMIT_ENTER: {
        struct job_trace *j = job_slot(e->job_id, true);
        j->ctx_id = e->arg1;
        j->t_syscall_enter = g_pending_tid_to_slot[e->tid];
        j->t_submit_enter = e->ts_ns;
        j->waiter_tid = e->tid;    /* demo assumes submit+wait same thread; noted limitation below */
        break;
    }
    case EVT_RING_PUSH:      job_slot(e->job_id, false)->t_ring_push     = e->ts_ns; break;
    case EVT_GPU_START:      job_slot(e->job_id, false)->t_gpu_start     = e->ts_ns; break;
    case EVT_GPU_COMPLETE: { struct job_trace *j = job_slot(e->job_id, false);
                              j->t_gpu_complete = e->ts_ns; j->engine_id = e->arg1; break; }
    case EVT_IH_TASKLET:     job_slot(e->job_id, false)->t_ih_tasklet    = e->ts_ns; break;
    case EVT_FENCE_SIGNAL: { struct job_trace *j = job_slot(e->job_id, false);
                              j->t_fence_signal = e->ts_ns; j->fence_error = e->arg1; break; }

    /* IRQ/softirq aren't job-id tagged (they're generic kernel events),
     * so attribute them to whichever job is currently "in flight" and
     * hasn't reached fence_signal yet -- correct for this demo's
     * single-job-at-a-time GPU thread; a multi-engine/multi-inflight
     * trace would instead correlate by CPU + tight time window, same
     * technique perf uses to attribute IRQs to workloads. */
    case EVT_IRQ_ENTRY:      most_recent_open_job()->t_irq_entry     = e->ts_ns; break;
    case EVT_SOFTIRQ_ENTRY:  most_recent_open_job()->t_softirq_entry = e->ts_ns; break;

    case EVT_SCHED_WAKEUP:
        for_each_open_job(j) if (j->waiter_tid == e->arg1 && j->t_fence_signal && !j->t_wakeup)
            j->t_wakeup = e->ts_ns;
        break;
    case EVT_SCHED_SWITCH:
        for_each_open_job(j) if (j->waiter_tid == e->arg2 && j->t_wakeup && !j->t_sched_in)
            j->t_sched_in = e->ts_ns;
        break;
    case EVT_SYSCALL_EXIT:
        for_each_open_job(j) if (j->waiter_tid == e->tid && j->t_sched_in && !j->t_syscall_exit) {
            j->t_syscall_exit = e->ts_ns;
            emit_report(j);         /* job's full lifecycle is now complete */
        }
        break;
    }
}
