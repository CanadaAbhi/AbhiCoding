#undef TRACE_SYSTEM
#define TRACE_SYSTEM gpu_drv

#if !defined(_TRACE_GPU_DRV_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_GPU_DRV_H
#include <linux/tracepoint.h>

/*
 * TRACE_EVENT, not printk: (1) near-zero overhead when disabled -- a
 * jump-label NOP at each callsite, not a branch + format-string cost;
 * (2) structured binary fields, filterable from ftrace/perf/eBPF without
 * string parsing; (3) a stable attach point other tools (this project's
 * own collector, bpftrace, perf script) can consume without coordinating
 * with driver internals. This is exactly the mechanism behind
 * amdgpu_trace.h, drm_trace.h, xhci-trace.h.
 */

TRACE_EVENT(gpu_ioctl_submit_enter,
    TP_PROTO(u32 ctx_id, u64 job_id, u32 ring_idx),
    TP_ARGS(ctx_id, job_id, ring_idx),
    TP_STRUCT__entry(__field(u32, ctx_id) __field(u64, job_id) __field(u32, ring_idx)),
    TP_fast_assign(__entry->ctx_id = ctx_id; __entry->job_id = job_id; __entry->ring_idx = ring_idx;),
    TP_printk("ctx=%u job=%llu ring=%u", __entry->ctx_id, __entry->job_id, __entry->ring_idx)
);

TRACE_EVENT(gpu_ring_push,
    TP_PROTO(u64 job_id, u32 wptr),
    TP_ARGS(job_id, wptr),
    TP_STRUCT__entry(__field(u64, job_id) __field(u32, wptr)),
    TP_fast_assign(__entry->job_id = job_id; __entry->wptr = wptr;),
    TP_printk("job=%llu wptr=%u", __entry->job_id, __entry->wptr)
);

TRACE_EVENT(gpu_job_start,
    TP_PROTO(u64 job_id, u32 engine_id),
    TP_ARGS(job_id, engine_id),
    TP_STRUCT__entry(__field(u64, job_id) __field(u32, engine_id)),
    TP_fast_assign(__entry->job_id = job_id; __entry->engine_id = engine_id;),
    TP_printk("job=%llu engine=%u", __entry->job_id, __entry->engine_id)
);

TRACE_EVENT(gpu_job_complete,
    TP_PROTO(u64 job_id, u32 engine_id, u32 duration_us),
    TP_ARGS(job_id, engine_id, duration_us),
    TP_STRUCT__entry(__field(u64, job_id) __field(u32, engine_id) __field(u32, duration_us)),
    TP_fast_assign(__entry->job_id = job_id; __entry->engine_id = engine_id; __entry->duration_us = duration_us;),
    TP_printk("job=%llu engine=%u dur=%uus", __entry->job_id, __entry->engine_id, __entry->duration_us)
);

TRACE_EVENT(gpu_ih_tasklet,
    TP_PROTO(u64 job_id),
    TP_ARGS(job_id),
    TP_STRUCT__entry(__field(u64, job_id)),
    TP_fast_assign(__entry->job_id = job_id;),
    TP_printk("job=%llu", __entry->job_id)
);

TRACE_EVENT(gpu_fence_signal,
    TP_PROTO(u64 job_id, int error),
    TP_ARGS(job_id, error),
    TP_STRUCT__entry(__field(u64, job_id) __field(int, error)),
    TP_fast_assign(__entry->job_id = job_id; __entry->error = error;),
    TP_printk("job=%llu err=%d", __entry->job_id, __entry->error)
);

#endif /* _TRACE_GPU_DRV_H */
#include <trace/define_trace.h>
