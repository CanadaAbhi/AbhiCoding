#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "gpu_ioctl_defs.h"   /* shared with userspace: GPU_IOC_SUBMIT value */

enum evt_type {
    EVT_SYSCALL_ENTER = 1, EVT_SYSCALL_EXIT,
    EVT_SUBMIT_ENTER, EVT_RING_PUSH, EVT_GPU_START, EVT_GPU_COMPLETE,
    EVT_IH_TASKLET, EVT_FENCE_SIGNAL,
    EVT_IRQ_ENTRY, EVT_IRQ_EXIT, EVT_SOFTIRQ_ENTRY, EVT_SOFTIRQ_EXIT,
    EVT_SCHED_SWITCH, EVT_SCHED_WAKEUP,
    EVT_DMA_START, EVT_DMA_COMPLETE,
};

struct trace_event {
    u64 ts_ns; u32 cpu; u32 pid, tid; u16 type;
    u64 job_id; u64 arg1, arg2; char comm[16];
};

struct { __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 1 << 20); } events SEC(".maps");
/*
 * A single global BPF_MAP_TYPE_RINGBUF (5.8+), not N per-CPU perf
 * buffers. Per-CPU buffers give you N independently-ordered streams
 * that userspace must k-way merge-sort by timestamp after the fact
 * (exactly what perf script/libtraceevent do). A single ringbuf commits
 * events in the order they're produced across ALL CPUs, so the stream
 * arriving in userspace is already globally ordered -- the correctness-
 * critical property for a cross-core causal-latency tool like this one.
 * The tradeoff is more contention on one buffer; acceptable at the event
 * rates here (thousands/sec, not millions).
 */

static __always_inline struct trace_event *reserve(void)
{
    struct trace_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return NULL;
    e->ts_ns = bpf_ktime_get_ns();
    /* == CLOCK_MONOTONIC. Userspace MUST timestamp with
     * clock_gettime(CLOCK_MONOTONIC, ...) -- not gettimeofday(), not
     * CLOCK_REALTIME/BOOTTIME -- or every latency number below silently
     * corrupts across the kernel/userspace boundary. This is the single
     * most common bug in homegrown tracing tools and worth stating
     * unprompted in an interview. */
    e->cpu = bpf_get_smp_processor_id();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->tid = (u32)bpf_get_current_pid_tgid();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    return e;
}

/* --- Our own tracepoints. Auto-exposed at
 * /sys/kernel/debug/tracing/events/gpu_drv/*; libbpf attaches to them
 * with the identical tp/<system>/<event> mechanism used for kernel-
 * native tracepoints below -- there's no special-case API for "custom"
 * vs "built-in," which is the entire point of using TRACE_EVENT. */
SEC("tp/gpu_drv/gpu_ioctl_submit_enter")
int on_submit_enter(struct trace_event_raw_gpu_ioctl_submit_enter *ctx)
{
    struct trace_event *e = reserve();
    if (!e) return 0;
    e->type = EVT_SUBMIT_ENTER; e->job_id = ctx->job_id;
    e->arg1 = ctx->ctx_id; e->arg2 = ctx->ring_idx;
    bpf_ringbuf_submit(e, 0);
    return 0;
}
SEC("tp/gpu_drv/gpu_ring_push")
int on_ring_push(struct trace_event_raw_gpu_ring_push *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_RING_PUSH; e->job_id = ctx->job_id; e->arg1 = ctx->wptr;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/gpu_drv/gpu_job_start")
int on_job_start(struct trace_event_raw_gpu_job_start *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_GPU_START; e->job_id = ctx->job_id; e->arg1 = ctx->engine_id;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/gpu_drv/gpu_job_complete")
int on_job_complete(struct trace_event_raw_gpu_job_complete *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_GPU_COMPLETE; e->job_id = ctx->job_id;
    e->arg1 = ctx->engine_id; e->arg2 = ctx->duration_us;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/gpu_drv/gpu_ih_tasklet")
int on_ih_tasklet(struct trace_event_raw_gpu_ih_tasklet *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_IH_TASKLET; e->job_id = ctx->job_id;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/gpu_drv/gpu_fence_signal")
int on_fence_signal(struct trace_event_raw_gpu_fence_signal *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_FENCE_SIGNAL; e->job_id = ctx->job_id; e->arg1 = ctx->error;
    bpf_ringbuf_submit(e, 0); return 0;
}

/* --- Real kernel-native tracepoints: genuine hardirq/softirq context
 * around the IH tasklet -- the interrupt itself is simulated, but the
 * kernel's own IRQ subsystem processing it is not. */
SEC("tp/irq/irq_handler_entry")
int on_irq_entry(struct trace_event_raw_irq_handler_entry *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_IRQ_ENTRY; e->arg1 = ctx->irq;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/irq/irq_handler_exit")
int on_irq_exit(struct trace_event_raw_irq_handler_exit *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_IRQ_EXIT; e->arg1 = ctx->irq; e->arg2 = ctx->ret;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/irq/softirq_entry")
int on_softirq_entry(struct trace_event_raw_softirq *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SOFTIRQ_ENTRY; e->arg1 = ctx->vec;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/irq/softirq_exit")
int on_softirq_exit(struct trace_event_raw_softirq *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SOFTIRQ_EXIT; e->arg1 = ctx->vec;
    bpf_ringbuf_submit(e, 0); return 0;
}

/* --- sched_switch/sched_wakeup via tp_btf (CO-RE, BTF-relocated struct
 * task_struct access) -- the wake-latency half of the pipeline: fence
 * signal -> waiter woken -> waiter actually scheduled onto a CPU. */
SEC("tp_btf/sched_switch")
int BPF_PROG(on_sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SCHED_SWITCH; e->arg1 = prev->pid; e->arg2 = next->pid;
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp_btf/sched_wakeup")
int BPF_PROG(on_sched_wakeup, struct task_struct *p)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SCHED_WAKEUP; e->arg1 = p->pid;
    bpf_ringbuf_submit(e, 0); return 0;
}

/* --- syscall boundary, filtered to our device's SUBMIT ioctl cmd so the
 * ring buffer isn't drowned in unrelated system-wide ioctl traffic.
 * (WAIT_FENCE's cmd value would be added the same way to close the loop
 * on the far end explicitly rather than inferring it from sched events.) */
SEC("tp/syscalls/sys_enter_ioctl")
int on_sys_enter_ioctl(struct trace_event_raw_sys_enter *ctx)
{
    if (ctx->args[1] != GPU_IOC_SUBMIT && ctx->args[1] != GPU_IOC_WAIT_FENCE) return 0;
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SYSCALL_ENTER; e->arg1 = ctx->args[1];
    bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tp/syscalls/sys_exit_ioctl")
int on_sys_exit_ioctl(struct trace_event_raw_sys_exit *ctx)
{
    struct trace_event *e = reserve(); if (!e) return 0;
    e->type = EVT_SYSCALL_EXIT; e->arg1 = ctx->ret;
    bpf_ringbuf_submit(e, 0); return 0;
}

char LICENSE[] SEC("license") = "GPL";
