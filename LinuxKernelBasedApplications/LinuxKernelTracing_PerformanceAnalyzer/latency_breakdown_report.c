static void emit_report(struct job_trace *j)
{
    printf("\n=== job %llu (ctx=%u engine=%u) ===\n", j->job_id, j->ctx_id, j->engine_id);
    printf("%-32s %8lld ns\n", "ioctl dispatch (enter->submit)",  (long long)(j->t_submit_enter  - j->t_syscall_enter));
    printf("%-32s %8lld ns\n", "validate+reserve (submit->push)", (long long)(j->t_ring_push      - j->t_submit_enter));
    printf("%-32s %8lld ns\n", "queueing (push->GPU start)",       (long long)(j->t_gpu_start      - j->t_ring_push));
    printf("%-32s %8lld ns\n", "GPU execution",                    (long long)(j->t_gpu_complete   - j->t_gpu_start));
    printf("%-32s %8lld ns\n", "IRQ assertion latency",             (long long)(j->t_irq_entry      - j->t_gpu_complete));
    printf("%-32s %8lld ns\n", "softirq dispatch latency",          (long long)(j->t_softirq_entry  - j->t_irq_entry));
    printf("%-32s %8lld ns\n", "IH processing + fence signal",      (long long)(j->t_fence_signal   - j->t_softirq_entry));
    printf("%-32s %8lld ns\n", "wake latency (signal->wakeup)",     (long long)(j->t_wakeup         - j->t_fence_signal));
    printf("%-32s %8lld ns\n", "runqueue wait (wakeup->sched-in)",  (long long)(j->t_sched_in       - j->t_wakeup));
    printf("%-32s %8lld ns\n", "return-to-user path",               (long long)(j->t_syscall_exit   - j->t_sched_in));
    printf("%-32s %8lld ns\n", "TOTAL end-to-end",                  (long long)(j->t_syscall_exit   - j->t_syscall_enter));
    render_timeline(j);
}
