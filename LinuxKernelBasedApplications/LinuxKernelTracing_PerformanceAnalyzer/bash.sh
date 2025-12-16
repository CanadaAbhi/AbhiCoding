# ftrace, zero custom code, good for a first-pass sanity check
trace-cmd record -e 'gpu_drv:*' -e 'irq:*' -e 'sched:sched_switch' -e 'sched:sched_wakeup' \
                 -e 'syscalls:sys_enter_ioctl' -e 'syscalls:sys_exit_ioctl' -- ./submit_bench
trace-cmd report | less           # or: kernelshark trace.dat  for the GUI timeline

# perf, when hardware-counter correlation matters too (cache misses during IH?)
perf record -e gpu_drv:gpu_fence_signal -e sched:sched_switch -e cache-misses -a -- ./submit_bench
perf script

# bpftrace, one-liner ad hoc probing without writing a .bpf.c at all
bpftrace -e 'tracepoint:gpu_drv:gpu_fence_signal { printf("job=%llu err=%d\n", args->job_id, args->error); }'
