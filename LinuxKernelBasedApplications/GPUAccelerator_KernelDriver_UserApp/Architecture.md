      accel_app.c (userspace)
         Buffer input, output;
         submit_job(input, output);
         wait_for_completion();
                |
              ioctl(/dev/accel0)
                v
+--------------------------------------------------------+
|                    accel_drv.ko                        |
|                                                          |
|  Buffer Manager        Command Queue       Scheduler     |
|  accel_bo.c            accel_sched.c       accel_sched.c |
|  idr handle table  ->  spinlock job list -> kthread loop |
|  dma_alloc_coherent    (struct accel_job)  dequeue->submit|
|  dma_mmap_coherent          |                    |        |
|         |                   +--------------------+        |
|         |                              |                    |
|  Fence                         Interrupt                    |
|  accel_fence.c                 accel_hwsim.c                |
|  dma_fence + sync_file   <---  hrtimer "hard-IRQ" ->        |
|  idr job-id table              workqueue "threaded" tail    |
|                                       |                       |
+---------------------------------------|-----------------------+
                                        v
                     +------------------------------------+
                     |   Accelerator Simulator (pluggable) |
                     |  (A) accel_hwsim.c software model   |
                     |  (B) QEMU PCI device (BAR0/MSI-X,   |
                     |      pcie_dma_drv.c pattern)         |
                     |  (C) FPGA (AXI-Lite + AXI-DMA IP,    |
                     |      same accel_hw_ops swap)          |
                     +------------------------------------+


accel_drv/
  include/accel_uapi.h
  drv/accel_priv.h
  drv/accel_core.c
  drv/accel_bufmgr.c
  drv/accel_cmdq.c
  drv/accel_sched.c
  drv/accel_fence.c
  drv/accel_irq.c
  sim/accel_hw_sim.c
  qemu/accel_qemu_dev.c        (skeleton, alternate backend)
  user/accel_lib.h
  user/accel_lib.c
  user/app.c
  Makefile
