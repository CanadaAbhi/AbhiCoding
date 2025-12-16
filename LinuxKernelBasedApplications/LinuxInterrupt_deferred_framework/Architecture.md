 hw_event_gen.ko                         irq_defer_lab.ko
 ================                        =================
 hrtimer (event_hz)                       request_irq() / request_threaded_irq()
       |                                          |
       v                                          v
 generic_handle_irq(virq)  ---[real Linux IRQ]---> Top Half (hard-irq context)
 (fake irqchip / irq_domain,                       - record ts_top
  no physical hardware needed)                     - hweg_ack()  <-- "acknowledge hardware"
                                                    - dispatch to deferred mechanism
                                                             |
                     +----------------------+----------------------+
                     |                      |                      |
               MODE_WORKQUEUE          MODE_TASKLET          MODE_THREADED_IRQ
               queue_work()            tasklet_schedule()    IRQ thread (sleeps)
                     |                      |                      |
              workqueue fn            tasklet fn (softirq)   thread hands off to
              heavy_processing()      heavy_processing()     ANOTHER workqueue ->
              (may sleep)             (must NOT sleep!)       heavy_processing()
                     |                      |                      |
                     +----------------------+----------------------+
                                            |
                                    push record to kfifo
                                    wake_up_interruptible(wq)
                                            |
                                            v
                              chardev poll()/read() -> irq_defer_app
