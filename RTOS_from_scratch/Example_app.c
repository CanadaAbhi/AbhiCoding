/**
 * @file example_app.c
 * @brief Comprehensive RTOS Example Application
 * 
 * Demonstrates:
 * - Task creation with different priorities
 * - Semaphores for synchronization
 * - Mutexes with priority inheritance
 * - Message queues
 * - Timers
 * - Interrupt handling
 */

 #include "rtos.h"
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdarg.h>
 
 /* ==================== Shared Resources ==================== */
 static semaphore_t *data_ready_sem;
 static mutex_t *console_mutex;
 static message_queue_t *msg_queue;
 static timer_t *periodic_timer;
 
 static int shared_counter = 0;
 
 /* ==================== Helper Function ==================== */
 void safe_printf(const char *format, ...) {
     rtos_mutex_lock(console_mutex, UINT32_MAX);
     va_list args;
     va_start(args, format);
     vprintf(format, args);
     va_end(args);
     rtos_mutex_unlock(console_mutex);
 }
 
 /* ==================== Example Tasks ==================== */
 
 /**
  * High Priority Producer Task
  * Demonstrates: High priority task, semaphore signaling
  */
 void high_priority_producer(void *arg) {
     int count = 0;
     
     while (1) {
         safe_printf("[HIGH PRIORITY] Producing data %d\n", count);
         
         // Simulate work
         for (volatile int i = 0; i < 100000; i++);
         
         shared_counter++;
         count++;
         
         // Signal consumer
         rtos_semaphore_signal(data_ready_sem);
         
         // Send message to queue
         int *msg = malloc(sizeof(int));
         *msg = count;
         rtos_queue_send(msg_queue, msg, UINT32_MAX);
         
         rtos_task_delay(500); // Delay 500 ticks
     }
 }
 
 /**
  * Medium Priority Consumer Task
  * Demonstrates: Medium priority, semaphore waiting, mutex usage
  */
 void medium_priority_consumer(void *arg) {
     while (1) {
         safe_printf("[MEDIUM PRIORITY] Waiting for data...\n");
         
         // Wait for data
         rtos_semaphore_wait(data_ready_sem, UINT32_MAX);
         
         safe_printf("[MEDIUM PRIORITY] Consuming data, counter = %d\n", shared_counter);
         
         rtos_task_delay(300);
     }
 }
 
 /**
  * Low Priority Task
  * Demonstrates: Low priority task, priority inversion scenarios
  */
 void low_priority_task(void *arg) {
     int iterations = 0;
     
     while (1) {
         rtos_mutex_lock(console_mutex, UINT32_MAX);
         printf("[LOW PRIORITY] Working... iteration %d\n", iterations++);
         
         // Simulate long work while holding mutex
         for (volatile int i = 0; i < 500000; i++);
         
         rtos_mutex_unlock(console_mutex);
         
         rtos_task_delay(1000);
     }
 }
 
 /**
  * Message Queue Receiver Task
  * Demonstrates: Message queue usage
  */
 void queue_receiver_task(void *arg) {
     while (1) {
         void *msg;
         
         safe_printf("[QUEUE RECEIVER] Waiting for message...\n");
         
         if (rtos_queue_receive(msg_queue, &msg, UINT32_MAX) == RTOS_OK) {
             int *data = (int *)msg;
             safe_printf("[QUEUE RECEIVER] Received message: %d\n", *data);
             free(msg);
         }
         
         rtos_task_delay(200);
     }
 }
 
 /**
  * Periodic Task (started by timer)
  * Demonstrates: Timer-driven periodic execution
  */
 void timer_callback(void *arg) {
     static int timer_count = 0;
     printf("[TIMER] Periodic callback #%d at tick %lu\n", timer_count++, (unsigned long)arg);
 }
 
 /**
  * Interrupt Simulation Task
  * Demonstrates: Interrupt handling
  */
 void interrupt_handler(void) {
     static int irq_count = 0;
     printf("[IRQ] Interrupt #%d handled\n", irq_count++);
     
     // Signal a task from interrupt
     rtos_semaphore_signal(data_ready_sem);
 }
 
 void interrupt_generator_task(void *arg) {
     while (1) {
         rtos_task_delay(2000);
         
         // Simulate interrupt
         printf("[IRQ GEN] Triggering interrupt...\n");
         interrupt_handler();
     }
 }
 
 /* ==================== Priority Inversion Demo ==================== */
 
 static mutex_t *shared_mutex;
 
 void priority_inversion_low(void *arg) {
     while (1) {
         printf("[PI LOW] Acquiring mutex...\n");
         rtos_mutex_lock(shared_mutex, UINT32_MAX);
         
         printf("[PI LOW] Got mutex, working...\n");
         for (volatile int i = 0; i < 1000000; i++);
         
         printf("[PI LOW] Releasing mutex\n");
         rtos_mutex_unlock(shared_mutex);
         
         rtos_task_delay(3000);
     }
 }
 
 void priority_inversion_medium(void *arg) {
     while (1) {
         printf("[PI MEDIUM] Running (no mutex needed)\n");
         for (volatile int i = 0; i < 500000; i++);
         
         rtos_task_delay(1500);
     }
 }
 
 void priority_inversion_high(void *arg) {
     rtos_task_delay(1000); // Let low priority task acquire mutex first
     
     while (1) {
         printf("[PI HIGH] Trying to acquire mutex (priority inheritance demo)...\n");
         rtos_mutex_lock(shared_mutex, UINT32_MAX);
         
         printf("[PI HIGH] Got mutex!\n");
         rtos_mutex_unlock(shared_mutex);
         
         rtos_task_delay(3000);
     }
 }
 
 /* ==================== Main Application ==================== */
 
 int main(void) {
     printf("=== Minimal RTOS Demo ===\n");
     printf("Features: 512 Priority Levels, Preemptive Scheduling\n");
     printf("Synchronization: Semaphores, Mutexes, Message Queues\n\n");
     
     // Initialize RTOS
     rtos_init();
     
     // Create synchronization objects
     data_ready_sem = rtos_semaphore_create(0);
     console_mutex = rtos_mutex_create();
     msg_queue = rtos_queue_create(10);
     shared_mutex = rtos_mutex_create();
     
     if (!data_ready_sem || !console_mutex || !msg_queue || !shared_mutex) {
         printf("ERROR: Failed to create synchronization objects\n");
         return -1;
     }
     
     printf("Creating tasks...\n");
     
     // Create tasks with different priorities (0 = highest, 511 = lowest)
     
     // High priority tasks (0-100)
     task_tcb_t *high_prod = rtos_task_create(high_priority_producer, "HighProd", 10, NULL);
     task_tcb_t *queue_recv = rtos_task_create(queue_receiver_task, "QueueRecv", 20, NULL);
     
     // Medium priority tasks (100-300)
     task_tcb_t *med_cons = rtos_task_create(medium_priority_consumer, "MedCons", 150, NULL);
     task_tcb_t *irq_gen = rtos_task_create(interrupt_generator_task, "IrqGen", 200, NULL);
     
     // Low priority tasks (300-500)
     task_tcb_t *low_task = rtos_task_create(low_priority_task, "LowTask", 400, NULL);
     
     // Priority inversion demo tasks
     task_tcb_t *pi_low = rtos_task_create(priority_inversion_low, "PI_Low", 450, NULL);
     task_tcb_t *pi_med = rtos_task_create(priority_inversion_medium, "PI_Med", 250, NULL);
     task_tcb_t *pi_high = rtos_task_create(priority_inversion_high, "PI_High", 50, NULL);
     
     if (!high_prod || !med_cons || !low_task || !queue_recv || !irq_gen ||
         !pi_low || !pi_med || !pi_high) {
         printf("ERROR: Failed to create tasks\n");
         return -1;
     }
     
     // Create periodic timer (1000ms period)
     periodic_timer = rtos_timer_create(1000, timer_callback, (void *)0, true);
     if (periodic_timer) {
         rtos_timer_start(periodic_timer);
         printf("Periodic timer started\n");
     }
     
     // Register interrupt handlers
     rtos_interrupt_register(0, interrupt_handler, 10);
     rtos_interrupt_enable(0);
     
     printf("\n=== Starting RTOS Scheduler ===\n\n");
     
     // Start scheduler
     rtos_start();
     
     // Simulate system tick (in real system, this would be timer interrupt)
     for (int i = 0; i < 100; i++) {
         rtos_tick_handler();
         
         // Small delay to simulate time
         for (volatile int j = 0; j < 1000000; j++);
     }
     
     printf("\n=== Demo Complete ===\n");
     
     // Cleanup
     rtos_semaphore_delete(data_ready_sem);
     rtos_mutex_delete(console_mutex);
     rtos_mutex_delete(shared_mutex);
     rtos_queue_delete(msg_queue);
     rtos_timer_delete(periodic_timer);
     
     return 0;
 }