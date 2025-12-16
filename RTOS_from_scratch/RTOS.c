/**
 * @file rtos.c
 * @brief RTOS Core Implementation
 */

 #include "rtos.h"
 #include <string.h>
 #include <stdio.h>
 #include <stdlib.h>
 
 /* ==================== Global Variables ==================== */
 static task_tcb_t task_pool[RTOS_MAX_TASKS];
 static task_tcb_t *ready_lists[RTOS_MAX_PRIORITIES];
 static task_tcb_t *current_task = NULL;
 static task_tcb_t *idle_task = NULL;
 static uint32_t rtos_ticks = 0;
 static bool scheduler_running = false;
 static uint32_t critical_nesting = 0;
 static uint32_t next_task_id = 0;
 
 /* Synchronization objects */
 static semaphore_t semaphore_pool[RTOS_MAX_SEMAPHORES];
 static mutex_t mutex_pool[RTOS_MAX_MUTEXES];
 static message_queue_t queue_pool[RTOS_MAX_QUEUES];
 static timer_t timer_pool[RTOS_MAX_TIMERS];
 
 /* Interrupt table */
 static interrupt_descriptor_t interrupt_table[RTOS_MAX_PRIORITIES];
 
 /* Priority bitmap for fast scheduling */
 static uint64_t priority_bitmap[8]; // 512 bits / 64 = 8
 
 /* ==================== Helper Functions ==================== */
 
 static void set_priority_bit(uint16_t priority) {
     priority_bitmap[priority / 64] |= (1ULL << (priority % 64));
 }
 
 static void clear_priority_bit(uint16_t priority) {
     priority_bitmap[priority / 64] &= ~(1ULL << (priority % 64));
 }
 
 static uint16_t get_highest_priority(void) {
     for (int i = 0; i < 8; i++) {
         if (priority_bitmap[i] != 0) {
             return (i * 64) + __builtin_ctzll(priority_bitmap[i]);
         }
     }
     return RTOS_MAX_PRIORITIES - 1; // Idle task priority
 }
 
 static void add_to_ready_list(task_tcb_t *task) {
     task->next = ready_lists[task->priority];
     ready_lists[task->priority] = task;
     set_priority_bit(task->priority);
     task->state = TASK_STATE_READY;
 }
 
 static void remove_from_ready_list(task_tcb_t *task) {
     task_tcb_t **current = &ready_lists[task->priority];
     
     while (*current) {
         if (*current == task) {
             *current = task->next;
             task->next = NULL;
             
             // If list is empty, clear priority bit
             if (ready_lists[task->priority] == NULL) {
                 clear_priority_bit(task->priority);
             }
             return;
         }
         current = &((*current)->next);
     }
 }
 
 static void add_to_wait_list(task_tcb_t **list, task_tcb_t *task) {
     task->next = *list;
     *list = task;
     task->state = TASK_STATE_BLOCKED;
 }
 
 static task_tcb_t* remove_from_wait_list(task_tcb_t **list) {
     if (*list == NULL) return NULL;
     
     // Get highest priority task from wait list
     task_tcb_t *highest = *list;
     task_tcb_t **highest_prev = list;
     task_tcb_t **current = &((*list)->next);
     task_tcb_t **prev = list;
     
     while (*current) {
         if ((*current)->priority < highest->priority) {
             highest = *current;
             highest_prev = prev;
         }
         prev = current;
         current = &((*current)->next);
     }
     
     *highest_prev = highest->next;
     highest->next = NULL;
     return highest;
 }
 
 /* ==================== Context Switching ==================== */
 
 static void context_switch(void) {
     if (!scheduler_running) return;
     
     uint16_t highest_priority = get_highest_priority();
     task_tcb_t *next_task = ready_lists[highest_priority];
     
     if (next_task != current_task && next_task != NULL) {
         task_tcb_t *prev_task = current_task;
         current_task = next_task;
         
         if (prev_task) {
             prev_task->state = TASK_STATE_READY;
         }
         current_task->state = TASK_STATE_RUNNING;
         
         // In real implementation, save/restore CPU registers here
         // This is a simplified version for demonstration
     }
 }
 
 /* ==================== Idle Task ==================== */
 
 static void idle_task_func(void *arg) {
     (void)arg;
     while (1) {
         // Could put CPU to sleep here
         __asm__ volatile("nop");
     }
 }
 
 /* ==================== Core API Implementation ==================== */
 
 void rtos_init(void) {
     memset(task_pool, 0, sizeof(task_pool));
     memset(ready_lists, 0, sizeof(ready_lists));
     memset(semaphore_pool, 0, sizeof(semaphore_pool));
     memset(mutex_pool, 0, sizeof(mutex_pool));
     memset(queue_pool, 0, sizeof(queue_pool));
     memset(timer_pool, 0, sizeof(timer_pool));
     memset(interrupt_table, 0, sizeof(interrupt_table));
     memset(priority_bitmap, 0, sizeof(priority_bitmap));
     
     current_task = NULL;
     rtos_ticks = 0;
     scheduler_running = false;
     critical_nesting = 0;
     next_task_id = 1;
     
     // Create idle task
     idle_task = rtos_task_create(idle_task_func, "IDLE", RTOS_MAX_PRIORITIES - 1, NULL);
 }
 
 void rtos_start(void) {
     scheduler_running = true;
     context_switch();
     
     // In real implementation, enable interrupts and start first task
     printf("RTOS Started\n");
 }
 
 task_tcb_t* rtos_task_create(void (*task_func)(void *), const char *name, 
                               uint16_t priority, void *arg) {
     if (priority >= RTOS_MAX_PRIORITIES) {
         return NULL;
     }
     
     // Find free TCB
     task_tcb_t *task = NULL;
     for (int i = 0; i < RTOS_MAX_TASKS; i++) {
         if (task_pool[i].state == TASK_STATE_DELETED || task_pool[i].task_id == 0) {
             task = &task_pool[i];
             break;
         }
     }
     
     if (!task) return NULL;
     
     // Initialize task
     memset(task, 0, sizeof(task_tcb_t));
     task->priority = priority;
     task->state = TASK_STATE_READY;
     task->task_id = next_task_id++;
     strncpy(task->name, name, sizeof(task->name) - 1);
     
     // Initialize stack (simplified - in real RTOS would setup full context)
     task->stack_ptr = &task->stack[RTOS_STACK_SIZE - 1];
     
     // Add to ready list
     add_to_ready_list(task);
     
     printf("Task '%s' created with priority %d\n", name, priority);
     
     return task;
 }
 
 rtos_status_t rtos_task_delete(task_tcb_t *task) {
     if (!task) task = current_task;
     if (!task) return RTOS_ERROR;
     
     rtos_critical_enter();
     
     if (task->state == TASK_STATE_READY) {
         remove_from_ready_list(task);
     }
     
     task->state = TASK_STATE_DELETED;
     
     if (task == current_task) {
         current_task = NULL;
         rtos_critical_exit();
         context_switch();
     } else {
         rtos_critical_exit();
     }
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_task_suspend(task_tcb_t *task) {
     if (!task) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     if (task->state == TASK_STATE_READY) {
         remove_from_ready_list(task);
     }
     
     task->state = TASK_STATE_SUSPENDED;
     
     if (task == current_task) {
         current_task = NULL;
         rtos_critical_exit();
         context_switch();
     } else {
         rtos_critical_exit();
     }
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_task_resume(task_tcb_t *task) {
     if (!task || task->state != TASK_STATE_SUSPENDED) {
         return RTOS_INVALID_PARAM;
     }
     
     rtos_critical_enter();
     add_to_ready_list(task);
     rtos_critical_exit();
     
     context_switch();
     return RTOS_OK;
 }
 
 void rtos_task_delay(uint32_t ticks) {
     if (ticks == 0) return;
     
     rtos_critical_enter();
     
     current_task->delay_ticks = ticks;
     remove_from_ready_list(current_task);
     current_task->state = TASK_STATE_BLOCKED;
     
     rtos_critical_exit();
     context_switch();
 }
 
 void rtos_task_yield(void) {
     context_switch();
 }
 
 task_tcb_t* rtos_task_get_current(void) {
     return current_task;
 }
 
 /* ==================== Semaphore Implementation ==================== */
 
 semaphore_t* rtos_semaphore_create(int32_t initial_count) {
     semaphore_t *sem = NULL;
     
     for (int i = 0; i < RTOS_MAX_SEMAPHORES; i++) {
         if (!semaphore_pool[i].valid) {
             sem = &semaphore_pool[i];
             break;
         }
     }
     
     if (!sem) return NULL;
     
     sem->count = initial_count;
     sem->wait_list = NULL;
     sem->valid = true;
     
     return sem;
 }
 
 rtos_status_t rtos_semaphore_wait(semaphore_t *sem, uint32_t timeout_ticks) {
     if (!sem || !sem->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     if (sem->count > 0) {
         sem->count--;
         rtos_critical_exit();
         return RTOS_OK;
     }
     
     if (timeout_ticks == 0) {
         rtos_critical_exit();
         return RTOS_WOULD_BLOCK;
     }
     
     // Block current task
     remove_from_ready_list(current_task);
     add_to_wait_list(&sem->wait_list, current_task);
     current_task->wait_object = sem;
     current_task->delay_ticks = timeout_ticks;
     
     rtos_critical_exit();
     context_switch();
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_semaphore_signal(semaphore_t *sem) {
     if (!sem || !sem->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     if (sem->wait_list) {
         // Wake up highest priority waiting task
         task_tcb_t *task = remove_from_wait_list(&sem->wait_list);
         if (task) {
             task->wait_object = NULL;
             add_to_ready_list(task);
         }
     } else {
         sem->count++;
     }
     
     rtos_critical_exit();
     context_switch();
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_semaphore_delete(semaphore_t *sem) {
     if (!sem || !sem->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     // Wake up all waiting tasks
     while (sem->wait_list) {
         task_tcb_t *task = remove_from_wait_list(&sem->wait_list);
         if (task) {
             task->wait_object = NULL;
             add_to_ready_list(task);
         }
     }
     
     sem->valid = false;
     rtos_critical_exit();
     
     return RTOS_OK;
 }
 
 /* ==================== Mutex Implementation ==================== */
 
 mutex_t* rtos_mutex_create(void) {
     mutex_t *mutex = NULL;
     
     for (int i = 0; i < RTOS_MAX_MUTEXES; i++) {
         if (!mutex_pool[i].valid) {
             mutex = &mutex_pool[i];
             break;
         }
     }
     
     if (!mutex) return NULL;
     
     mutex->owner = NULL;
     mutex->original_priority = 0;
     mutex->lock_count = 0;
     mutex->wait_list = NULL;
     mutex->valid = true;
     
     return mutex;
 }
 
 rtos_status_t rtos_mutex_lock(mutex_t *mutex, uint32_t timeout_ticks) {
     if (!mutex || !mutex->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     // Check if current task already owns mutex (recursive)
     if (mutex->owner == current_task) {
         mutex->lock_count++;
         rtos_critical_exit();
         return RTOS_OK;
     }
     
     // Check if mutex is available
     if (mutex->owner == NULL) {
         mutex->owner = current_task;
         mutex->original_priority = current_task->priority;
         mutex->lock_count = 1;
         rtos_critical_exit();
         return RTOS_OK;
     }
     
     // Priority inheritance: boost owner's priority if needed
     if (current_task->priority < mutex->owner->priority) {
         remove_from_ready_list(mutex->owner);
         mutex->owner->priority = current_task->priority;
         add_to_ready_list(mutex->owner);
     }
     
     if (timeout_ticks == 0) {
         rtos_critical_exit();
         return RTOS_WOULD_BLOCK;
     }
     
     // Block current task
     remove_from_ready_list(current_task);
     add_to_wait_list(&mutex->wait_list, current_task);
     current_task->wait_object = mutex;
     current_task->delay_ticks = timeout_ticks;
     
     rtos_critical_exit();
     context_switch();
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_mutex_unlock(mutex_t *mutex) {
     if (!mutex || !mutex->valid) return RTOS_INVALID_PARAM;
     if (mutex->owner != current_task) return RTOS_ERROR;
     
     rtos_critical_enter();
     
     mutex->lock_count--;
     
     if (mutex->lock_count == 0) {
         // Restore original priority
         if (current_task->priority != mutex->original_priority) {
             remove_from_ready_list(current_task);
             current_task->priority = mutex->original_priority;
             add_to_ready_list(current_task);
         }
         
         mutex->owner = NULL;
         
         // Wake up highest priority waiting task
         if (mutex->wait_list) {
             task_tcb_t *task = remove_from_wait_list(&mutex->wait_list);
             if (task) {
                 task->wait_object = NULL;
                 mutex->owner = task;
                 mutex->original_priority = task->priority;
                 mutex->lock_count = 1;
                 add_to_ready_list(task);
             }
         }
     }
     
     rtos_critical_exit();
     context_switch();
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_mutex_delete(mutex_t *mutex) {
     if (!mutex || !mutex->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     // Restore owner's priority if needed
     if (mutex->owner && mutex->owner->priority != mutex->original_priority) {
         remove_from_ready_list(mutex->owner);
         mutex->owner->priority = mutex->original_priority;
         add_to_ready_list(mutex->owner);
     }
     
     // Wake up all waiting tasks
     while (mutex->wait_list) {
         task_tcb_t *task = remove_from_wait_list(&mutex->wait_list);
         if (task) {
             task->wait_object = NULL;
             add_to_ready_list(task);
         }
     }
     
     mutex->valid = false;
     rtos_critical_exit();
     
     return RTOS_OK;
 }
 
 /* ==================== Message Queue Implementation ==================== */
 
 message_queue_t* rtos_queue_create(uint32_t size) {
     message_queue_t *queue = NULL;
     
     for (int i = 0; i < RTOS_MAX_QUEUES; i++) {
         if (!queue_pool[i].valid) {
             queue = &queue_pool[i];
             break;
         }
     }
     
     if (!queue) return NULL;
     
     queue->buffer = (void **)malloc(size * sizeof(void *));
     if (!queue->buffer) return NULL;
     
     queue->size = size;
     queue->head = 0;
     queue->tail = 0;
     queue->count = 0;
     queue->wait_list = NULL;
     queue->valid = true;
     
     return queue;
 }
 
 rtos_status_t rtos_queue_send(message_queue_t *queue, void *message, uint32_t timeout_ticks) {
     if (!queue || !queue->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     if (queue->count < queue->size) {
         queue->buffer[queue->tail] = message;
         queue->tail = (queue->tail + 1) % queue->size;
         queue->count++;
         
         // Wake up waiting receiver
         if (queue->wait_list) {
             task_tcb_t *task = remove_from_wait_list(&queue->wait_list);
             if (task) {
                 task->wait_object = NULL;
                 add_to_ready_list(task);
             }
         }
         
         rtos_critical_exit();
         context_switch();
         return RTOS_OK;
     }
     
     rtos_critical_exit();
     return RTOS_ERROR;
 }
 
 rtos_status_t rtos_queue_receive(message_queue_t *queue, void **message, uint32_t timeout_ticks) {
     if (!queue || !queue->valid || !message) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     if (queue->count > 0) {
         *message = queue->buffer[queue->head];
         queue->head = (queue->head + 1) % queue->size;
         queue->count--;
         rtos_critical_exit();
         return RTOS_OK;
     }
     
     if (timeout_ticks == 0) {
         rtos_critical_exit();
         return RTOS_WOULD_BLOCK;
     }
     
     // Block current task
     remove_from_ready_list(current_task);
     add_to_wait_list(&queue->wait_list, current_task);
     current_task->wait_object = queue;
     current_task->delay_ticks = timeout_ticks;
     
     rtos_critical_exit();
     context_switch();
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_queue_delete(message_queue_t *queue) {
     if (!queue || !queue->valid) return RTOS_INVALID_PARAM;
     
     rtos_critical_enter();
     
     free(queue->buffer);
     
     while (queue->wait_list) {
         task_tcb_t *task = remove_from_wait_list(&queue->wait_list);
         if (task) {
             task->wait_object = NULL;
             add_to_ready_list(task);
         }
     }
     
     queue->valid = false;
     rtos_critical_exit();
     
     return RTOS_OK;
 }
 
 /* ==================== Timer Implementation ==================== */
 
 timer_t* rtos_timer_create(uint32_t period_ms, timer_callback_t callback, void *arg, bool periodic) {
     timer_t *timer = NULL;
     
     for (int i = 0; i < RTOS_MAX_TIMERS; i++) {
         if (!timer_pool[i].valid) {
             timer = &timer_pool[i];
             break;
         }
     }
     
     if (!timer) return NULL;
     
     timer->period_ticks = (period_ms * RTOS_TICK_RATE_HZ) / 1000;
     timer->remaining_ticks = timer->period_ticks;
     timer->callback = callback;
     timer->arg = arg;
     timer->periodic = periodic;
     timer->active = false;
     timer->valid = true;
     
     return timer;
 }
 
 rtos_status_t rtos_timer_start(timer_t *timer) {
     if (!timer || !timer->valid) return RTOS_INVALID_PARAM;
     
     timer->remaining_ticks = timer->period_ticks;
     timer->active = true;
     return RTOS_OK;
 }
 
 rtos_status_t rtos_timer_stop(timer_t *timer) {
     if (!timer || !timer->valid) return RTOS_INVALID_PARAM;
     
     timer->active = false;
     return RTOS_OK;
 }
 
 rtos_status_t rtos_timer_delete(timer_t *timer) {
     if (!timer || !timer->valid) return RTOS_INVALID_PARAM;
     
     timer->valid = false;
     timer->active = false;
     return RTOS_OK;
 }
 
 /* ==================== Interrupt Management ==================== */
 
 rtos_status_t rtos_interrupt_register(uint16_t irq_num, void (*handler)(void), uint16_t priority) {
     if (irq_num >= RTOS_MAX_PRIORITIES) return RTOS_INVALID_PARAM;
     
     interrupt_table[irq_num].handler = handler;
     interrupt_table[irq_num].priority = priority;
     interrupt_table[irq_num].enabled = false;
     
     return RTOS_OK;
 }
 
 rtos_status_t rtos_interrupt_enable(uint16_t irq_num) {
     if (irq_num >= RTOS_MAX_PRIORITIES) return RTOS_INVALID_PARAM;
     
     interrupt_table[irq_num].enabled = true;
     return RTOS_OK;
 }
 
 rtos_status_t rtos_interrupt_disable(uint16_t irq_num) {
     if (irq_num >= RTOS_MAX_PRIORITIES) return RTOS_INVALID_PARAM;
     
     interrupt_table[irq_num].enabled = false;
     return RTOS_OK;
 }
 
 void rtos_critical_enter(void) {
     // Disable interrupts
     critical_nesting++;
 }
 
 void rtos_critical_exit(void) {
     if (critical_nesting > 0) {
         critical_nesting--;
     }
     // Enable interrupts when nesting reaches 0
 }
 
 /* ==================== System Tick Handler ==================== */
 
 void rtos_tick_handler(void) {
     rtos_ticks++;
     
     // Process delayed tasks
     for (int i = 0; i < RTOS_MAX_TASKS; i++) {
         task_tcb_t *task = &task_pool[i];
         
         if (task->state == TASK_STATE_BLOCKED && task->delay_ticks > 0) {
             task->delay_ticks--;
             
             if (task->delay_ticks == 0 && task->wait_object == NULL) {
                 add_to_ready_list(task);
             }
         }
     }
     
     // Process timers
     for (int i = 0; i < RTOS_MAX_TIMERS; i++) {
         timer_t *timer = &timer_pool[i];
         
         if (timer->valid && timer->active) {
             timer->remaining_ticks--;
             
             if (timer->remaining_ticks == 0) {
                 if (timer->callback) {
                     timer->callback(timer->arg);
                 }
                 
                 if (timer->periodic) {
                     timer->remaining_ticks = timer->period_ticks;
                 } else {
                     timer->active = false;
                 }
             }
         }
     }
     
     // Trigger context switch
     context_switch();
 }