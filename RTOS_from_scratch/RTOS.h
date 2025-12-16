/**
 * @file rtos.h
 * @brief Minimal RTOS with Preemptive Priority Scheduling
 * 
 * Features:
 * - 512 priority levels (0 = highest, 511 = lowest)
 * - Preemptive priority-based scheduling
 * - Semaphores, Mutexes, Event Flags
 * - Message Queues
 * - Timer support
 * - Minimal memory footprint
 */

 #ifndef RTOS_H
 #define RTOS_H
 
 #include <stdint.h>
 #include <stdbool.h>
 #include <stddef.h>
 
 /* ==================== Configuration ==================== */
 #define RTOS_MAX_TASKS          32
 #define RTOS_MAX_PRIORITIES     512
 #define RTOS_STACK_SIZE         1024
 #define RTOS_MAX_SEMAPHORES     16
 #define RTOS_MAX_MUTEXES        16
 #define RTOS_MAX_QUEUES         8
 #define RTOS_MAX_TIMERS         16
 #define RTOS_TICK_RATE_HZ       1000
 
 /* ==================== Task States ==================== */
 typedef enum {
     TASK_STATE_READY = 0,
     TASK_STATE_RUNNING,
     TASK_STATE_BLOCKED,
     TASK_STATE_SUSPENDED,
     TASK_STATE_DELETED
 } task_state_t;
 
 /* ==================== Task Control Block ==================== */
 typedef struct task_tcb {
     uint32_t *stack_ptr;              /* Current stack pointer */
     uint32_t stack[RTOS_STACK_SIZE];  /* Task stack */
     uint16_t priority;                /* Task priority (0-511) */
     task_state_t state;               /* Current task state */
     uint32_t delay_ticks;             /* Delay counter */
     char name[16];                    /* Task name */
     struct task_tcb *next;            /* Next task in list */
     void *wait_object;                /* Object task is waiting on */
     uint32_t task_id;                 /* Unique task ID */
 } task_tcb_t;
 
 /* ==================== Semaphore ==================== */
 typedef struct {
     int32_t count;                    /* Semaphore count */
     task_tcb_t *wait_list;            /* Tasks waiting on semaphore */
     bool valid;                       /* Is this semaphore valid */
 } semaphore_t;
 
 /* ==================== Mutex ==================== */
 typedef struct {
     task_tcb_t *owner;                /* Current mutex owner */
     uint16_t original_priority;       /* Owner's original priority */
     uint32_t lock_count;              /* Recursive lock count */
     task_tcb_t *wait_list;            /* Tasks waiting on mutex */
     bool valid;                       /* Is this mutex valid */
 } mutex_t;
 
 /* ==================== Message Queue ==================== */
 typedef struct {
     void **buffer;                    /* Message buffer */
     uint32_t size;                    /* Queue size */
     uint32_t head;                    /* Queue head */
     uint32_t tail;                    /* Queue tail */
     uint32_t count;                   /* Number of messages */
     task_tcb_t *wait_list;            /* Tasks waiting to receive */
     bool valid;                       /* Is this queue valid */
 } message_queue_t;
 
 /* ==================== Timer ==================== */
 typedef void (*timer_callback_t)(void *arg);
 
 typedef struct {
     uint32_t period_ticks;            /* Timer period */
     uint32_t remaining_ticks;         /* Remaining ticks */
     timer_callback_t callback;        /* Timer callback */
     void *arg;                        /* Callback argument */
     bool periodic;                    /* Is timer periodic */
     bool active;                      /* Is timer active */
     bool valid;                       /* Is this timer valid */
 } timer_t;
 
 /* ==================== Interrupt Priority Levels ==================== */
 typedef struct {
     void (*handler)(void);            /* Interrupt handler */
     uint16_t priority;                /* Interrupt priority (0-511) */
     bool enabled;                     /* Is interrupt enabled */
 } interrupt_descriptor_t;
 
 /* ==================== Return Codes ==================== */
 typedef enum {
     RTOS_OK = 0,
     RTOS_ERROR,
     RTOS_TIMEOUT,
     RTOS_INVALID_PARAM,
     RTOS_NO_MEMORY,
     RTOS_WOULD_BLOCK
 } rtos_status_t;
 
 /* ==================== Core API ==================== */
 
 /**
  * @brief Initialize the RTOS
  */
 void rtos_init(void);
 
 /**
  * @brief Start the RTOS scheduler
  */
 void rtos_start(void);
 
 /**
  * @brief Create a new task
  * @param task_func Task function
  * @param name Task name
  * @param priority Task priority (0-511, 0 = highest)
  * @param arg Task argument
  * @return Task handle or NULL on error
  */
 task_tcb_t* rtos_task_create(void (*task_func)(void *), 
                               const char *name,
                               uint16_t priority, 
                               void *arg);
 
 /**
  * @brief Delete a task
  * @param task Task to delete (NULL = current task)
  */
 rtos_status_t rtos_task_delete(task_tcb_t *task);
 
 /**
  * @brief Suspend a task
  */
 rtos_status_t rtos_task_suspend(task_tcb_t *task);
 
 /**
  * @brief Resume a task
  */
 rtos_status_t rtos_task_resume(task_tcb_t *task);
 
 /**
  * @brief Delay current task
  * @param ticks Number of ticks to delay
  */
 void rtos_task_delay(uint32_t ticks);
 
 /**
  * @brief Yield CPU to next ready task
  */
 void rtos_task_yield(void);
 
 /**
  * @brief Get current task
  */
 task_tcb_t* rtos_task_get_current(void);
 
 /* ==================== Semaphore API ==================== */
 
 /**
  * @brief Create a semaphore
  * @param initial_count Initial count
  */
 semaphore_t* rtos_semaphore_create(int32_t initial_count);
 
 /**
  * @brief Wait on semaphore
  * @param sem Semaphore
  * @param timeout_ticks Timeout in ticks (0 = no wait, UINT32_MAX = forever)
  */
 rtos_status_t rtos_semaphore_wait(semaphore_t *sem, uint32_t timeout_ticks);
 
 /**
  * @brief Signal semaphore
  */
 rtos_status_t rtos_semaphore_signal(semaphore_t *sem);
 
 /**
  * @brief Delete semaphore
  */
 rtos_status_t rtos_semaphore_delete(semaphore_t *sem);
 
 /* ==================== Mutex API ==================== */
 
 /**
  * @brief Create a mutex
  */
 mutex_t* rtos_mutex_create(void);
 
 /**
  * @brief Lock mutex
  * @param mutex Mutex
  * @param timeout_ticks Timeout in ticks
  */
 rtos_status_t rtos_mutex_lock(mutex_t *mutex, uint32_t timeout_ticks);
 
 /**
  * @brief Unlock mutex
  */
 rtos_status_t rtos_mutex_unlock(mutex_t *mutex);
 
 /**
  * @brief Delete mutex
  */
 rtos_status_t rtos_mutex_delete(mutex_t *mutex);
 
 /* ==================== Message Queue API ==================== */
 
 /**
  * @brief Create message queue
  * @param size Queue size
  */
 message_queue_t* rtos_queue_create(uint32_t size);
 
 /**
  * @brief Send message to queue
  * @param queue Queue
  * @param message Message pointer
  * @param timeout_ticks Timeout
  */
 rtos_status_t rtos_queue_send(message_queue_t *queue, void *message, uint32_t timeout_ticks);
 
 /**
  * @brief Receive message from queue
  * @param queue Queue
  * @param message Pointer to receive message
  * @param timeout_ticks Timeout
  */
 rtos_status_t rtos_queue_receive(message_queue_t *queue, void **message, uint32_t timeout_ticks);
 
 /**
  * @brief Delete queue
  */
 rtos_status_t rtos_queue_delete(message_queue_t *queue);
 
 /* ==================== Timer API ==================== */
 
 /**
  * @brief Create a timer
  * @param period_ms Period in milliseconds
  * @param callback Callback function
  * @param arg Callback argument
  * @param periodic Is timer periodic
  */
 timer_t* rtos_timer_create(uint32_t period_ms, timer_callback_t callback, void *arg, bool periodic);
 
 /**
  * @brief Start a timer
  */
 rtos_status_t rtos_timer_start(timer_t *timer);
 
 /**
  * @brief Stop a timer
  */
 rtos_status_t rtos_timer_stop(timer_t *timer);
 
 /**
  * @brief Delete a timer
  */
 rtos_status_t rtos_timer_delete(timer_t *timer);
 
 /* ==================== Interrupt API ==================== */
 
 /**
  * @brief Register interrupt handler
  * @param irq_num Interrupt number (0-511)
  * @param handler Handler function
  * @param priority Interrupt priority
  */
 rtos_status_t rtos_interrupt_register(uint16_t irq_num, void (*handler)(void), uint16_t priority);
 
 /**
  * @brief Enable interrupt
  */
 rtos_status_t rtos_interrupt_enable(uint16_t irq_num);
 
 /**
  * @brief Disable interrupt
  */
 rtos_status_t rtos_interrupt_disable(uint16_t irq_num);
 
 /**
  * @brief Enter critical section
  */
 void rtos_critical_enter(void);
 
 /**
  * @brief Exit critical section
  */
 void rtos_critical_exit(void);
 
 /* ==================== System Tick ==================== */
 
 /**
  * @brief System tick handler (called from timer interrupt)
  */
 void rtos_tick_handler(void);
 
 #endif /* RTOS_H */