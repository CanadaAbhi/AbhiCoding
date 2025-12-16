/*
 * RTOS Project 1: LED Blinking with Multiple Tasks
 * 
 * Demonstrates:
 * - Creating multiple tasks with different priorities
 * - Preemptive scheduling
 * - Task switching and timing
 * - Simulates two LEDs blinking at different frequencies
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <pthread.h>
 #include <unistd.h>
 #include <time.h>
 #include <stdbool.h>
 #include <signal.h>
 
 // Simulated LED states
 volatile bool led1_state = false;
 volatile bool led2_state = false;
 volatile bool running = true;
 
 // Get current time in milliseconds
 long long get_time_ms() {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
 }
 
 // Task delay function (simulates vTaskDelay)
 void task_delay(int ms) {
     usleep(ms * 1000);
 }
 
 // Task 1: Blink LED at 1 Hz (500ms on, 500ms off)
 void* task_led1(void* param) {
     printf("[Task LED1] Started - Priority: HIGH (1 Hz)\n");
     
     while (running) {
         led1_state = !led1_state;
         printf("[%lld ms] LED1: %s\n", get_time_ms(), 
                led1_state ? "█ ON " : "○ OFF");
         task_delay(500);  // 500ms delay = 1Hz
     }
     
     return NULL;
 }
 
 // Task 2: Blink LED at 2 Hz (250ms on, 250ms off)
 void* task_led2(void* param) {
     printf("[Task LED2] Started - Priority: MEDIUM (2 Hz)\n");
     
     while (running) {
         led2_state = !led2_state;
         printf("[%lld ms] LED2: %s\n", get_time_ms(), 
                led2_state ? "█ ON " : "○ OFF");
         task_delay(250);  // 250ms delay = 2Hz
     }
     
     return NULL;
 }
 
 // Task 3: System Monitor (low priority)
 void* task_monitor(void* param) {
     printf("[Task Monitor] Started - Priority: LOW\n");
     int count = 0;
     
     while (running) {
         count++;
         printf("[%lld ms] [MONITOR] System uptime: %d seconds | LED1:%s LED2:%s\n", 
                get_time_ms(), count,
                led1_state ? "ON " : "OFF",
                led2_state ? "ON " : "OFF");
         task_delay(2000);  // Update every 2 seconds
     }
     
     return NULL;
 }
 
 // Signal handler for graceful shutdown
 void signal_handler(int sig) {
     printf("\n\n[SHUTDOWN] Stopping all tasks...\n");
     running = false;
 }
 
 int main() {
     pthread_t thread_led1, thread_led2, thread_monitor;
     struct sched_param param1, param2, param3;
     
     printf("╔════════════════════════════════════════════════════════╗\n");
     printf("║   RTOS Project 1: LED Blinking with Multiple Tasks    ║\n");
     printf("╚════════════════════════════════════════════════════════╝\n\n");
     
     printf("Concepts demonstrated:\n");
     printf("  - Task creation and scheduling\n");
     printf("  - Different task priorities\n");
     printf("  - Preemptive multitasking\n");
     printf("  - Periodic task execution\n\n");
     
     printf("Tasks:\n");
     printf("  LED1: 1 Hz (500ms period) - HIGH priority\n");
     printf("  LED2: 2 Hz (250ms period) - MEDIUM priority\n");
     printf("  Monitor: Every 2s - LOW priority\n\n");
     
     printf("Press Ctrl+C to stop\n");
     printf("════════════════════════════════════════════════════════\n\n");
     
     // Setup signal handler
     signal(SIGINT, signal_handler);
     
     // Create threads (tasks)
     pthread_create(&thread_led1, NULL, task_led1, NULL);
     pthread_create(&thread_led2, NULL, task_led2, NULL);
     pthread_create(&thread_monitor, NULL, task_monitor, NULL);
     
     // Set thread priorities (if running as root)
     param1.sched_priority = 30;  // High priority
     param2.sched_priority = 20;  // Medium priority
     param3.sched_priority = 10;  // Low priority
     
     pthread_setschedparam(thread_led1, SCHED_FIFO, &param1);
     pthread_setschedparam(thread_led2, SCHED_FIFO, &param2);
     pthread_setschedparam(thread_monitor, SCHED_FIFO, &param3);
     
     // Wait for all threads to complete
     pthread_join(thread_led1, NULL);
     pthread_join(thread_led2, NULL);
     pthread_join(thread_monitor, NULL);
     
     printf("\n[SHUTDOWN] All tasks stopped\n");
     printf("Program terminated successfully\n");
     
     return 0;
 }