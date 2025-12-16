/*
 * RTOS Project 2: Producer-Consumer using Queue
 * 
 * Demonstrates:
 * - Inter-task communication using queues
 * - Blocking and non-blocking queue operations
 * - Producer-consumer pattern
 * - Thread synchronization
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <pthread.h>
 #include <unistd.h>
 #include <time.h>
 #include <stdbool.h>
 #include <signal.h>
 
 #define QUEUE_SIZE 10
 
 typedef struct {
     int data[QUEUE_SIZE];
     int front;
     int rear;
     int count;
     pthread_mutex_t mutex;
     pthread_cond_t not_empty;
     pthread_cond_t not_full;
 } Queue;
 
 Queue message_queue;
 volatile bool running = true;
 
 // Get current time in milliseconds
 long long get_time_ms() {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
 }
 
 // Initialize queue
 void queue_init(Queue* q) {
     q->front = 0;
     q->rear = 0;
     q->count = 0;
     pthread_mutex_init(&q->mutex, NULL);
     pthread_cond_init(&q->not_empty, NULL);
     pthread_cond_init(&q->not_full, NULL);
 }
 
 // Send to queue (blocking)
 bool queue_send(Queue* q, int data, int timeout_ms) {
     pthread_mutex_lock(&q->mutex);
     
     // Wait if queue is full
     while (q->count >= QUEUE_SIZE && running) {
         printf("[%lld ms] [QUEUE] Full, producer waiting...\n", get_time_ms());
         
         if (timeout_ms > 0) {
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             ts.tv_sec += timeout_ms / 1000;
             ts.tv_nsec += (timeout_ms % 1000) * 1000000;
             
             if (pthread_cond_timedwait(&q->not_full, &q->mutex, &ts) != 0) {
                 pthread_mutex_unlock(&q->mutex);
                 return false;  // Timeout
             }
         } else {
             pthread_cond_wait(&q->not_full, &q->mutex);
         }
     }
     
     if (!running) {
         pthread_mutex_unlock(&q->mutex);
         return false;
     }
     
     // Add data to queue
     q->data[q->rear] = data;
     q->rear = (q->rear + 1) % QUEUE_SIZE;
     q->count++;
     
     printf("[%lld ms] [QUEUE] Sent: %d (count: %d/%d)\n", 
            get_time_ms(), data, q->count, QUEUE_SIZE);
     
     // Signal consumer
     pthread_cond_signal(&q->not_empty);
     pthread_mutex_unlock(&q->mutex);
     
     return true;
 }
 
 // Receive from queue (blocking)
 bool queue_receive(Queue* q, int* data, int timeout_ms) {
     pthread_mutex_lock(&q->mutex);
     
     // Wait if queue is empty
     while (q->count <= 0 && running) {
         if (timeout_ms > 0) {
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             ts.tv_sec += timeout_ms / 1000;
             ts.tv_nsec += (timeout_ms % 1000) * 1000000;
             
             if (pthread_cond_timedwait(&q->not_empty, &q->mutex, &ts) != 0) {
                 pthread_mutex_unlock(&q->mutex);
                 return false;  // Timeout
             }
         } else {
             pthread_cond_wait(&q->not_empty, &q->mutex);
         }
     }
     
     if (!running && q->count == 0) {
         pthread_mutex_unlock(&q->mutex);
         return false;
     }
     
     // Get data from queue
     *data = q->data[q->front];
     q->front = (q->front + 1) % QUEUE_SIZE;
     q->count--;
     
     printf("[%lld ms] [QUEUE] Received: %d (count: %d/%d)\n", 
            get_time_ms(), *data, q->count, QUEUE_SIZE);
     
     // Signal producer
     pthread_cond_signal(&q->not_full);
     pthread_mutex_unlock(&q->mutex);
     
     return true;
 }
 
 // Producer task - generates sensor data
 void* task_producer(void* param) {
     int sensor_id = *(int*)param;
     int sequence = 0;
     
     printf("[%lld ms] [PRODUCER %d] Started\n", get_time_ms(), sensor_id);
     
     while (running) {
         // Simulate sensor reading
         int sensor_data = (sensor_id * 1000) + sequence;
         
         printf("[%lld ms] [PRODUCER %d] Generated data: %d\n", 
                get_time_ms(), sensor_id, sensor_data);
         
         // Send to queue
         if (queue_send(&message_queue, sensor_data, 2000)) {
             sequence++;
         }
         
         // Simulate sensor sampling rate
         usleep((rand() % 500 + 300) * 1000);  // 300-800ms
     }
     
     printf("[%lld ms] [PRODUCER %d] Stopped\n", get_time_ms(), sensor_id);
     return NULL;
 }
 
 // Consumer task - processes data
 void* task_consumer(void* param) {
     int consumer_id = *(int*)param;
     int data;
     int processed_count = 0;
     
     printf("[%lld ms] [CONSUMER %d] Started\n", get_time_ms(), consumer_id);
     
     while (running) {
         // Receive from queue
         if (queue_receive(&message_queue, &data, 1000)) {
             printf("[%lld ms] [CONSUMER %d] Processing data: %d\n", 
                    get_time_ms(), consumer_id, data);
             
             // Simulate processing time
             usleep((rand() % 200 + 100) * 1000);  // 100-300ms
             
             processed_count++;
             printf("[%lld ms] [CONSUMER %d] Completed: %d (total: %d)\n", 
                    get_time_ms(), consumer_id, data, processed_count);
         }
     }
     
     printf("[%lld ms] [CONSUMER %d] Stopped (processed %d items)\n", 
            get_time_ms(), consumer_id, processed_count);
     return NULL;
 }
 
 // Monitor task
 void* task_monitor(void* param) {
     printf("[%lld ms] [MONITOR] Started\n", get_time_ms());
     
     while (running) {
         sleep(3);
         pthread_mutex_lock(&message_queue.mutex);
         printf("\n[%lld ms] ═══ QUEUE STATUS ═══\n", get_time_ms());
         printf("  Items in queue: %d/%d\n", message_queue.count, QUEUE_SIZE);
         printf("  Usage: %.1f%%\n", 
                (message_queue.count * 100.0) / QUEUE_SIZE);
         printf("═══════════════════════════\n\n");
         pthread_mutex_unlock(&message_queue.mutex);
     }
     
     return NULL;
 }
 
 // Signal handler
 void signal_handler(int sig) {
     printf("\n\n[SHUTDOWN] Stopping all tasks...\n");
     running = false;
     pthread_cond_broadcast(&message_queue.not_empty);
     pthread_cond_broadcast(&message_queue.not_full);
 }
 
 int main() {
     pthread_t producers[2], consumers[2], monitor;
     int producer_ids[2] = {1, 2};
     int consumer_ids[2] = {1, 2};
     
     printf("╔════════════════════════════════════════════════════════╗\n");
     printf("║   RTOS Project 2: Producer-Consumer using Queue       ║\n");
     printf("╚════════════════════════════════════════════════════════╝\n\n");
     
     printf("Concepts demonstrated:\n");
     printf("  - Queue-based inter-task communication\n");
     printf("  - Blocking send/receive operations\n");
     printf("  - Thread synchronization with mutexes\n");
     printf("  - Producer-consumer pattern\n\n");
     
     printf("Configuration:\n");
     printf("  Queue size: %d items\n", QUEUE_SIZE);
     printf("  Producers: 2 tasks (simulating sensors)\n");
     printf("  Consumers: 2 tasks (processing data)\n\n");
     
     printf("Press Ctrl+C to stop\n");
     printf("════════════════════════════════════════════════════════\n\n");
     
     // Setup signal handler
     signal(SIGINT, signal_handler);
     srand(time(NULL));
     
     // Initialize queue
     queue_init(&message_queue);
     
     // Create producer tasks
     for (int i = 0; i < 2; i++) {
         pthread_create(&producers[i], NULL, task_producer, &producer_ids[i]);
     }
     
     // Create consumer tasks
     for (int i = 0; i < 2; i++) {
         pthread_create(&consumers[i], NULL, task_consumer, &consumer_ids[i]);
     }
     
     // Create monitor task
     pthread_create(&monitor, NULL, task_monitor, NULL);
     
     // Wait for all threads
     for (int i = 0; i < 2; i++) {
         pthread_join(producers[i], NULL);
         pthread_join(consumers[i], NULL);
     }
     pthread_join(monitor, NULL);
     
     // Cleanup
     pthread_mutex_destroy(&message_queue.mutex);
     pthread_cond_destroy(&message_queue.not_empty);
     pthread_cond_destroy(&message_queue.not_full);
     
     printf("\n[SHUTDOWN] All tasks stopped\n");
     return 0;
 }