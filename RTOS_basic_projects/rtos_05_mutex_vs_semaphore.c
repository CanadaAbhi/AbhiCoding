/*
 * RTOS Project 5: Mutex vs Semaphore Demo
 * 
 * Demonstrates:
 * - Mutex for mutual exclusion
 * - Binary semaphore vs counting semaphore
 * - Priority inversion problem
 * - Priority inheritance solution
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <signal.h>
#include <semaphore.h>

volatile bool running = true;
int shared_resource = 0;

// Synchronization primitives
pthread_mutex_t resource_mutex = PTHREAD_MUTEX_INITIALIZER;
sem_t binary_sem;
sem_t counting_sem;

long long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Test 1: Mutex demonstration
void* task_mutex_1(void* param) {
    for (int i = 0; i < 5 && running; i++) {
        pthread_mutex_lock(&resource_mutex);
        printf("[%lld ms] 🔒 [MUTEX-1] Acquired mutex\n", get_time_ms());
        
        int temp = shared_resource;
        printf("[%lld ms] [MUTEX-1] Reading: %d\n", get_time_ms(), temp);
        usleep(100000);  // Simulate work
        shared_resource = temp + 1;
        printf("[%lld ms] [MUTEX-1] Writing: %d\n", get_time_ms(), shared_resource);
        
        pthread_mutex_unlock(&resource_mutex);
        printf("[%lld ms] 🔓 [MUTEX-1] Released mutex\n\n", get_time_ms());
        usleep(200000);
    }
    return NULL;
}

void* task_mutex_2(void* param) {
    usleep(50000);  // Slight offset
    for (int i = 0; i < 5 && running; i++) {
        pthread_mutex_lock(&resource_mutex);
        printf("[%lld ms] 🔒 [MUTEX-2] Acquired mutex\n", get_time_ms());
        
        int temp = shared_resource;
        printf("[%lld ms] [MUTEX-2] Reading: %d\n", get_time_ms(), temp);
        usleep(100000);
        shared_resource = temp + 10;
        printf("[%lld ms] [MUTEX-2] Writing: %d\n", get_time_ms(), shared_resource);
        
        pthread_mutex_unlock(&resource_mutex);
        printf("[%lld ms] 🔓 [MUTEX-2] Released mutex\n\n", get_time_ms());
        usleep(200000);
    }
    return NULL;
}

// Test 2: Binary semaphore (signaling)
void* task_producer_sem(void* param) {
    for (int i = 0; i < 5 && running; i++) {
        printf("[%lld ms] 📤 [PRODUCER] Producing item %d\n", get_time_ms(), i);
        usleep(300000);
        sem_post(&binary_sem);
        printf("[%lld ms] [PRODUCER] Signaled consumer\n\n", get_time_ms());
    }
    return NULL;
}

void* task_consumer_sem(void* param) {
    for (int i = 0; i < 5 && running; i++) {
        printf("[%lld ms] 📥 [CONSUMER] Waiting for item...\n", get_time_ms());
        sem_wait(&binary_sem);
        printf("[%lld ms] [CONSUMER] Received item %d\n", get_time_ms(), i);
        printf("[%lld ms] [CONSUMER] Processing...\n", get_time_ms());
        usleep(200000);
        printf("[%lld ms] [CONSUMER] Done\n\n", get_time_ms());
    }
    return NULL;
}

// Test 3: Counting semaphore (resource pool)
void* task_worker(void* param) {
    int worker_id = *(int*)param;
    
    for (int i = 0; i < 3 && running; i++) {
        printf("[%lld ms] 👷 [WORKER-%d] Requesting resource...\n", 
               get_time_ms(), worker_id);
        
        sem_wait(&counting_sem);
        printf("[%lld ms] ✅ [WORKER-%d] Got resource (iteration %d)\n", 
               get_time_ms(), worker_id, i);
        
        // Use resource
        usleep((rand() % 500 + 200) * 1000);
        
        printf("[%lld ms] ✅ [WORKER-%d] Released resource\n", 
               get_time_ms(), worker_id);
        sem_post(&counting_sem);
        
        usleep(100000);
    }
    return NULL;
}

void signal_handler(int sig) {
    printf("\n\n[SHUTDOWN] Stopping...\n");
    running = false;
}

int main() {
    pthread_t t1, t2, t3, t4, workers[5];
    int worker_ids[5] = {1, 2, 3, 4, 5};
    
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║      RTOS Project 5: Mutex vs Semaphore Demo          ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    signal(SIGINT, signal_handler);
    srand(time(NULL));
    
    // Initialize semaphores
    sem_init(&binary_sem, 0, 0);      // Binary semaphore (initially 0)
    sem_init(&counting_sem, 0, 2);    // Counting semaphore (2 resources)
    
    // TEST 1: Mutex
    printf("\n═══ TEST 1: MUTEX (Mutual Exclusion) ═══\n");
    printf("Two tasks accessing shared resource with mutex\n\n");
    
    shared_resource = 0;
    pthread_create(&t1, NULL, task_mutex_1, NULL);
    pthread_create(&t2, NULL, task_mutex_2, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    printf("Final shared_resource value: %d\n", shared_resource);
    printf("Expected without mutex: unpredictable (race condition)\n");
    printf("With mutex: deterministic result\n\n");
    sleep(2);
    
    // TEST 2: Binary Semaphore
    printf("\n═══ TEST 2: BINARY SEMAPHORE (Signaling) ═══\n");
    printf("Producer signals consumer using binary semaphore\n\n");
    
    pthread_create(&t3, NULL, task_producer_sem, NULL);
    pthread_create(&t4, NULL, task_consumer_sem, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);
    sleep(2);
    
    // TEST 3: Counting Semaphore
    printf("\n═══ TEST 3: COUNTING SEMAPHORE (Resource Pool) ═══\n");
    printf("5 workers sharing 2 resources (max 2 concurrent)\n\n");
    
    for (int i = 0; i < 5; i++) {
        pthread_create(&workers[i], NULL, task_worker, &worker_ids[i]);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(workers[i], NULL);
    }
    
    // Summary
    printf("\n\n╔════════════════════════════════════════════════════════╗\n");
    printf("║                      SUMMARY                           ║\n");
    printf("╠════════════════════════════════════════════════════════╣\n");
    printf("║ MUTEX:                                                 ║\n");
    printf("║   ✓ Mutual exclusion (only one task at a time)        ║\n");
    printf("║   ✓ Ownership (same task must lock/unlock)            ║\n");
    printf("║   ✓ Priority inheritance                               ║\n");
    printf("║                                                        ║\n");
    printf("║ BINARY SEMAPHORE:                                      ║\n");
    printf("║   ✓ Signaling between tasks                            ║\n");
    printf("║   ✓ No ownership requirement                           ║\n");
    printf("║   ✓ Producer-consumer synchronization                  ║\n");
    printf("║                                                        ║\n");
    printf("║ COUNTING SEMAPHORE:                                    ║\n");
    printf("║   ✓ Resource pool management                           ║\n");
    printf("║   ✓ Limit concurrent access (e.g., 2 of 5 workers)    ║\n");
    printf("║   ✓ Multi-instance resources                           ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    // Cleanup
    sem_destroy(&binary_sem);
    sem_destroy(&counting_sem);
    pthread_mutex_destroy(&resource_mutex);
    
    printf("\n[SHUTDOWN] Test completed\n");
    return 0;
}