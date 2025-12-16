/*
 * RTOS Project 6: RTOS-Based Logger
 * Multiple tasks log messages to single logger task via queue
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>

#define LOG_QUEUE_SIZE 50
#define MAX_LOG_LEN 128

typedef enum { LOG_INFO, LOG_WARN, LOG_ERROR, LOG_DEBUG } LogLevel;

typedef struct {
    LogLevel level;
    long long timestamp;
    char message[MAX_LOG_LEN];
    int task_id;
} LogMessage;

typedef struct {
    LogMessage messages[LOG_QUEUE_SIZE];
    int head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
} LogQueue;

LogQueue log_queue;
volatile bool running = true;
FILE* log_file = NULL;

long long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void queue_init(LogQueue* q) {
    q->head = q->tail = q->count = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

bool log_enqueue(const char* msg, LogLevel level, int task_id) {
    pthread_mutex_lock(&log_queue.mutex);
    
    if (log_queue.count >= LOG_QUEUE_SIZE) {
        pthread_mutex_unlock(&log_queue.mutex);
        return false;
    }
    
    LogMessage* lm = &log_queue.messages[log_queue.tail];
    lm->level = level;
    lm->timestamp = get_time_ms();
    lm->task_id = task_id;
    strncpy(lm->message, msg, MAX_LOG_LEN - 1);
    
    log_queue.tail = (log_queue.tail + 1) % LOG_QUEUE_SIZE;
    log_queue.count++;
    
    pthread_cond_signal(&log_queue.not_empty);
    pthread_mutex_unlock(&log_queue.mutex);
    return true;
}

void* task_logger(void* param) {
    log_file = fopen("rtos_log.txt", "w");
    const char* level_str[] = {"INFO", "WARN", "ERROR", "DEBUG"};
    
    printf("[LOGGER] Started\n\n");
    
    while (running || log_queue.count > 0) {
        pthread_mutex_lock(&log_queue.mutex);
        
        while (log_queue.count == 0 && running) {
            pthread_cond_wait(&log_queue.not_empty, &log_queue.mutex);
        }
        
        if (log_queue.count > 0) {
            LogMessage* lm = &log_queue.messages[log_queue.head];
            
            printf("[%lld] [%s] Task-%d: %s\n", 
                   lm->timestamp, level_str[lm->level], lm->task_id, lm->message);
            
            if (log_file) {
                fprintf(log_file, "[%lld] [%s] Task-%d: %s\n",
                       lm->timestamp, level_str[lm->level], lm->task_id, lm->message);
                fflush(log_file);
            }
            
            log_queue.head = (log_queue.head + 1) % LOG_QUEUE_SIZE;
            log_queue.count--;
        }
        
        pthread_mutex_unlock(&log_queue.mutex);
    }
    
    if (log_file) fclose(log_file);
    printf("\n[LOGGER] Stopped\n");
    return NULL;
}

void* task_worker(void* param) {
    int id = *(int*)param;
    char msg[MAX_LOG_LEN];
    
    snprintf(msg, MAX_LOG_LEN, "Worker %d started", id);
    log_enqueue(msg, LOG_INFO, id);
    
    for (int i = 0; i < 10 && running; i++) {
        snprintf(msg, MAX_LOG_LEN, "Processing item %d", i);
        log_enqueue(msg, LOG_DEBUG, id);
        
        if (i % 3 == 0) {
            snprintf(msg, MAX_LOG_LEN, "Checkpoint reached at item %d", i);
            log_enqueue(msg, LOG_INFO, id);
        }
        
        if (rand() % 10 == 0) {
            snprintf(msg, MAX_LOG_LEN, "Warning: High load detected");
            log_enqueue(msg, LOG_WARN, id);
        }
        
        usleep((rand() % 200 + 100) * 1000);
    }
    
    snprintf(msg, MAX_LOG_LEN, "Worker %d completed", id);
    log_enqueue(msg, LOG_INFO, id);
    return NULL;
}

void signal_handler(int sig) {
    printf("\n\n[SHUTDOWN] Stopping...\n");
    running = false;
    pthread_cond_broadcast(&log_queue.not_empty);
}

int main() {
    pthread_t logger, workers[5];
    int ids[5] = {1, 2, 3, 4, 5};
    
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║         RTOS Project 6: RTOS-Based Logger             ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    signal(SIGINT, signal_handler);
    srand(time(NULL));
    queue_init(&log_queue);
    
    pthread_create(&logger, NULL, task_logger, NULL);
    
    for (int i = 0; i < 5; i++) {
        pthread_create(&workers[i], NULL, task_worker, &ids[i]);
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(workers[i], NULL);
    }
    
    running = false;
    pthread_cond_broadcast(&log_queue.not_empty);
    pthread_join(logger, NULL);
    
    printf("\nLog saved to rtos_log.txt\n");
    printf("[SHUTDOWN] Complete\n");
    return 0;
}