#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define TP_THREADS 4
#define TP_QUEUE_CAP 64
#define NUM_TASKS 20

typedef void (*task_fn_t)(void *arg);

typedef struct {
    task_fn_t fn;
    void *arg;
} tp_task_t;

typedef struct {
    tp_task_t queue[TP_QUEUE_CAP];
    int head, tail, count;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty, not_full;
    pthread_t workers[TP_THREADS];
    int shutdown;
} thread_pool_t;

static void *tp_worker(void *arg) {
    thread_pool_t *pool = (thread_pool_t *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (pool->count == 0 && !pool->shutdown)
            pthread_cond_wait(&pool->not_empty, &pool->mtx);

        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->mtx);
            break;
        }

        tp_task_t task = pool->queue[pool->head];
        pool->head = (pool->head + 1) % TP_QUEUE_CAP;
        pool->count--;
        pthread_cond_signal(&pool->not_full);
        pthread_mutex_unlock(&pool->mtx);

        task.fn(task.arg);
    }
    return NULL;
}

void tp_init(thread_pool_t *pool) {
    pool->head = pool->tail = pool->count = 0;
    pool->shutdown = 0;
    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->not_empty, NULL);
    pthread_cond_init(&pool->not_full, NULL);
    for (int i = 0; i < TP_THREADS; i++)
        pthread_create(&pool->workers[i], NULL, tp_worker, pool);
}

int tp_submit(thread_pool_t *pool, task_fn_t fn, void *arg) {
    pthread_mutex_lock(&pool->mtx);
    while (pool->count == TP_QUEUE_CAP && !pool->shutdown)
        pthread_cond_wait(&pool->not_full, &pool->mtx);
    if (pool->shutdown) { pthread_mutex_unlock(&pool->mtx); return -1; }

    pool->queue[pool->tail].fn = fn;
    pool->queue[pool->tail].arg = arg;
    pool->tail = (pool->tail + 1) % TP_QUEUE_CAP;
    pool->count++;
    pthread_cond_signal(&pool->not_empty);
    pthread_mutex_unlock(&pool->mtx);
    return 0;
}

void tp_shutdown(thread_pool_t *pool) {
    pthread_mutex_lock(&pool->mtx);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->not_empty);
    pthread_mutex_unlock(&pool->mtx);
    for (int i = 0; i < TP_THREADS; i++)
        pthread_join(pool->workers[i], NULL);
}

void example_task(void *arg) {
    long id = (long)arg;
    printf("[thread %lu] executing task %ld\n", (unsigned long)pthread_self(), id);
    usleep(10000); // simulate work
}

int main(void) {
    thread_pool_t pool;
    tp_init(&pool);

    for (long i = 0; i < NUM_TASKS; i++) {
        tp_submit(&pool, example_task, (void *)i);
    }

    tp_shutdown(&pool);
    printf("All %d tasks completed, pool shut down.\n", NUM_TASKS);
    return 0;
}
