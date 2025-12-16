#include <stdio.h>
#include <pthread.h>

#define PC_CAP 8
#define NUM_ITEMS 30

typedef struct {
    int buf[PC_CAP];
    int head, tail, count;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty, not_full;
} pc_queue_t;

void pc_init(pc_queue_t *q) {
    q->head = q->tail = q->count = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

void pc_put(pc_queue_t *q, int val) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == PC_CAP) pthread_cond_wait(&q->not_full, &q->mtx);
    q->buf[q->tail] = val;
    q->tail = (q->tail + 1) % PC_CAP;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

int pc_get(pc_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == 0) pthread_cond_wait(&q->not_empty, &q->mtx);
    int val = q->buf[q->head];
    q->head = (q->head + 1) % PC_CAP;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return val;
}

void *producer(void *arg) {
    pc_queue_t *q = (pc_queue_t *)arg;
    for (int i = 0; i < NUM_ITEMS; i++) {
        pc_put(q, i);
        printf("[producer] produced %d\n", i);
    }
    return NULL;
}

void *consumer(void *arg) {
    pc_queue_t *q = (pc_queue_t *)arg;
    for (int i = 0; i < NUM_ITEMS; i++) {
        int v = pc_get(q);
        printf("            [consumer] consumed %d\n", v);
    }
    return NULL;
}

int main(void) {
    pc_queue_t q;
    pc_init(&q);

    pthread_t prod_t, cons_t;
    pthread_create(&prod_t, NULL, producer, &q);
    pthread_create(&cons_t, NULL, consumer, &q);

    pthread_join(prod_t, NULL);
    pthread_join(cons_t, NULL);

    printf("Done: all %d items produced and consumed.\n", NUM_ITEMS);
    return 0;
}
