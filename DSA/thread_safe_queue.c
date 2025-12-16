#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct ts_node {
    void *data;
    struct ts_node *next;
} ts_node_t;

typedef struct {
    ts_node_t *head, *tail;
    pthread_mutex_t mtx;
    pthread_cond_t not_empty;
    size_t size;
} ts_queue_t;

void ts_queue_init(ts_queue_t *q) {
    q->head = q->tail = NULL;
    q->size = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

void ts_queue_push(ts_queue_t *q, void *data) {
    ts_node_t *n = malloc(sizeof(ts_node_t));
    n->data = data;
    n->next = NULL;

    pthread_mutex_lock(&q->mtx);
    if (q->tail) q->tail->next = n; else q->head = n;
    q->tail = n;
    q->size++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

void *ts_queue_pop(ts_queue_t *q) {
    pthread_mutex_lock(&q->mtx);
    while (q->head == NULL) pthread_cond_wait(&q->not_empty, &q->mtx);
    ts_node_t *n = q->head;
    void *data = n->data;
    q->head = n->next;
    if (!q->head) q->tail = NULL;
    q->size--;
    pthread_mutex_unlock(&q->mtx);
    free(n);
    return data;
}

#define NUM_ITEMS 20

void *pusher(void *arg) {
    ts_queue_t *q = (ts_queue_t *)arg;
    for (long i = 0; i < NUM_ITEMS; i++) {
        long *val = malloc(sizeof(long));
        *val = i;
        ts_queue_push(q, val);
    }
    return NULL;
}

void *popper(void *arg) {
    ts_queue_t *q = (ts_queue_t *)arg;
    for (int i = 0; i < NUM_ITEMS; i++) {
        long *val = (long *)ts_queue_pop(q);
        printf("popped %ld\n", *val);
        free(val);
    }
    return NULL;
}

int main(void) {
    ts_queue_t q;
    ts_queue_init(&q);

    pthread_t t1, t2;
    pthread_create(&t1, NULL, pusher, &q);
    pthread_create(&t2, NULL, popper, &q);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Done. Final queue size = %zu (expected 0)\n", q.size);
    return 0;
}
