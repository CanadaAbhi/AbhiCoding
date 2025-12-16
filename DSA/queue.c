#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#define QUEUE_CAP 4 // small for demo

typedef struct {
    int data[QUEUE_CAP];
    size_t head, tail, count;
} queue_t;

void queue_init(queue_t *q) { q->head = q->tail = q->count = 0; }

bool queue_enqueue(queue_t *q, int val) {
    if (q->count == QUEUE_CAP) return false;
    q->data[q->tail] = val;
    q->tail = (q->tail + 1) % QUEUE_CAP;
    q->count++;
    return true;
}

bool queue_dequeue(queue_t *q, int *out) {
    if (q->count == 0) return false;
    *out = q->data[q->head];
    q->head = (q->head + 1) % QUEUE_CAP;
    q->count--;
    return true;
}

int main(void) {
    queue_t q;
    queue_init(&q);

    for (int i = 1; i <= 5; i++) {
        bool ok = queue_enqueue(&q, i);
        printf("enqueue(%d) = %s\n", i, ok ? "OK" : "FULL");
    }

    int val;
    printf("Dequeuing (expect 1 2 3 4): ");
    while (queue_dequeue(&q, &val)) printf("%d ", val);
    printf("\n");

    return 0;
}
