#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

#define CB_CAPACITY 4 // small for demo

typedef struct {
    int buf[CB_CAPACITY];
    size_t head, tail, count;
} circ_buf_t;

void cb_init(circ_buf_t *cb) { cb->head = cb->tail = cb->count = 0; }

void cb_push(circ_buf_t *cb, int val) {
    cb->buf[cb->head] = val;
    cb->head = (cb->head + 1) % CB_CAPACITY;
    if (cb->count < CB_CAPACITY) {
        cb->count++;
    } else {
        cb->tail = (cb->tail + 1) % CB_CAPACITY;
    }
}

bool cb_pop(circ_buf_t *cb, int *out) {
    if (cb->count == 0) return false;
    *out = cb->buf[cb->tail];
    cb->tail = (cb->tail + 1) % CB_CAPACITY;
    cb->count--;
    return true;
}

int main(void) {
    circ_buf_t cb;
    cb_init(&cb);

    for (int i = 1; i <= 6; i++) {
        printf("Pushing %d\n", i);
        cb_push(&cb, i); // capacity is 4, so 1 and 2 get overwritten
    }

    int val;
    printf("Draining buffer (expect 3 4 5 6): ");
    while (cb_pop(&cb, &val)) printf("%d ", val);
    printf("\n");

    return 0;
}
