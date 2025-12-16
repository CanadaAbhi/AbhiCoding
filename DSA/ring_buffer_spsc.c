#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define RB_SIZE 8 // power of two, small for demo
#define RB_MASK (RB_SIZE - 1)

typedef struct {
    uint8_t buf[RB_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} ring_buf_t;

void rb_init(ring_buf_t *rb) { rb->head = rb->tail = 0; }

bool rb_put(ring_buf_t *rb, uint8_t byte) {
    uint32_t head = rb->head;
    uint32_t next = (head + 1) & RB_MASK;
    if (next == rb->tail) return false;
    rb->buf[head] = byte;
    rb->head = next;
    return true;
}

bool rb_get(ring_buf_t *rb, uint8_t *byte) {
    uint32_t tail = rb->tail;
    if (tail == rb->head) return false;
    *byte = rb->buf[tail];
    rb->tail = (tail + 1) & RB_MASK;
    return true;
}

int main(void) {
    ring_buf_t rb;
    rb_init(&rb);

    // Fill to capacity (RB_SIZE - 1 usable slots due to reserved-slot design)
    for (uint8_t i = 0; i < RB_SIZE; i++) {
        bool ok = rb_put(&rb, i);
        printf("put(%u) = %s\n", i, ok ? "OK" : "DROPPED (full)");
    }

    uint8_t val;
    printf("Draining: ");
    while (rb_get(&rb, &val)) printf("%u ", val);
    printf("\n");

    return 0;
}
