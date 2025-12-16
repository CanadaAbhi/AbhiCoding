#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>

#define NUM_THREADS 4
#define INCREMENTS_PER_THREAD 100000

typedef struct {
    atomic_int value;
} atomic_counter_t;

void counter_init(atomic_counter_t *c, int initial) {
    atomic_init(&c->value, initial);
}

int counter_increment(atomic_counter_t *c) {
    return atomic_fetch_add(&c->value, 1) + 1;
}

int counter_get(atomic_counter_t *c) {
    return atomic_load(&c->value);
}

int counter_bounded_increment(atomic_counter_t *c, int max) {
    int cur = atomic_load(&c->value);
    while (cur < max) {
        if (atomic_compare_exchange_weak(&c->value, &cur, cur + 1))
            return cur + 1;
    }
    return cur;
}

void *worker(void *arg) {
    atomic_counter_t *c = (atomic_counter_t *)arg;
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        counter_increment(c);
    }
    return NULL;
}

int main(void) {
    atomic_counter_t counter;
    counter_init(&counter, 0);

    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, worker, &counter);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    int expected = NUM_THREADS * INCREMENTS_PER_THREAD;
    int got = counter_get(&counter);
    printf("Counter final value = %d (expected %d) %s\n",
           got, expected, got == expected ? "PASS" : "FAIL");

    // Bounded CAS demo
    atomic_counter_t bounded;
    counter_init(&bounded, 0);
    for (int i = 0; i < 10; i++) {
        int r = counter_bounded_increment(&bounded, 5);
        printf("bounded_increment -> %d\n", r);
    }

    return 0;
}
