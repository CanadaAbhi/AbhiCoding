#include <stdio.h>
#include <stdbool.h>

#define STACK_CAP 4 // small for demo

typedef struct {
    int data[STACK_CAP];
    int top;
} stack_t;

void stack_init(stack_t *s) { s->top = -1; }

bool stack_push(stack_t *s, int val) {
    if (s->top >= STACK_CAP - 1) return false;
    s->data[++s->top] = val;
    return true;
}

bool stack_pop(stack_t *s, int *out) {
    if (s->top < 0) return false;
    *out = s->data[s->top--];
    return true;
}

bool stack_peek(const stack_t *s, int *out) {
    if (s->top < 0) return false;
    *out = s->data[s->top];
    return true;
}

int main(void) {
    stack_t s;
    stack_init(&s);

    for (int i = 1; i <= 5; i++) {
        bool ok = stack_push(&s, i);
        printf("push(%d) = %s\n", i, ok ? "OK" : "OVERFLOW");
    }

    int val;
    printf("Popping (expect 4 3 2 1): ");
    while (stack_pop(&s, &val)) printf("%d ", val);
    printf("\n");

    bool empty_pop = stack_pop(&s, &val);
    printf("Pop on empty stack returns: %s\n", empty_pop ? "true (BUG)" : "false (correct)");

    return 0;
}
