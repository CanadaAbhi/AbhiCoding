#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define FSA_BLOCK_SIZE 64
#define FSA_BLOCK_COUNT 8 // small for demo

typedef struct fsa_block {
    struct fsa_block *next;
} fsa_block_t;

typedef struct {
    uint8_t pool[FSA_BLOCK_COUNT][FSA_BLOCK_SIZE];
    fsa_block_t *free_list;
    size_t free_count;
} fixed_allocator_t;

void fsa_init(fixed_allocator_t *a) {
    a->free_list = NULL;
    for (int i = FSA_BLOCK_COUNT - 1; i >= 0; i--) {
        fsa_block_t *b = (fsa_block_t *)a->pool[i];
        b->next = a->free_list;
        a->free_list = b;
    }
    a->free_count = FSA_BLOCK_COUNT;
}

void *fsa_alloc(fixed_allocator_t *a) {
    if (!a->free_list) return NULL;
    fsa_block_t *b = a->free_list;
    a->free_list = b->next;
    a->free_count--;
    memset(b, 0, FSA_BLOCK_SIZE);
    return (void *)b;
}

void fsa_free(fixed_allocator_t *a, void *ptr) {
    if (!ptr) return;
    fsa_block_t *b = (fsa_block_t *)ptr;
    b->next = a->free_list;
    a->free_list = b;
    a->free_count++;
}

int main(void) {
    fixed_allocator_t alloc;
    fsa_init(&alloc);

    printf("Initial free_count = %zu\n", alloc.free_count);

    void *ptrs[FSA_BLOCK_COUNT + 1];
    for (int i = 0; i < FSA_BLOCK_COUNT; i++) {
        ptrs[i] = fsa_alloc(&alloc);
        printf("alloc() -> %p, free_count = %zu\n", ptrs[i], alloc.free_count);
    }

    void *overflow = fsa_alloc(&alloc);
    printf("Pool-exhausted alloc() -> %p (expected NULL)\n", overflow);

    fsa_free(&alloc, ptrs[0]);
    fsa_free(&alloc, ptrs[1]);
    printf("After freeing 2 blocks, free_count = %zu\n", alloc.free_count);

    void *reused = fsa_alloc(&alloc);
    printf("Re-alloc after free -> %p (should reuse freed block)\n", reused);

    return 0;
}
