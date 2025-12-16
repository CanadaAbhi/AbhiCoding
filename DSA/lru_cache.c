#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LRU_CAP 3 // small for demo
#define LRU_BUCKETS 16

typedef struct lru_node {
    int key, value;
    struct lru_node *prev, *next;
    struct lru_node *hnext;
} lru_node_t;

typedef struct {
    lru_node_t pool[LRU_CAP];
    lru_node_t *free_list;
    lru_node_t *buckets[LRU_BUCKETS];
    lru_node_t *head, *tail;
    int count;
} lru_cache_t;

static unsigned lru_hash(int key) { return ((unsigned)key * 2654435761u) % LRU_BUCKETS; }

void lru_init(lru_cache_t *c) {
    memset(c, 0, sizeof(*c));
    for (int i = 0; i < LRU_CAP - 1; i++) c->pool[i].hnext = &c->pool[i + 1];
    c->free_list = &c->pool[0];
    c->head = c->tail = NULL;
}

static void lru_detach(lru_cache_t *c, lru_node_t *n) {
    if (n->prev) n->prev->next = n->next; else c->head = n->next;
    if (n->next) n->next->prev = n->prev; else c->tail = n->prev;
    n->prev = n->next = NULL;
}

static void lru_push_front(lru_cache_t *c, lru_node_t *n) {
    n->prev = NULL;
    n->next = c->head;
    if (c->head) c->head->prev = n;
    c->head = n;
    if (!c->tail) c->tail = n;
}

static lru_node_t *lru_find(lru_cache_t *c, int key) {
    unsigned idx = lru_hash(key);
    for (lru_node_t *n = c->buckets[idx]; n; n = n->hnext) {
        if (n->key == key) return n;
    }
    return NULL;
}

int lru_get(lru_cache_t *c, int key, int *out) {
    lru_node_t *n = lru_find(c, key);
    if (!n) return 0;
    lru_detach(c, n);
    lru_push_front(c, n);
    *out = n->value;
    return 1;
}

void lru_put(lru_cache_t *c, int key, int value) {
    lru_node_t *n = lru_find(c, key);
    if (n) {
        n->value = value;
        lru_detach(c, n);
        lru_push_front(c, n);
        return;
    }

    if (c->count == LRU_CAP) {
        lru_node_t *victim = c->tail;
        printf("  [evicting key=%d]\n", victim->key);
        lru_detach(c, victim);
        unsigned vidx = lru_hash(victim->key);
        lru_node_t **pp = &c->buckets[vidx];
        while (*pp != victim) pp = &(*pp)->hnext;
        *pp = victim->hnext;
        victim->hnext = c->free_list;
        c->free_list = victim;
        c->count--;
    }

    lru_node_t *fresh = c->free_list;
    c->free_list = fresh->hnext;
    fresh->key = key;
    fresh->value = value;

    unsigned idx = lru_hash(key);
    fresh->hnext = c->buckets[idx];
    c->buckets[idx] = fresh;

    lru_push_front(c, fresh);
    c->count++;
}

int main(void) {
    lru_cache_t cache;
    lru_init(&cache);

    lru_put(&cache, 1, 100);
    lru_put(&cache, 2, 200);
    lru_put(&cache, 3, 300); // cache full: [3,2,1]

    int val;
    lru_get(&cache, 1, &val); // touch 1 -> [1,3,2]
    printf("get(1) = %d\n", val);

    lru_put(&cache, 4, 400); // should evict key=2 (LRU)

    printf("get(2) = %s (expected NOT FOUND)\n",
           lru_get(&cache, 2, &val) ? "FOUND (BUG)" : "NOT FOUND");
    printf("get(3) = %d\n", lru_get(&cache, 3, &val) ? val : -1);
    printf("get(4) = %d\n", lru_get(&cache, 4, &val) ? val : -1);

    return 0;
}
