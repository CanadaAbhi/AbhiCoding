#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HT_BUCKETS 16 // small for demo

typedef struct ht_entry {
    char *key;
    int value;
    struct ht_entry *next;
} ht_entry_t;

typedef struct {
    ht_entry_t *buckets[HT_BUCKETS];
} hash_table_t;

static unsigned long ht_hash(const char *key) {
    unsigned long h = 5381;
    while (*key) h = ((h << 5) + h) + (unsigned char)(*key++);
    return h % HT_BUCKETS;
}

void ht_init(hash_table_t *ht) { memset(ht->buckets, 0, sizeof(ht->buckets)); }

void ht_put(hash_table_t *ht, const char *key, int value) {
    unsigned long idx = ht_hash(key);
    for (ht_entry_t *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { e->value = value; return; }
    }
    ht_entry_t *e = malloc(sizeof(ht_entry_t));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

int ht_get(hash_table_t *ht, const char *key, int *out) {
    unsigned long idx = ht_hash(key);
    for (ht_entry_t *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    }
    return 0;
}

void ht_free(hash_table_t *ht) {
    for (int i = 0; i < HT_BUCKETS; i++) {
        ht_entry_t *e = ht->buckets[i];
        while (e) {
            ht_entry_t *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
        ht->buckets[i] = NULL;
    }
}

int main(void) {
    hash_table_t ht;
    ht_init(&ht);

    ht_put(&ht, "arm", 1);
    ht_put(&ht, "nvidia", 2);
    ht_put(&ht, "qualcomm", 3);
    ht_put(&ht, "arm", 100); // update

    int val;
    const char *keys[] = { "arm", "nvidia", "qualcomm", "missing" };
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        if (ht_get(&ht, keys[i], &val))
            printf("%s -> %d\n", keys[i], val);
        else
            printf("%s -> NOT FOUND\n", keys[i]);
    }

    ht_free(&ht);
    return 0;
}
