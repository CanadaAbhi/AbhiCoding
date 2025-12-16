# Cache Hit/Miss Counter with Access Pattern Analysis

## Report Format (exact spec)

```
Total accesses : 100000
Cache hits     : 85000
Cache misses   : 15000
Hit rate       : 85%
Miss rate      : 15%
```

The numbers above are just the **template** — actual hit/miss counts depend entirely on the access pattern's spatial/temporal locality relative to the cache's line size and capacity. The simulator below measures the *real* numbers for four distinct patterns and explains the divergence.

## Implementation

```c
/* ============================================================
 * cache_hitmiss_counter.c
 * Direct-mapped cache hit/miss counter with pluggable access
 * pattern generators. Zero heap allocation, deterministic.
 * ============================================================ */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define CACHE_SIZE    32768u          /* 32KB cache            */
#define LINE_SIZE     64u             /* 64B per line          */
#define NUM_LINES     (CACHE_SIZE / LINE_SIZE)      /* 512 lines   */

#define ARRAY_ELEMS   65536u          /* array size in ints (256KB = 8x cache) */
#define ELEM_SIZE     4u              /* sizeof(int)           */
#define ELEMS_PER_LINE (LINE_SIZE / ELEM_SIZE)      /* 16 ints/line */

#define NUM_ACCESSES  100000u

typedef struct {
    uint32_t tag[NUM_LINES];   /* static pool -- no malloc */
    uint32_t valid[NUM_LINES];
} Cache;

typedef struct {
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
} CacheStats;

static void cache_reset(Cache *c, CacheStats *s)
{
    memset(c, 0, sizeof(*c));
    memset(s, 0, sizeof(*s));
}

/* index = (byte_addr / LINE_SIZE) % NUM_LINES
 * tag   = (byte_addr / LINE_SIZE) / NUM_LINES              */
static int cache_access(Cache *c, CacheStats *s, uint32_t byte_addr)
{
    uint32_t line_num  = byte_addr / LINE_SIZE;
    uint32_t index      = line_num % NUM_LINES;
    uint32_t tag         = line_num / NUM_LINES;

    s->accesses++;

    if (c->valid[index] && c->tag[index] == tag) {
        s->hits++;
        return 1;   /* HIT */
    }

    /* miss -> load this line (evicts whatever was in `index` before) */
    c->valid[index] = 1;
    c->tag[index]   = tag;
    s->misses++;
    return 0;       /* MISS */
}

static void print_report(const char *pattern_name, const CacheStats *s)
{
    double hit_rate  = 100.0 * (double)s->hits   / (double)s->accesses;
    double miss_rate = 100.0 * (double)s->misses / (double)s->accesses;

    printf("=== Pattern: %s ===\n", pattern_name);
    printf("Total accesses : %llu\n", (unsigned long long)s->accesses);
    printf("Cache hits     : %llu\n", (unsigned long long)s->hits);
    printf("Cache misses   : %llu\n", (unsigned long long)s->misses);
    printf("Hit rate       : %.2f%%\n", hit_rate);
    printf("Miss rate      : %.2f%%\n\n", miss_rate);
}

/* xorshift32 -- deterministic PRNG, no rand()/srand() dependency */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}
```

## Four Access Pattern Generators

```c
/* Pattern 1: array[i]  -- pure sequential unit stride */
static void pattern_sequential(Cache *c, CacheStats *s)
{
    cache_reset(c, s);
    for (uint32_t n = 0; n < NUM_ACCESSES; n++) {
        uint32_t i = n % ARRAY_ELEMS;
        uint32_t addr = i * ELEM_SIZE;
        cache_access(c, s, addr);
    }
}

/* Pattern 2: array[i+1] -- unit stride, shifted by 1 element */
static void pattern_offset_by_one(Cache *c, CacheStats *s)
{
    cache_reset(c, s);
    for (uint32_t n = 0; n < NUM_ACCESSES; n++) {
        uint32_t i = (n + 1) % ARRAY_ELEMS;
        uint32_t addr = i * ELEM_SIZE;
        cache_access(c, s, addr);
    }
}

/* Pattern 3: array[i+16] -- stride of EXACTLY one cache line (16 ints = 64B) */
static void pattern_stride_16(Cache *c, CacheStats *s)
{
    cache_reset(c, s);
    for (uint32_t n = 0; n < NUM_ACCESSES; n++) {
        uint32_t i = (n * 16u) % ARRAY_ELEMS;
        uint32_t addr = i * ELEM_SIZE;
        cache_access(c, s, addr);
    }
}

/* Pattern 4: array[random_index] -- uniform random, defeats locality */
static void pattern_random(Cache *c, CacheStats *s)
{
    cache_reset(c, s);
    uint32_t rng_state = 0xACE1u;   /* fixed seed for reproducibility */
    for (uint32_t n = 0; n < NUM_ACCESSES; n++) {
        uint32_t i = xorshift32(&rng_state) % ARRAY_ELEMS;
        uint32_t addr = i * ELEM_SIZE;
        cache_access(c, s, addr);
    }
}

int main(void)
{
    static Cache c;        /* static -- avoids stack overflow, 512*8B ≈ 4KB anyway */
    CacheStats s;

    pattern_sequential(&c, &s);
    print_report("array[i]  (sequential)", &s);

    pattern_offset_by_one(&c, &s);
    print_report("array[i+1] (offset unit stride)", &s);

    pattern_stride_16(&c, &s);
    print_report("array[i+16] (stride == cache line size)", &s);

    pattern_random(&c, &s);
    print_report("array[random_index]", &s);

    return 0;
}
```

## Expected Output

```
=== Pattern: array[i]  (sequential) ===
Total accesses : 100000
Cache hits     : 93750
Cache misses   : 6250
Hit rate       : 93.75%
Miss rate      : 6.25%

=== Pattern: array[i+1] (offset unit stride) ===
Total accesses : 100000
Cache hits     : 93750
Cache misses   : 6250
Hit rate       : 93.75%
Miss rate      : 6.25%

=== Pattern: array[i+16] (stride == cache line size) ===
Total accesses : 100000
Cache hits     : 0
Cache misses   : 100000
Hit rate       : 0.00%
Miss rate      : 100.00%

=== Pattern: array[random_index] ===
Total accesses : 100000
Cache hits     : ~12500
Cache misses   : ~87500
Hit rate       : ~12.50%
Miss rate      : ~87.50%
```

## Why Each Number Comes Out That Way

| Pattern | Elements/line touched consecutively | Hit rate | Root cause |
|---|---|---|---|
| `array[i]` | 16 (all of them, in order) | **93.75%** (15/16) | 1 compulsory miss loads the line, next 15 sequential accesses land in the *same* 64B line → hit. Exactly `(ELEMS_PER_LINE-1)/ELEMS_PER_LINE`. |
| `array[i+1]` | 16 (shifted by 1) | **93.75%** | Identical spatial-locality structure — the +1 offset shifts *which* line boundary falls where, but doesn't change the steady-state fraction of intra-line hits. |
| `array[i+16]` | **1** | **0%** | Stride of 16 ints = 64 bytes = exactly `LINE_SIZE`. Every single access lands on a brand-new line; the other 15 words of each line are *never revisited* before the stream moves on. This is the textbook "stride aliases with line size" pathology. |
| `array[random_index]` | statistically ~1 | **~12.5%** (≈1/8) | Array is 8× the cache size (256KB / 32KB). Under uniform random selection, `1/8` of accesses happen to redraw the exact line currently sitting in that set before it gets evicted by one of the other 7 lines competing for the same direct-mapped slot. Hit rate ≈ `cache_size / working_set_size`. |

## Key Insight

This is the same lesson from the direct-mapped/set-associative/fully-associative and LRU/FIFO/Random work in this series, now viewed through the **stride** lens instead of the **replacement policy** lens:

- **Stride < line size** (patterns 1 & 2): spatial locality is exploited automatically — you get "free" hits on every word within a fetched line.
- **Stride == line size** (pattern 3): this is the *worst possible* stride — it guarantees you touch exactly one word per line and never come back, so you pay full miss cost while wasting 15/16 of every fetched line's bandwidth. This is a real-world bug pattern (e.g., iterating a 2D array's outer dimension when the row size equals the cache line size, or scanning a struct array with padding that happens to equal 64B).
- **Random access** (pattern 4): no exploitable structure at all — hit rate collapses to roughly `cache_size / footprint_size`, which is the fundamental capacity-miss floor no replacement policy can fix (as already shown in the FIFO/LRU/Random comparison — policy only matters *when* there's locality to exploit in the first place).

---

Would you like this **Hit/Miss Counter** combined with the earlier **Direct-Mapped, N-way Set-Associative, Fully-Associative, LRU, and FIFO/Random** simulators into a single downloadable `.docx` "Cache Simulator Suite" reference document for your portfolio/interview prep?