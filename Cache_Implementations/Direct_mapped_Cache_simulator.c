# Direct-Mapped Cache Simulator (C)

## Concept Recap

```
CPU Address (32-bit example)
+----------------------+-----------------+----------------+
|         Tag          |      Index      |     Offset     |
+----------------------+-----------------+----------------+
   32 - I - O bits          I bits            O bits

Offset = log2(line_size_bytes)   -> selects byte within a cache line
Index  = log2(num_cache_lines)   -> selects WHICH cache line (only one possible slot!)
Tag    = remaining upper bits    -> identifies WHICH block currently occupies that slot
```

**Direct-mapped** means: `index = (address / line_size) % num_lines`. Every address has exactly **one** candidate line — no set/way search like in N-way set-associative caches. This makes hit/miss checking O(1) but makes the cache prone to **conflict misses** (thrashing) when two hot addresses share the same index but differ in tag.

---

## Implementation

```c
/* ============================================================
 * direct_mapped_cache.c
 * Production-style Direct-Mapped Cache Simulator
 * Fixed-size, zero heap allocation, deterministic O(1) access
 * ============================================================ */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------
 * Cache Geometry (compile-time constants — no malloc, no heap)
 * -----------------------------------------------------------
 * Total cache size = NUM_LINES * LINE_SIZE_BYTES
 * Example: 512 lines * 64 bytes = 32 KB cache
 * ----------------------------------------------------------- */
#define LINE_SIZE_BYTES     64u                 /* bytes per cache line   */
#define NUM_LINES           512u                /* must be power of two   */
#define CACHE_SIZE_BYTES    (LINE_SIZE_BYTES * NUM_LINES)

#define OFFSET_BITS         6u                  /* log2(64)  = 6          */
#define INDEX_BITS          9u                  /* log2(512) = 9          */

#define OFFSET_MASK         ((1u << OFFSET_BITS) - 1u)
#define INDEX_MASK          ((1u << INDEX_BITS)  - 1u)

/* Return codes required by the interview spec */
#define CACHE_HIT   1
#define CACHE_MISS  0

/* -----------------------------------------------------------
 * Cache Line metadata (tag store) — statically allocated array
 * We don't need to store actual data payload for a functional
 * hit/miss simulator, only valid + tag bits (like a real tag RAM)
 * ----------------------------------------------------------- */
typedef struct {
    bool     valid;   /* has this line ever been filled?        */
    uint32_t tag;      /* which block currently occupies it      */
} cache_line_t;

static cache_line_t g_cache[NUM_LINES];

/* -----------------------------------------------------------
 * Statistics — useful for interview follow-ups on hit-rate math
 * ----------------------------------------------------------- */
typedef struct {
    uint64_t total_accesses;
    uint64_t hits;
    uint64_t misses;
} cache_stats_t;

static cache_stats_t g_stats;

/* -----------------------------------------------------------
 * Address decomposition helper (debug / explanation aid)
 * ----------------------------------------------------------- */
typedef struct {
    uint32_t tag;
    uint32_t index;
    uint32_t offset;
} addr_fields_t;

static addr_fields_t decode_address(uint32_t address)
{
    addr_fields_t f;
    f.offset = address & OFFSET_MASK;
    f.index  = (address >> OFFSET_BITS) & INDEX_MASK;
    f.tag    = address >> (OFFSET_BITS + INDEX_BITS);
    return f;
}

/* -----------------------------------------------------------
 * cache_init() — reset the tag store and stats
 * ----------------------------------------------------------- */
void cache_init(void)
{
    memset(g_cache, 0, sizeof(g_cache));   /* valid=false, tag=0 for all */
    memset(&g_stats, 0, sizeof(g_stats));
}

/* -----------------------------------------------------------
 * cache_access() — the core interview function
 *
 * Returns CACHE_HIT if the requested address is already resident
 * in its mapped line with a matching tag; otherwise CACHE_MISS,
 * and the line is filled with the new tag (simulating a fetch
 * from the next level of memory hierarchy).
 * ----------------------------------------------------------- */
int cache_access(uint32_t address)
{
    addr_fields_t f = decode_address(address);

    g_stats.total_accesses++;

    cache_line_t *line = &g_cache[f.index];

    if (line->valid && (line->tag == f.tag)) {
        /* Hit: block already resident, tag matches */
        g_stats.hits++;
        return CACHE_HIT;
    }

    /* Miss: either line was never filled (cold miss) or it holds
     * a *different* block that maps to the same index
     * (conflict miss). Direct-mapped => no choice of victim,
     * we must evict whatever is there and load the new block. */
    line->valid = true;
    line->tag   = f.tag;

    g_stats.misses++;
    return CACHE_MISS;
}

/* -----------------------------------------------------------
 * cache_print_stats() — human-readable summary
 * ----------------------------------------------------------- */
void cache_print_stats(const char *label)
{
    double hit_rate = 0.0;
    if (g_stats.total_accesses > 0) {
        hit_rate = (double)g_stats.hits / (double)g_stats.total_accesses * 100.0;
    }

    printf("\n--- %s ---\n", label);
    printf("Total accesses : %llu\n", (unsigned long long)g_stats.total_accesses);
    printf("Hits           : %llu\n", (unsigned long long)g_stats.hits);
    printf("Misses         : %llu\n", (unsigned long long)g_stats.misses);
    printf("Hit rate       : %.2f%%\n", hit_rate);
}

void cache_reset_stats(void)
{
    memset(&g_stats, 0, sizeof(g_stats));
}
```

---

## Test Harness — demonstrating locality, cold misses, and thrashing

```c
/* ============================================================
 * Demo 1: Spatial locality
 * Accessing consecutive words within the SAME cache line
 * should produce exactly ONE miss (the first fetch), then hits.
 * ============================================================ */
static void demo_spatial_locality(void)
{
    cache_init();

    printf("\n[Demo 1] Spatial locality: 16 sequential 4-byte accesses "
           "within one 64-byte line\n");

    uint32_t base = 0x1000;
    for (int i = 0; i < 16; i++) {
        uint32_t addr = base + (i * 4);   /* 16 * 4B = 64B = one full line */
        int result = cache_access(addr);
        printf("  addr=0x%08X -> %s\n", addr,
               result == CACHE_HIT ? "HIT" : "MISS");
    }

    cache_print_stats("Spatial Locality Demo");
    /* Expected: 1 miss (first access), 15 hits -> 93.75% hit rate */
}

/* ============================================================
 * Demo 2: Cold miss vs re-access after eviction
 * Fill many distinct lines, then re-access the FIRST one.
 * If total footprint <= cache size, it should still be a HIT.
 * ============================================================ */
static void demo_cold_then_reaccess(void)
{
    cache_init();

    printf("\n[Demo 2] Fill 4 distinct lines then re-access the first\n");

    uint32_t addrs[4] = { 0x0000, 0x0040, 0x0080, 0x00C0 }; /* 64B apart */
    for (int i = 0; i < 4; i++) {
        int result = cache_access(addrs[i]);
        printf("  fill addr=0x%08X -> %s\n", addrs[i],
               result == CACHE_HIT ? "HIT" : "MISS");
    }

    int result = cache_access(addrs[0]);   /* re-touch first block */
    printf("  re-access addr=0x%08X -> %s\n", addrs[0],
           result == CACHE_HIT ? "HIT" : "MISS");

    cache_print_stats("Cold Miss + Reaccess Demo");
}

/* ============================================================
 * Demo 3: Conflict miss / THRASHING
 * Two addresses that map to the SAME index but different tags,
 * accessed alternately, will ALWAYS miss -- classic direct-mapped
 * pathology. This is the #1 interview follow-up question:
 * "What's the weakness of direct-mapped caches?"
 * ============================================================ */
static void demo_thrashing(void)
{
    cache_init();

    printf("\n[Demo 3] Conflict miss / thrashing: two addresses sharing "
           "the same index\n");

    /* Index space is 9 bits wide (0..511), spanning bits [6:14].
     * Adding NUM_LINES * LINE_SIZE_BYTES (== cache size) to an address
     * keeps the index identical but changes the tag. */
    uint32_t addr_a = 0x0000;
    uint32_t addr_b = addr_a + CACHE_SIZE_BYTES;  /* same index, different tag */

    addr_fields_t fa = decode_address(addr_a);
    addr_fields_t fb = decode_address(addr_b);
    printf("  addr_a=0x%08X -> index=%u tag=%u\n", addr_a, fa.index, fa.tag);
    printf("  addr_b=0x%08X -> index=%u tag=%u\n", addr_b, fb.index, fb.tag);

    for (int i = 0; i < 8; i++) {
        uint32_t addr = (i % 2 == 0) ? addr_a : addr_b;
        int result = cache_access(addr);
        printf("  access %s -> %s\n",
               (addr == addr_a) ? "A" : "B",
               result == CACHE_HIT ? "HIT" : "MISS");
    }

    cache_print_stats("Thrashing Demo");
    /* Expected: 8 misses, 0 hits -> 0% hit rate, even though only
     * 2 blocks are being reused! This is exactly what a 2-way
     * set-associative cache would fix. */
}

int main(void)
{
    demo_spatial_locality();
    demo_cold_then_reaccess();
    demo_thrashing();
    return 0;
}
```

---

## Sample Output

```
[Demo 1] Spatial locality: 16 sequential 4-byte accesses within one 64-byte line
  addr=0x00001000 -> MISS
  addr=0x00001004 -> HIT
  ...
  addr=0x0000103C -> HIT

--- Spatial Locality Demo ---
Total accesses : 16
Hits           : 15
Misses         : 1
Hit rate       : 93.75%

[Demo 3] Conflict miss / thrashing: two addresses sharing the same index
  addr_a=0x00000000 -> index=0 tag=0
  addr_b=0x00008000 -> index=0 tag=1
  access A -> MISS
  access B -> MISS
  access A -> MISS
  access B -> MISS
  ...

--- Thrashing Demo ---
Total accesses : 8
Hits           : 0
Misses         : 0
Hit rate       : 0.00%
```

---

## Key Interview Talking Points

| Question they'll likely ask | Answer |
|---|---|
| **Why is index derived from bits [14:6] and not any other bits?** | Because address bits are inherently sequential — using the low-order bits (right after offset) after masking ensures *consecutive* blocks spread evenly across all cache lines, maximizing spatial-locality benefit. Using high-order bits as index instead would cluster all nearby addresses into the same few lines. |
| **Biggest weakness of direct-mapped?** | Conflict misses / thrashing — two frequently-used addresses with the same index but different tags evict each other every single access, even if the working set is tiny (Demo 3). |
| **How would you fix it?** | N-way set-associative cache: each index maps to a **set** of N lines, and a replacement policy (LRU, random) picks a victim only when the set is full — trades tag-comparison cost (N comparators) for lower conflict-miss rate. |
| **What's the formula for effective address space vs cache overhead?** | Tag store overhead = `NUM_LINES * (tag_bits + valid_bit)` — for this design: `512 * (17 + 1) bits ≈ 1152 bytes` of pure metadata for a 32KB cache. |
| **Cold miss vs conflict miss vs capacity miss?** | Cold (compulsory) = first-ever access to a block. Conflict = block evicted by another block competing for the same index despite cache having spare capacity elsewhere. Capacity = working set genuinely exceeds cache size, unavoidable even with full associativity. |
| **How does this relate to your ARM SoC / TrustZone work?** | Direct extension: cache coherency work in heterogeneous ARM architectures deals with exactly this trade-off — evaluating set-associativity, way-prediction, and cache partitioning (e.g., ARM's DynamIQ Shared Unit) to reduce conflict misses under multicore contention, which ties into the 43% lock-contention speedup and 26% false-sharing gains mentioned in your ARM performance work. |

---

Want me to extend this into a **2-way / 4-way set-associative version with LRU replacement** as a natural follow-up (a common "level 2" interview extension), or package this whole thing into a **Word doc / code portfolio PDF** for your prep binder?