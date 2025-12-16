# Fully Associative Cache Simulator + 3-Way Comparison

## Concept: The "any block, any line" placement policy

```
Direct-Mapped                2-Way Set-Assoc              Fully Associative
(1 way per set)              (2 ways per set)             (1 set, N ways)

Address                      Address                      Address
  |                            |                             |
  +-> index picks              +-> index picks               +-> NO index bits
  |   exactly 1 line           |   exactly 1 SET              |   at all
  |   (nowhere else            |   (2 candidate                |
  |   to go)                   |    lines to check)            +-> block can go
  |                            |                                   in ANY of the
  v                            v                                   N cache lines
compare 1 tag              compare 2 tags                          |
                                                                     v
                                                              compare ALL N tags
                                                              (parallel CAM search)
```

Fully associative removes the index field entirely — the whole address (minus offset) is tag, and hardware must search **every line simultaneously** using a Content-Addressable Memory (CAM). This eliminates conflict misses completely (any block can live anywhere) but multiplies comparator hardware by the number of lines.

---

## Implementation

```c
/* ============================================================
 * fully_associative_cache.c
 * Fully Associative Cache Simulator + comparison harness against
 * Direct-Mapped and 2-Way Set-Associative caches of EQUAL total
 * capacity. Zero heap allocation, static pools throughout.
 * ============================================================ */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define LINE_SIZE_BYTES   64u
#define CACHE_SIZE_BYTES  (32u * 1024u)
#define TOTAL_LINES       (CACHE_SIZE_BYTES / LINE_SIZE_BYTES)   /* 512 */

#define CACHE_HIT   1
#define CACHE_MISS  0

typedef struct {
    bool     valid;
    uint32_t tag;
    uint64_t last_used;
} cache_line_t;

typedef struct {
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
    uint64_t total_lookup_cost;   /* sum of tag-comparators activated per access */
} stats_t;

static void stats_print(const char *name, const stats_t *s, uint32_t assoc)
{
    double hit_rate  = s->accesses ? (100.0 * (double)s->hits  / (double)s->accesses) : 0.0;
    double miss_rate = s->accesses ? (100.0 * (double)s->misses / (double)s->accesses) : 0.0;
    double avg_cost  = s->accesses ? ((double)s->total_lookup_cost / (double)s->accesses) : 0.0;

    printf("%-24s | hit=%6.2f%% | miss=%6.2f%% | comparators/access=%3.0f | (%u-way)\n",
           name, hit_rate, miss_rate, avg_cost, assoc);
}

/* ============================================================
 * 1) DIRECT-MAPPED CACHE  (512 sets x 1 way)
 *    index_bits = 9, offset_bits = 6
 *    lookup cost = 1 comparator, always
 * ============================================================ */
typedef struct {
    cache_line_t lines[TOTAL_LINES];
    stats_t      stats;
} dm_cache_t;

static void dm_init(dm_cache_t *c) { memset(c, 0, sizeof(*c)); }

static int dm_access(dm_cache_t *c, uint32_t address)
{
    uint32_t offset_bits = 6;   /* log2(64)  */
    uint32_t index_bits  = 9;   /* log2(512) */
    uint32_t index = (address >> offset_bits) & ((1u << index_bits) - 1u);
    uint32_t tag   = address >> (offset_bits + index_bits);

    c->stats.accesses++;
    c->stats.total_lookup_cost += 1;   /* exactly 1 tag comparator exists */

    cache_line_t *line = &c->lines[index];
    if (line->valid && line->tag == tag) {
        c->stats.hits++;
        return CACHE_HIT;
    }

    line->valid = true;   /* only 1 way -> no LRU decision, always overwrite */
    line->tag   = tag;
    c->stats.misses++;
    return CACHE_MISS;
}

/* ============================================================
 * 2) 2-WAY SET-ASSOCIATIVE CACHE  (256 sets x 2 ways)
 *    index_bits = 8, offset_bits = 6
 *    lookup cost = 2 comparators, always (both ways checked in parallel)
 * ============================================================ */
#define TW_SETS 256u
#define TW_WAYS 2u

typedef struct {
    cache_line_t ways[TW_SETS][TW_WAYS];
    uint64_t     clock;
    stats_t      stats;
} tw_cache_t;

static void tw_init(tw_cache_t *c) { memset(c, 0, sizeof(*c)); }

static int tw_access(tw_cache_t *c, uint32_t address)
{
    uint32_t offset_bits = 6;
    uint32_t index_bits  = 8;
    uint32_t index = (address >> offset_bits) & ((1u << index_bits) - 1u);
    uint32_t tag   = address >> (offset_bits + index_bits);

    c->stats.accesses++;
    c->clock++;
    c->stats.total_lookup_cost += TW_WAYS;   /* 2 comparators fire every access */

    cache_line_t *set = c->ways[index];

    for (uint32_t w = 0; w < TW_WAYS; w++) {
        if (set[w].valid && set[w].tag == tag) {
            set[w].last_used = c->clock;
            c->stats.hits++;
            return CACHE_HIT;
        }
    }

    /* miss -> pick empty way, else true LRU between the 2 ways */
    uint32_t victim = (!set[0].valid) ? 0 : (!set[1].valid) ? 1 :
                       (set[0].last_used < set[1].last_used) ? 0 : 1;

    set[victim].valid     = true;
    set[victim].tag       = tag;
    set[victim].last_used = c->clock;
    c->stats.misses++;
    return CACHE_MISS;
}

/* ============================================================
 * 3) FULLY ASSOCIATIVE CACHE  (1 set x 512 ways)
 *    NO index bits at all -> entire remaining address is tag
 *    lookup cost = 512 comparators, always (true CAM parallel search)
 *
 *    Any incoming block may be placed in ANY of the 512 lines,
 *    so a conflict miss is structurally impossible until the
 *    cache is genuinely full (capacity miss only).
 * ============================================================ */
typedef struct {
    cache_line_t lines[TOTAL_LINES];
    uint64_t     clock;
    stats_t      stats;
} fa_cache_t;

static void fa_init(fa_cache_t *c) { memset(c, 0, sizeof(*c)); }

static int fa_access(fa_cache_t *c, uint32_t address)
{
    uint32_t offset_bits = 6;                 /* log2(64)  */
    uint32_t tag = address >> offset_bits;    /* NO index field consumed */

    c->stats.accesses++;
    c->clock++;
    c->stats.total_lookup_cost += TOTAL_LINES;  /* 512 comparators fire every access */

    /* Step: search ALL 512 lines in parallel (modeled here as a loop,
     * but in real hardware this is a single-cycle CAM lookup) */
    for (uint32_t i = 0; i < TOTAL_LINES; i++) {
        if (c->lines[i].valid && c->lines[i].tag == tag) {
            c->lines[i].last_used = c->clock;
            c->stats.hits++;
            return CACHE_HIT;
        }
    }

    /* miss -> find empty line first, else global LRU victim across ALL 512 */
    uint32_t victim = 0;
    bool found_empty = false;

    for (uint32_t i = 0; i < TOTAL_LINES; i++) {
        if (!c->lines[i].valid) {
            victim = i;
            found_empty = true;
            break;
        }
    }

    if (!found_empty) {
        uint64_t oldest = c->lines[0].last_used;
        for (uint32_t i = 1; i < TOTAL_LINES; i++) {
            if (c->lines[i].last_used < oldest) {
                oldest = c->lines[i].last_used;
                victim = i;
            }
        }
    }

    c->lines[victim].valid     = true;
    c->lines[victim].tag       = tag;
    c->lines[victim].last_used = c->clock;
    c->stats.misses++;
    return CACHE_MISS;
}
```

---

## Comparison Harness — identical trace, three caches

```c
/* ============================================================
 * Demo trace: 5 addresses that all collide on the SAME set
 * index in the direct-mapped and 2-way caches, cycled 6x.
 *
 * This isolates conflict-miss behavior:
 *   - Direct-mapped capacity for this set = 1  -> always evicts
 *   - 2-way capacity for this set          = 2  -> still < 5, thrashes
 *   - Fully-associative capacity           = 512 -> trivially fits all 5
 * ============================================================ */
static void run_conflict_demo(void)
{
    dm_cache_t dm; dm_init(&dm);
    tw_cache_t tw; tw_init(&tw);
    fa_cache_t fa; fa_init(&fa);

    uint32_t base[5];
    for (int i = 0; i < 5; i++) {
        base[i] = (uint32_t)i * CACHE_SIZE_BYTES;   /* same index, distinct tag */
    }

    printf("\n=== Conflict-Miss Stress Test (5 colliding blocks, 6 rounds) ===\n");

    for (int round = 0; round < 6; round++) {
        for (int i = 0; i < 5; i++) {
            dm_access(&dm, base[i]);
            tw_access(&tw, base[i]);
            fa_access(&fa, base[i]);
        }
    }

    stats_print("Direct-Mapped",        &dm.stats, 1);
    stats_print("2-Way Set-Associative", &tw.stats, 2);
    stats_print("Fully Associative",     &fa.stats, TOTAL_LINES);
}

/* ============================================================
 * Second trace: normal locality pattern (small working set,
 * well within total cache capacity, no artificial collisions).
 * Shows associativity gives near-zero benefit when there's no
 * conflict pressure -- the working set just fits everywhere.
 * ============================================================ */
static void run_locality_demo(void)
{
    dm_cache_t dm; dm_init(&dm);
    tw_cache_t tw; tw_init(&tw);
    fa_cache_t fa; fa_init(&fa);

    printf("\n=== Normal Locality Test (sequential 64B-strided array, wraps 4x) ===\n");

    uint32_t working_set_bytes = 8 * 1024;   /* well under 32KB cache */
    for (int pass = 0; pass < 4; pass++) {
        for (uint32_t addr = 0; addr < working_set_bytes; addr += LINE_SIZE_BYTES) {
            dm_access(&dm, addr);
            tw_access(&tw, addr);
            fa_access(&fa, addr);
        }
    }

    stats_print("Direct-Mapped",        &dm.stats, 1);
    stats_print("2-Way Set-Associative", &tw.stats, 2);
    stats_print("Fully Associative",     &fa.stats, TOTAL_LINES);
}

int main(void)
{
    run_conflict_demo();
    run_locality_demo();
    return 0;
}
```

---

## Expected Output

```
=== Conflict-Miss Stress Test (5 colliding blocks, 6 rounds) ===
Direct-Mapped            | hit=  0.00% | miss=100.00% | comparators/access=  1 | (1-way)
2-Way Set-Associative    | hit=  0.00% | miss=100.00% | comparators/access=  2 | (2-way)
Fully Associative        | hit= 83.33% | miss= 16.67% | comparators/access=512 | (512-way)

=== Normal Locality Test (sequential 64B-strided array, wraps 4x) ===
Direct-Mapped            | hit= 75.00% | miss= 25.00% | comparators/access=  1 | (1-way)
2-Way Set-Associative    | hit= 75.00% | miss= 25.00% | comparators/access=  2 | (2-way)
Fully Associative        | hit= 75.00% | miss= 25.00% | comparators/access=512 | (512-way)
```

---

## Measured Results — Summary Table

| Metric | Direct-Mapped | 2-Way Set-Assoc | Fully Associative |
|---|---|---|---|
| **Conflict-heavy trace: hit rate** | 0% | 0% | **83.33%** |
| **Conflict-heavy trace: miss rate** | 100% | 100% | **16.67%** |
| **Normal locality trace: hit rate** | 75% | 75% | 75% (identical — no conflicts to solve) |
| **Lookup cost (comparators/access)** | **1** | **2** | **512** |
| **Victim selection cost on miss** | O(1), no choice | O(1), 2-way LRU compare | O(N) scan across all lines (or CAM priority encoder in HW) |
| **Hardware structure** | SRAM + 1 comparator | SRAM + 2 comparators | Full CAM (tag stored + compared in same cell) |
| **Silicon cost per bit** | Lowest | Low | **Highest** (CAM cells are ~2-3x larger than SRAM per bit) |
| **Where used in real SoCs** | L1 I-cache sometimes, simple embedded caches | L1 D-cache (Cortex-A), typical sweet spot | Small structures only: TLBs, victim caches, BTBs (16-64 entries) |

---

## Key Insight

The two traces prove the core trade-off precisely:

1. **When conflicts exist** (working set concentrated on a few colliding indices) → fully associative wins decisively (0% → 83%), because placement flexibility is the entire fix.
2. **When conflicts don't exist** (working set naturally spread across sets, well within capacity) → all three caches perform **identically**, because associativity only matters when multiple blocks fight over the same set. Paying **512x the comparator hardware** here buys **zero** extra hit rate.

This is exactly why fully-associative caches are **never used for large L1/L2/L3 caches** in production silicon — the O(N) comparator cost doesn't scale — but **are** used for small, latency-critical, highly conflict-prone structures like TLBs (typically 32–64 fully-associative entries) and victim caches (4–16 entries), where N is small enough that the CAM cost is negligible but the conflict-elimination benefit is large. This directly parallels the 4-tier hybrid TLB/THP strategy trade-off referenced in the prior tree/Fenwick work — associativity degree is tuned per structure based on its expected conflict pressure, not applied uniformly.

---

Want me to extend this into a **⭐⭐⭐⭐⭐ N-way parametrized sweep** (run 1-way through 512-way on the same trace and plot hit-rate vs. lookup-cost as a single table, to show the full diminishing-returns curve), or package this three-cache comparison into a downloadable `.docx`/`.pptx` for your interview portfolio?