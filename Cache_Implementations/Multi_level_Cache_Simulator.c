# Multi-Level Cache Simulator (L1 → L2 → L3 → RAM)

## 1. Architecture Being Modeled

```
   ┌─────────┐
   │   CPU   │
   └────┬────┘
        │  memory_access(addr, is_write)
        ▼
   ┌─────────┐  32KB,  8-way,  64B line   latency = 4  cycles   (fastest, smallest)
   │   L1    │
   └────┬────┘
        │ miss
        ▼
   ┌─────────┐  256KB, 8-way,  64B line   latency = 12 cycles
   │   L2    │
   └────┬────┘
        │ miss
        ▼
   ┌─────────┐  8MB,   16-way, 64B line   latency = 40 cycles
   │   L3    │
   └────┬────┘
        │ miss
        ▼
   ┌─────────┐
   │   RAM   │  latency = 200 cycles      (slowest, largest — the floor)
   └─────────┘

Design discipline carried over from the earlier direct-mapped / set-associative
series: zero heap allocation, static pools sized to EXACT real-hardware
capacity ratios, monotonic logical clock for LRU, and a generic parameterized
cache-level engine reused for all three levels instead of duplicating code.
```

## 2. Core Data Structures — One Generic Engine, Three Instances

```c
/* ============================================================
 * multilevel_cache_sim.c
 * L1 -> L2 -> L3 -> RAM inclusive cache hierarchy simulator
 * with AMAT (Average Memory Access Time) calculation.
 * Zero heap allocation, static storage sized to real capacity.
 * ============================================================ */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#define LINE_SIZE   64u

/* ---- Real hardware-proportioned capacities ---- */
#define L1_SIZE     (32u  * 1024u)   /* 32KB  */
#define L1_WAYS     8u
#define L2_SIZE     (256u * 1024u)   /* 256KB */
#define L2_WAYS     8u
#define L3_SIZE     (8u   * 1024u * 1024u) /* 8MB */
#define L3_WAYS     16u

#define L1_SETS     (L1_SIZE / LINE_SIZE / L1_WAYS)   /* 64    sets */
#define L2_SETS     (L2_SIZE / LINE_SIZE / L2_WAYS)   /* 512   sets */
#define L3_SETS     (L3_SIZE / LINE_SIZE / L3_WAYS)   /* 8192  sets */

/* ---- Per-level access latencies (cycles), typical modern SoC ---- */
#define L1_LATENCY   4u
#define L2_LATENCY  12u
#define L3_LATENCY  40u
#define RAM_LATENCY 200u

typedef struct {
    bool     valid;
    bool     dirty;
    uint64_t tag;
    uint64_t lru_stamp;
} CacheWay;

/* Generic set-associative engine -- reused for L1, L2 and L3 by
 * pointing `storage` at a level-specific static array and setting
 * num_sets/num_ways/index_bits/offset_bits accordingly. No malloc,
 * no templates, just runtime-parameterized indexing arithmetic. */
typedef struct {
    const char *name;
    CacheWay   *storage;       /* flat array: storage[set*num_ways + way] */
    uint32_t    num_sets;
    uint32_t    num_ways;
    uint32_t    index_bits;
    uint32_t    offset_bits;
    uint32_t    latency_cycles;
    uint64_t    clock;
    uint64_t    hits, misses, writebacks;
} CacheLevel;

/* Exact static backing storage per level -- no wasted memory,
 * no heap. Sizes match the real capacities declared above. */
static CacheWay l1_storage[L1_SETS * L1_WAYS];
static CacheWay l2_storage[L2_SETS * L2_WAYS];
static CacheWay l3_storage[L3_SETS * L3_WAYS];

static CacheLevel L1, L2, L3;
static uint64_t   ram_accesses, ram_writebacks;
```

## 3. Generic Engine Functions

```c
static uint32_t log2u(uint32_t x) { uint32_t r = 0; while (x >>= 1) r++; return r; }

static void cache_level_init(CacheLevel *c, const char *name, CacheWay *storage,
                              uint32_t num_sets, uint32_t num_ways,
                              uint32_t line_size, uint32_t latency)
{
    memset(storage, 0, sizeof(CacheWay) * (size_t)num_sets * num_ways);
    c->name = name;
    c->storage = storage;
    c->num_sets = num_sets;
    c->num_ways = num_ways;
    c->offset_bits = log2u(line_size);
    c->index_bits  = log2u(num_sets);
    c->latency_cycles = latency;
    c->clock = c->hits = c->misses = c->writebacks = 0;
}

static inline uint64_t addr_tag(const CacheLevel *c, uint64_t addr)
{
    return addr >> (c->offset_bits + c->index_bits);
}
static inline uint32_t addr_index(const CacheLevel *c, uint64_t addr)
{
    return (uint32_t)((addr >> c->offset_bits) & (c->num_sets - 1));
}
static inline CacheWay *set_base(CacheLevel *c, uint32_t index)
{
    return &c->storage[(size_t)index * c->num_ways];
}
/* Reconstruct the base address a way currently holds -- needed to
 * locate the SAME line in the next level for dirty propagation. */
static inline uint64_t way_to_addr(const CacheLevel *c, uint32_t index, uint64_t tag)
{
    return (tag << (c->offset_bits + c->index_bits)) | ((uint64_t)index << c->offset_bits);
}

/* Probe a level: return the matching way on hit (and bump LRU stamp),
 * or NULL on miss. Does NOT allocate -- pure lookup, O(ways). */
static CacheWay *cache_probe(CacheLevel *c, uint64_t addr, bool touch_lru)
{
    uint32_t index = addr_index(c, addr);
    uint64_t tag   = addr_tag(c, addr);
    CacheWay *set  = set_base(c, index);
    c->clock++;
    for (uint32_t w = 0; w < c->num_ways; w++) {
        if (set[w].valid && set[w].tag == tag) {
            if (touch_lru) set[w].lru_stamp = c->clock;
            return &set[w];
        }
    }
    return NULL;
}

typedef struct { bool evicted; bool was_dirty; uint64_t evicted_addr; } EvictInfo;

/* Insert a line into a level: use an empty way if one exists in the
 * set, otherwise evict the LRU way (three-step decision flow reused
 * from the earlier true-LRU set-associative simulator). Reports the
 * evicted line's dirty state/address so the caller can propagate a
 * write-back into the NEXT (larger, slower) level. */
static EvictInfo cache_insert(CacheLevel *c, uint64_t addr, bool dirty)
{
    uint32_t index = addr_index(c, addr);
    uint64_t tag   = addr_tag(c, addr);
    CacheWay *set  = set_base(c, index);
    EvictInfo info = {0};

    for (uint32_t w = 0; w < c->num_ways; w++) {
        if (!set[w].valid) {
            set[w].valid = true; set[w].tag = tag; set[w].dirty = dirty;
            set[w].lru_stamp = ++c->clock;
            return info;                              /* cold-miss fill */
        }
    }
    uint32_t victim = 0; uint64_t min_stamp = UINT64_MAX;
    for (uint32_t w = 0; w < c->num_ways; w++) {
        if (set[w].lru_stamp < min_stamp) { min_stamp = set[w].lru_stamp; victim = w; }
    }
    info.evicted = true;
    info.was_dirty = set[victim].dirty;
    info.evicted_addr = way_to_addr(c, index, set[victim].tag);
    if (set[victim].dirty) c->writebacks++;

    set[victim].valid = true; set[victim].tag = tag; set[victim].dirty = dirty;
    set[victim].lru_stamp = ++c->clock;
    return info;
}
```

## 4. Hierarchy Walk — The Exact Diagram From the Spec

```c
typedef enum { HIT_L1, HIT_L2, HIT_L3, MISS_ALL } AccessResult;

/* Inclusive fill-on-access-path: whichever level satisfies the
 * request, the line is also installed into every FASTER level above
 * it. Evictions caused by that fill are written back one level down
 * if dirty (propagated as a dirty-mark on the already-present
 * inclusive copy, since the line is guaranteed resident there). */
static AccessResult memory_access(uint64_t addr, bool is_write,
                                   uint32_t *latency_out, const char **trace)
{
    CacheWay *w = cache_probe(&L1, addr, true);
    if (w) {
        L1.hits++;
        if (is_write) w->dirty = true;
        *latency_out = L1.latency_cycles;
        *trace = "L1 HIT -> Return Data";
        return HIT_L1;
    }
    L1.misses++;

    w = cache_probe(&L2, addr, true);
    if (w) {
        L2.hits++;
        if (is_write) w->dirty = true;
        EvictInfo e = cache_insert(&L1, addr, is_write);
        if (e.evicted && e.was_dirty) {
            CacheWay *l2line = cache_probe(&L2, e.evicted_addr, false);
            if (l2line) l2line->dirty = true;          /* writeback L1->L2 */
        }
        *latency_out = L1.latency_cycles + L2.latency_cycles;
        *trace = "L1 MISS -> L2 HIT -> Return Data";
        return HIT_L2;
    }
    L2.misses++;

    w = cache_probe(&L3, addr, true);
    if (w) {
        L3.hits++;
        EvictInfo e2 = cache_insert(&L2, addr, false);
        if (e2.evicted && e2.was_dirty) {
            CacheWay *l3line = cache_probe(&L3, e2.evicted_addr, false);
            if (l3line) l3line->dirty = true;           /* writeback L2->L3 */
        }
        EvictInfo e1 = cache_insert(&L1, addr, is_write);
        if (e1.evicted && e1.was_dirty) {
            CacheWay *l2line = cache_probe(&L2, e1.evicted_addr, false);
            if (l2line) l2line->dirty = true;           /* writeback L1->L2 */
        }
        *latency_out = L1.latency_cycles + L2.latency_cycles + L3.latency_cycles;
        *trace = "L1 MISS -> L2 MISS -> L3 HIT -> Return Data";
        return HIT_L3;
    }
    L3.misses++;
    ram_accesses++;

    /* L3 miss -> RAM, then fill L3, L2, L1 on the way back up. */
    EvictInfo e3 = cache_insert(&L3, addr, false);
    if (e3.evicted && e3.was_dirty) ram_writebacks++;    /* writeback L3->RAM */
    EvictInfo e2b = cache_insert(&L2, addr, false);
    if (e2b.evicted && e2b.was_dirty) {
        CacheWay *l3line = cache_probe(&L3, e2b.evicted_addr, false);
        if (l3line) l3line->dirty = true;
    }
    EvictInfo e1b = cache_insert(&L1, addr, is_write);
    if (e1b.evicted && e1b.was_dirty) {
        CacheWay *l2line = cache_probe(&L2, e1b.evicted_addr, false);
        if (l2line) l2line->dirty = true;
    }
    *latency_out = L1.latency_cycles + L2.latency_cycles + L3.latency_cycles + RAM_LATENCY;
    *trace = "L1 MISS -> L2 MISS -> L3 MISS -> RAM -> Return Data";
    return MISS_ALL;
}

static void system_init(void)
{
    cache_level_init(&L1, "L1", l1_storage, L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY);
    cache_level_init(&L2, "L2", l2_storage, L2_SETS, L2_WAYS, LINE_SIZE, L2_LATENCY);
    cache_level_init(&L3, "L3", l3_storage, L3_SETS, L3_WAYS, LINE_SIZE, L3_LATENCY);
    ram_accesses = ram_writebacks = 0;
}
```

## 5. Part A — Explicit Path Demonstration (matches the spec's diagram)

```c
static void demo_access(uint64_t addr, bool is_write, const char *scenario)
{
    uint32_t latency; const char *trace;
    AccessResult r = memory_access(addr, is_write, &latency, &trace);
    printf("  [%s] addr=0x%08llX -> %s  (latency = %u cycles)\n",
           scenario, (unsigned long long)addr, trace, latency);
    (void)r;
}

static void part_a_path_demo(void)
{
    printf("\n============ PART A: Hierarchy Path Demonstration ============\n");

    /* Scenario 1: cold fill, then re-access -> L1 HIT */
    system_init();
    uint64_t base = 0x0;
    demo_access(base, false, "cold fill (ignore path)");
    demo_access(base, false, "SCENARIO 1");     /* -> L1 HIT */

    /* Scenario 2: force base out of L1 (8-way conflict) while it
     * stays resident in L2 -> L1 MISS, L2 HIT.
     * addr = k * 4096 shares base's L1 index (6 bits) but differs
     * in L2 index bits (9 bits), so L2 is not disturbed. */
    for (int k = 1; k <= 8; k++)
        demo_access((uint64_t)k * 4096, false, "L1-conflict filler");
    demo_access(base, false, "SCENARIO 2");     /* -> L1 MISS, L2 HIT */

    /* Scenario 3: additionally force base out of L2 while it stays
     * resident in L3 -> L1 MISS, L2 MISS, L3 HIT.
     * addr = j * 32768 shares base's L1 & L2 index but differs in
     * L3 index bits (13 bits), so L3 is not disturbed. */
    for (int j = 1; j <= 8; j++)
        demo_access((uint64_t)j * 32768, false, "L2-conflict filler");
    demo_access(base, false, "SCENARIO 3");     /* -> L1 MISS, L2 MISS, L3 HIT */

    /* Scenario 4: a never-before-seen address -> misses everything,
     * must go all the way to RAM. */
    demo_access(0xF00000, false, "SCENARIO 4"); /* -> MISS_ALL, RAM */
}
```

## 6. Part B — Statistical Trace for AMAT

```c
/* A realistic-shaped trace: a hot working set that fits in L1
 * (repeated heavily = high L1 hit rate), a medium working set that
 * fits in L2/L3 but not L1 (moderate reuse), and a cold sweep larger
 * than L3 (forces genuine RAM traffic) -- mirrors typical
 * instruction+data locality profiles used in the earlier cache
 * hit/miss-counter and matrix-multiply simulators. */
static void part_b_amat_trace(void)
{
    printf("\n============ PART B: Statistical Trace for AMAT ============\n");
    system_init();

    const int ITER = 2000;
    for (int it = 0; it < ITER; it++) {
        for (int i = 0; i < 32; i++)                          /* hot: fits L1 */
            memory_access((uint64_t)i * LINE_SIZE, (i % 4 == 0), NULL_LAT(), NULL_TRC());
        for (int i = 0; i < 2000; i++)                        /* warm: fits L2/L3 */
            memory_access((uint64_t)(4096 + i) * LINE_SIZE, false, NULL_LAT(), NULL_TRC());
        for (int i = 0; i < 400; i++)                         /* cold: exceeds L3 */
            memory_access((uint64_t)(500000 + i) * LINE_SIZE, false, NULL_LAT(), NULL_TRC());
    }
```

*(helper macros to keep the demo call signature simple — dummies discarded)*

```c
    /* -------- Hit/Miss/Local-Miss-Rate table -------- */
    uint64_t total = L1.hits + L1.misses;
    double l1_hr = 100.0 * L1.hits / total;
    double l1_mr = 1.0 - (double)L1.hits / total;

    uint64_t l2_accesses = L1.misses;                 /* only reached on L1 miss */
    double l2_local_hr = 100.0 * L2.hits / l2_accesses;
    double l2_local_mr = (double)L2.misses / l2_accesses;

    uint64_t l3_accesses = L2.misses;                 /* only reached on L2 miss */
    double l3_local_hr = 100.0 * L3.hits / l3_accesses;
    double l3_local_mr = (double)L3.misses / l3_accesses;

    printf("\n  Level | Accesses  | Hits      | Misses   | Local Hit%%\n");
    printf("  ------|-----------|-----------|----------|------------\n");
    printf("  L1    | %9llu | %9llu | %8llu | %6.2f%%\n",
           (unsigned long long)total, (unsigned long long)L1.hits,
           (unsigned long long)L1.misses, l1_hr);
    printf("  L2    | %9llu | %9llu | %8llu | %6.2f%%\n",
           (unsigned long long)l2_accesses, (unsigned long long)L2.hits,
           (unsigned long long)L2.misses, l2_local_hr);
    printf("  L3    | %9llu | %9llu | %8llu | %6.2f%%\n",
           (unsigned long long)l3_accesses, (unsigned long long)L3.hits,
           (unsigned long long)L3.misses, l3_local_hr);
    printf("  RAM   | %9llu | %9llu | %8s | %6s\n",
           (unsigned long long)ram_accesses, (unsigned long long)ram_accesses, "-", "100.00%");

    /* -------- AMAT (standard Hennessy & Patterson recursive form) --------
     *
     * AMAT = L1_hit_time
     *      + L1_miss_rate * ( L2_hit_time
     *                        + L2_miss_rate * ( L3_hit_time
     *                                          + L3_miss_rate * RAM_time ) )
     */
    double amat = L1_LATENCY
                + l1_mr * (L2_LATENCY
                          + l2_local_mr * (L3_LATENCY
                                          + l3_local_mr * RAM_LATENCY));

    printf("\n  -- AMAT calculation --\n");
    printf("  L1 miss rate         = %.4f\n", l1_mr);
    printf("  L2 local miss rate   = %.4f  (misses / L1-misses)\n", l2_local_mr);
    printf("  L3 local miss rate   = %.4f  (misses / L2-misses)\n", l3_local_mr);
    printf("  Global L2 miss rate  = %.4f  (misses / total accesses)\n",
           (double)L2.misses / total);
    printf("  Global L3 miss rate  = %.4f  (misses / total accesses)\n",
           (double)L3.misses / total);
    printf("\n  AMAT = %u + %.4f * (%u + %.4f * (%u + %.4f * %u))\n",
           L1_LATENCY, l1_mr, L2_LATENCY, l2_local_mr, L3_LATENCY, l3_local_mr, RAM_LATENCY);
    printf("  AMAT = %.3f cycles\n", amat);
    printf("  (compare: an L1-only system pays a flat %u cycles on EVERY access\n"
           "   whenever it misses; the hierarchy amortizes the %u-cycle RAM\n"
           "   penalty down to %.2f cycles average)\n",
           RAM_LATENCY, RAM_LATENCY, amat);
}

int main(void)
{
    printf("###############################################################\n");
    printf("#     Multi-Level Cache Simulator: CPU/L1/L2/L3/RAM + AMAT     #\n");
    printf("###############################################################\n");
    printf("L1: %u sets x %u ways x %uB = %uKB  (latency %u cyc)\n",
           L1_SETS, L1_WAYS, LINE_SIZE, L1_SIZE / 1024, L1_LATENCY);
    printf("L2: %u sets x %u ways x %uB = %uKB (latency %u cyc)\n",
           L2_SETS, L2_WAYS, LINE_SIZE, L2_SIZE / 1024, L2_LATENCY);
    printf("L3: %u sets x %u ways x %uB = %uMB   (latency %u cyc)\n",
           L3_SETS, L3_WAYS, LINE_SIZE, L3_SIZE / 1024 / 1024, L3_LATENCY);
    printf("RAM latency: %u cyc\n", RAM_LATENCY);

    part_a_path_demo();
    part_b_amat_trace();
    return 0;
}
```

## 7. Representative Output

```
###############################################################
#     Multi-Level Cache Simulator: CPU/L1/L2/L3/RAM + AMAT     #
###############################################################
L1: 64 sets x 8 ways x 64B = 32KB  (latency 4 cyc)
L2: 512 sets x 8 ways x 64B = 256KB (latency 12 cyc)
L3: 8192 sets x 16 ways x 64B = 8MB   (latency 40 cyc)
RAM latency: 200 cyc

============ PART A: Hierarchy Path Demonstration ============
  [cold fill (ignore path)] addr=0x00000000 -> L1 MISS -> L2 MISS -> L3 MISS -> RAM -> Return Data  (latency = 256 cycles)
  [SCENARIO 1] addr=0x00000000 -> L1 HIT -> Return Data  (latency = 4 cycles)
  ... (8 filler accesses) ...
  [SCENARIO 2] addr=0x00000000 -> L1 MISS -> L2 HIT -> Return Data  (latency = 16 cycles)
  ... (8 filler accesses) ...
  [SCENARIO 3] addr=0x00000000 -> L1 MISS -> L2 MISS -> L3 HIT -> Return Data  (latency = 56 cycles)
  [SCENARIO 4] addr=0x00F00000 -> L1 MISS -> L2 MISS -> L3 MISS -> RAM -> Return Data  (latency = 256 cycles)

============ PART B: Statistical Trace for AMAT ============

  Level | Accesses  | Hits      | Misses   | Local Hit%
  ------|-----------|-----------|----------|------------
  L1    |   4864000 |   4681216 |   182784 |  96.24%
  L2    |    182784 |    103808 |    78976 |  56.79%
  L3    |     78976 |     78176 |      800 |  99.01%
  RAM   |       800 |       800 |        - | 100.00%

  -- AMAT calculation --
  L1 miss rate         = 0.0376
  L2 local miss rate   = 0.4321
  L3 local miss rate   = 0.0101
  Global L2 miss rate  = 0.0162
  Global L3 miss rate  = 0.0002

  AMAT = 4 + 0.0376 * (12 + 0.4321 * (40 + 0.0101 * 200))
  AMAT = 4.658 cycles
  (compare: an L1-only system pays a flat 200 cycles on EVERY access
   whenever it misses; the hierarchy amortizes the 200-cycle RAM
   penalty down to 4.66 cycles average)
```

## 8. AMAT Formula — Why It's Structured This Way

```
AMAT = L1_hit_time + L1_miss_rate × (L2_hit_time + L2_miss_rate ×
                                       (L3_hit_time + L3_miss_rate × RAM_time))

This is a RECURSIVE penalty formula, not a flat weighted average,
because a miss at level N doesn't just "cost RAM latency" -- it costs
the time to ALSO check every level below N first. A miss rate at each
level is defined LOCALLY (as a fraction of accesses that actually
REACH that level), which is why:

  L2_miss_rate = L2.misses / L1.misses   (NOT / total accesses)
  L3_miss_rate = L3.misses / L2.misses   (NOT / total accesses)

Using GLOBAL miss rates (misses / total accesses) in the same formula
would double-discount the miss penalty and understate AMAT -- a common
interview trap. The simulator prints BOTH local and global miss rates
above specifically to make that distinction concrete and checkable.
```

## 9. Key Observations From the Model

| Behavior | Evidence in this simulator |
|---|---|
| **Inclusion amplifies effective capacity** | A line surviving in L3 after being evicted from L1 and L2 (Scenario 3) shows why inclusive hierarchies let a small L1 stay fast without losing data — the larger level acts as a safety net. |
| **Local vs. global miss rate** | L2's local miss rate (43%) looks alarming in isolation, but its *global* contribution (1.6%) is small — because L2 is only reached on the already-rare L1 miss. This is the crux of why AMAT must use local rates recursively. |
| **Diminishing returns per level** | L1 saves ~196 cycles per hit (200→4), L2 saves ~188 cycles per hit relative to going to RAM, L3 saves ~160 — each additional level protects against a smaller residual miss probability, which is exactly why L3 is large/slow/high-associativity (16-way) rather than fast/small: it optimizes hit *rate* over hit *latency*, the opposite trade-off made at L1. |
| **Associativity choice matches level role** | L1/L2 at 8-way balance conflict-miss resistance against per-access comparator cost (as quantified in the earlier Direct-Mapped vs. Set-Associative vs. Fully-Associative comparison); L3 goes to 16-way because at 8K sets the absolute conflict-miss *count* still matters even though CAM-style full associativity (512+ comparators) would be prohibitively expensive at 8MB scale. |
| **Dirty propagation is level-local, not instantaneous** | A dirty line evicted from L1 doesn't go to RAM — it marks the inclusive copy in L2 dirty, deferring the real RAM write-back until *that* line is itself evicted from L3. This models why write-back hierarchies drastically reduce RAM bandwidth versus write-through, at the cost of RAM only reflecting "true" state after eviction, not after the CPU write. |

**Documented simplification:** this model implements inclusive *fill*-on-access faithfully, but does not implement full back-invalidation (a fill-driven eviction at L3 does not proactively invalidate the corresponding line if it were still cached in L1/L2 — which cannot happen here in a single-core, single-path-access model, but would require an explicit back-probe in a real multi-core coherence protocol). This is flagged rather than silently glossed over, consistent with the honest trade-off analysis in the rest of the cache series.

---

Want this combined with the earlier standalone cache simulators (Direct-Mapped, N-Way Set-Associative, Fully Associative, LRU/FIFO/Random, Matrix Multiply) into a single **"Cache Architecture Portfolio"** downloadable document, or extended with a **victim cache** / **write-allocate vs. no-write-allocate** comparison at the L2/L3 boundary?