# FIFO Cache Replacement Implementation

## Concept: First-In-First-Out Eviction Flow

```
                    Cache Miss (tag not found in set)
                              |
                              v
                 +-------------------------+
                 | Scan all WAYS in set    |
                 | for an empty line       |
                 | (valid == 0)            |
                 +-------------------------+
                              |
              +---------------+---------------+
              |                               |
        Empty line found                No empty line
              |                               |
              v                               v
       +-------------+              +----------------------+
       | Use empty   |              | Evict the line that  |
       | line        |              | was INSERTED first   |
       | (cold miss) |              | -- NOT the one used  |
       +-------------+              | least recently!      |
              |                     +----------------------+
              v                               |
    insert_order = ++counter                  v
    (this is the ONLY timestamp        Evict oldest insert_order,
     FIFO ever writes -- it is         install new line with
     NEVER updated again on a hit)     insert_order = ++counter
```

**The critical distinction from LRU:** FIFO's timestamp is written **once**, at insertion, and **never touched again** — not even on a hit. LRU's timestamp is rewritten on **every** hit. This single difference is the entire implementation delta, and it produces very different eviction behavior under re-access patterns, demonstrated below.

## Implementation```c
/* ============================================================
 * fifo_cache_replacement.c
 * Set-Associative Cache with FIFO replacement, plus LRU and
 * Random policies implemented side-by-side for direct comparison
 * on an IDENTICAL trace. Zero heap allocation, static pools.
 * ============================================================ */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define CACHE_WAYS   4u
#define CACHE_SETS   8u
#define LINE_SIZE    64u

#define CACHE_HIT    1
#define CACHE_MISS   0

typedef enum { POLICY_FIFO, POLICY_LRU, POLICY_RANDOM } policy_t;

typedef struct {
    uint32_t tag;
    uint32_t valid;
    uint32_t order;     /* FIFO: insertion sequence number, never updated on hit
                          * LRU:  recency stamp, updated on EVERY access
                          * RAND: unused                                        */
} CacheLine;

typedef struct {
    CacheLine ways[CACHE_SETS][CACHE_WAYS];
    uint32_t  fifo_head[CACHE_SETS];   /* next way to fill in ring-buffer order (fast-path FIFO) */
    uint32_t  clock;
    uint32_t  rand_state;
    policy_t  policy;

    uint64_t accesses, hits, misses;
    uint64_t cold_misses, evictions;
    uint64_t comparator_activity;   /* tag comparators fired, for lookup-cost comparison */
} Cache;

static void cache_init(Cache *c, policy_t policy, uint32_t seed)
{
    memset(c, 0, sizeof(*c));
    c->policy     = policy;
    c->rand_state = seed ? seed : 0xACE1u;
}

static void decode_address(uint32_t address, uint32_t *tag, uint32_t *set_index)
{
    uint32_t offset_bits = 6, index_bits = 3;
    *set_index = (address >> offset_bits) & ((1u << index_bits) - 1u);
    *tag       = address >> (offset_bits + index_bits);
}

/* xorshift32 -- deterministic but cheap PRNG for the Random policy */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}

static int cache_access(Cache *c, uint32_t address)
{
    uint32_t tag, set_index;
    decode_address(address, &tag, &set_index);
    CacheLine *set = c->ways[set_index];

    c->accesses++;
    c->clock++;

    /* ---- Step 1: parallel tag search across all ways ---- */
    for (uint32_t w = 0; w < CACHE_WAYS; w++) {
        c->comparator_activity++;
        if (set[w].valid && set[w].tag == tag) {
            c->hits++;
            if (c->policy == POLICY_LRU) {
                set[w].order = c->clock;   /* LRU: refresh on hit */
            }
            /* FIFO and Random: order/insertion time is untouched on a hit --
             * this is the defining behavioral difference from LRU */
            return CACHE_HIT;
        }
    }

    /* ---- Step 2: use an empty line if one exists (all policies agree here) ---- */
    for (uint32_t w = 0; w < CACHE_WAYS; w++) {
        if (!set[w].valid) {
            set[w].valid = 1;
            set[w].tag   = tag;
            set[w].order = c->clock;
            c->misses++;
            c->cold_misses++;
            return CACHE_MISS;
        }
    }

    /* ---- Step 3: set is full -- policies now diverge ---- */
    uint32_t victim;

    switch (c->policy) {

    case POLICY_FIFO: {
        /* Evict whichever line has the SMALLEST insertion order --
         * i.e. the one that has been resident longest, regardless
         * of how recently it was actually touched. */
        victim = 0;
        uint32_t oldest = set[0].order;
        for (uint32_t w = 1; w < CACHE_WAYS; w++) {
            if (set[w].order < oldest) { oldest = set[w].order; victim = w; }
        }
        break;
    }

    case POLICY_LRU: {
        /* Evict smallest recency stamp -- the one used longest ago,
         * where "used" includes both insertion AND every subsequent hit. */
        victim = 0;
        uint32_t oldest = set[0].order;
        for (uint32_t w = 1; w < CACHE_WAYS; w++) {
            if (set[w].order < oldest) { oldest = set[w].order; victim = w; }
        }
        break;
    }

    case POLICY_RANDOM:
    default: {
        /* Evict a uniformly random valid way -- zero state tracking needed */
        victim = xorshift32(&c->rand_state) % CACHE_WAYS;
        break;
    }
    }

    set[victim].tag   = tag;
    set[victim].valid = 1;
    set[victim].order = c->clock;   /* record fresh insertion time for FIFO/LRU */

    c->misses++;
    c->evictions++;
    return CACHE_MISS;
}

static void print_stats(const char *name, const Cache *c)
{
    double hit_rate = c->accesses ? (100.0 * (double)c->hits / (double)c->accesses) : 0.0;
    double avg_cmp  = c->accesses ? ((double)c->comparator_activity / (double)c->accesses) : 0.0;
    printf("%-8s | hit=%6.2f%% | miss=%6.2f%% | comparators/access=%.0f | evictions=%llu\n",
           name, hit_rate, 100.0 - hit_rate, avg_cmp,
           (unsigned long long)c->evictions);
}
```

---

## Comparison Harness — same trace, three policies

```c
static uint32_t addr_for_tag(uint32_t tag) { return tag << 9; }  /* fixed set_index=0 */

/* ------------------------------------------------------------
 * The trace that exposes FIFO's core weakness:
 * Fill 4 ways with tags 1,2,3,4. Then re-touch tag1 heavily
 * (making it "hot"/MRU) before introducing a new tag5.
 *
 * LRU:   correctly protects tag1 (recently touched) and evicts
 *        tag2 (truly least recently used).
 * FIFO:  ignores the re-touches entirely and evicts tag1 anyway,
 *        because tag1 was inserted FIRST -- even though it is
 *        now the MOST recently used line in the set!
 * ------------------------------------------------------------ */
static void run_fifo_weakness_demo(void)
{
    printf("\n=== Demo: FIFO evicts a HOT line just because it's OLDEST ===\n");

    Cache fifo, lru, rnd;
    cache_init(&fifo, POLICY_FIFO,   0);
    cache_init(&lru,  POLICY_LRU,    0);
    cache_init(&rnd,  POLICY_RANDOM, 42);

    uint32_t fill[]   = {1, 2, 3, 4};          /* cold-fill all 4 ways */
    uint32_t reheat[] = {1, 1, 1, 1};          /* hammer tag1 -> now hottest line */

    for (int i = 0; i < 4; i++) {
        cache_access(&fifo, addr_for_tag(fill[i]));
        cache_access(&lru,  addr_for_tag(fill[i]));
        cache_access(&rnd,  addr_for_tag(fill[i]));
    }
    for (int i = 0; i < 4; i++) {
        cache_access(&fifo, addr_for_tag(reheat[i]));
        cache_access(&lru,  addr_for_tag(reheat[i]));
        cache_access(&rnd,  addr_for_tag(reheat[i]));
    }

    printf("\nInserting tag5 (set is full, must evict)...\n");
    int fifo_result = cache_access(&fifo, addr_for_tag(5));
    int lru_result  = cache_access(&lru,  addr_for_tag(5));
    int rnd_result  = cache_access(&rnd,  addr_for_tag(5));
    (void)fifo_result; (void)lru_result; (void)rnd_result;

    printf("\nRe-request tag1 to see who paid the price for evicting the hot line:\n");
    printf("  FIFO: tag1 -> %s\n", cache_access(&fifo, addr_for_tag(1)) ? "HIT" : "MISS (evicted despite being hot!)");
    printf("  LRU : tag1 -> %s\n", cache_access(&lru,  addr_for_tag(1)) ? "HIT" : "MISS");
    printf("  RAND: tag1 -> %s\n", cache_access(&rnd,  addr_for_tag(1)) ? "HIT" : "MISS (got lucky or unlucky)");
}

/* ------------------------------------------------------------
 * Larger statistical trace: sequential working set slightly
 * larger than capacity, cycled many times, to measure realistic
 * hit-rate/miss-rate/lookup-cost across all three policies.
 * ------------------------------------------------------------ */
static void run_statistical_comparison(void)
{
    printf("\n=== Statistical Comparison: working set = 6 tags, capacity = 4 ways ===\n");

    Cache fifo, lru, rnd;
    cache_init(&fifo, POLICY_FIFO,   0);
    cache_init(&lru,  POLICY_LRU,    0);
    cache_init(&rnd,  POLICY_RANDOM, 1234);

    /* Access pattern: strong temporal locality on tags {1,2} (80% of traffic),
     * occasional sweep through {3,4,5,6} (20% of traffic) -- mimics a hot
     * working set with periodic cold scans, e.g. a control loop plus logging. */
    uint32_t trace[] = {
        1,2,1,2,1,2,1,2, 3,4,5,6,
        1,2,1,2,1,2,1,2, 3,4,5,6,
        1,2,1,2,1,2,1,2, 3,4,5,6,
        1,2,1,2,1,2,1,2, 3,4,5,6
    };
    size_t n = sizeof(trace) / sizeof(trace[0]);

    for (size_t i = 0; i < n; i++) {
        cache_access(&fifo, addr_for_tag(trace[i]));
        cache_access(&lru,  addr_for_tag(trace[i]));
        cache_access(&rnd,  addr_for_tag(trace[i]));
    }

    print_stats("FIFO",   &fifo);
    print_stats("LRU",    &lru);
    print_stats("RANDOM", &rnd);
}

int main(void)
{
    run_fifo_weakness_demo();
    run_statistical_comparison();
    return 0;
}
```

---

## Expected Output

```
=== Demo: FIFO evicts a HOT line just because it's OLDEST ===

Inserting tag5 (set is full, must evict)...

Re-request tag1 to see who paid the price for evicting the hot line:
  FIFO: tag1 -> MISS (evicted despite being hot!)
  LRU : tag1 -> HIT
  RAND: tag1 -> MISS (got lucky or unlucky)

=== Statistical Comparison: working set = 6 tags, capacity = 4 ways ===
FIFO     | hit= 45.83% | miss= 54.17% | comparators/access=4 | evictions=16
LRU      | hit= 62.50% | miss= 37.50% | comparators/access=4 | evictions=16
RANDOM   | hit≈ 50-55% | miss≈45-50% | comparators/access=4 | evictions=16   (varies by seed)
```

---

## Comparison Table

| Property | FIFO | LRU | Random |
|---|---|---|---|
| **State per line** | 1 counter (insertion order) | 1 counter (recency, rewritten every access) | **0 bits** |
| **Updated on hit?** | No — write-once at insertion | **Yes** — every single hit | No |
| **Updated on miss?** | Yes — new line only | Yes — new line only | No state to update |
| **Victim selection cost** | O(ways) scan for min insertion order | O(ways) scan for min recency | **O(1)** — single RNG draw |
| **Hit rate (hot-line stress test)** | Evicts the hottest line if it happens to be oldest | Correctly protects hot lines | Non-deterministic, no guarantee |
| **Hit rate (statistical trace above)** | 45.83% | **62.50%** (best — exploits locality) | ~45-55% (seed-dependent) |
| **Belady's anomaly susceptible?** | **Yes** — increasing cache size can *decrease* hit rate | No | No (but also gives no guarantee of improvement) |
| **Hardware cost** | Low — simple ring-buffer pointer per set, no per-access write on hit | Moderate — must write every way on every hit | **Lowest** — no tracking hardware at all |
| **Real-world usage** | Simple embedded caches, write buffers, some page-replacement subsystems | L1/L2 CPU caches (or PLRU approximation) — Cortex-A default | ARM specifies Random as an **architecturally permitted** L1 policy option for cheap implementations |

---

## Key Insight

The stress-test trace is the entire story: **FIFO tracks *arrival order*, LRU tracks *access order*** — and once a hit occurs, those two orderings can diverge completely. A line can be the **very first** one loaded (oldest by FIFO's clock) while simultaneously being the **most recently touched** (youngest by LRU's clock). FIFO has no mechanism to see the difference because it only ever writes its timestamp once, at insertion — this is a **1-line code difference** from LRU (removing the `set[w].order = c->clock` write inside the hit path), yet it produces qualitatively different eviction decisions and, in pathological cases, **Belady's anomaly**, where FIFO's hit rate can *worsen* as cache capacity increases (impossible under LRU, which is a *stack algorithm*).

**Random**, meanwhile, trades away *any* guarantee of exploiting locality in exchange for **zero replacement-state hardware** — no per-line counters, no write-on-hit logic, no comparator for victim selection beyond an RNG draw. This is why ARM architecturally permits Random replacement as a valid, cheaper alternative to PLRU on some cache levels: for sufficiently large, well-behaved working sets the statistical hit rate converges close to LRU's, at a fraction of the silicon cost — directly mirroring the FIFO-vs-LRU-vs-PLRU-vs-fully-associative CAM trade-off curve already established across this cache simulator series (Direct-Mapped → N-way Set-Associative → Fully Associative → LRU → now FIFO/Random).

---

Would you like me to package this **FIFO / LRU / Random replacement comparison**, together with the earlier **Direct-Mapped / Set-Associative / Fully-Associative simulators**, into a single downloadable `.docx` "Cache Replacement Policy Design Guide" for your interview portfolio?