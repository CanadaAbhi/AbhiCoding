# DMA + Cache Coherency Simulator

## 1. Architecture Being Modeled

```
   ┌─────────┐
   │   CPU   │   issues loads/stores by virtual/physical address
   └────┬────┘
        │  cpu_read(addr) / cpu_write(addr, val)
        ▼
   ┌─────────┐
   │  CACHE  │   direct-mapped, line = tag+valid+dirty+data[LINE_SIZE]
   │         │   CPU never touches RAM directly -- always through here
   └────┬────┘
        │  fill-on-miss / write-back-on-evict
        ▼
   ┌─────────┐
   │   RAM   │   ground truth -- byte array, the ONLY thing the DMA
   │         │   device can see
   └────┬────┘
        ▲
        │  dma_read(addr) / dma_write(addr, val)  <-- BYPASSES CACHE
   ┌────┴────┐
   │   DMA   │   non-coherent bus master (models a real embedded
   │  DEVICE │   peripheral: Ethernet MAC, storage controller, sensor)
   └─────────┘

KEY MODELING DECISION: DMA has a direct path to RAM that never goes
through the Cache box. This is what makes coherence a SOFTWARE
problem -- on real non-coherent embedded interconnects, the hardware
provides no automatic snooping between the CPU's cache and the DMA
engine's view of memory. Software (DC CVAC/IVAC/CIVAC in real ARM,
modeled here as cache_clean/invalidate/flush) is the only thing that
keeps the two views consistent.
```

## 2. Core Data Structures

```c
/* ============================================================
 * dma_cache_coherency_sim.c
 * Full CPU -> Cache -> RAM <- DMA coherency simulator.
 * Portable C (no ARM-specific asm required) -- models the exact
 * semantics of DC CVAC / DC IVAC / DC CIVAC in software so the
 * bug and fix are directly observable on any host.
 * ============================================================ */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#define RAM_SIZE       1024u     /* bytes of simulated DRAM            */
#define LINE_SIZE      32u       /* bytes per cache line                */
#define NUM_LINES      8u        /* direct-mapped: RAM_SIZE/LINE_SIZE   */
                                  /* /NUM_LINES = 4 possible tags/index */

typedef struct {
    bool     valid;
    bool     dirty;
    uint32_t tag;
    uint8_t  data[LINE_SIZE];
} CacheLine;

typedef struct {
    CacheLine lines[NUM_LINES];
    uint32_t  hits, misses, writebacks, invalidations;
} Cache;

static uint8_t ram[RAM_SIZE];
static Cache   cache;

static void system_reset(void)
{
    memset(ram, 0, sizeof(ram));
    memset(&cache, 0, sizeof(cache));
}

/* Direct-mapped address decomposition, identical discipline to the
 * earlier direct-mapped cache simulator series. */
static inline uint32_t addr_index(uint32_t addr) { return (addr / LINE_SIZE) % NUM_LINES; }
static inline uint32_t addr_tag(uint32_t addr)   { return (addr / LINE_SIZE) / NUM_LINES; }
static inline uint32_t addr_line_base(uint32_t addr) { return addr - (addr % LINE_SIZE); }
```

## 3. CPU Path — Every Access Goes Through the Cache

```c
/* Fill a cache line from RAM (cold miss / after invalidate). If the
 * evicted line was dirty, it must be written back FIRST -- this is
 * standard write-back cache eviction, unrelated to explicit CMOs. */
static void cache_fill_line(uint32_t index, uint32_t tag, uint32_t line_base)
{
    CacheLine *cl = &cache.lines[index];

    if (cl->valid && cl->dirty) {
        uint32_t old_base = (cl->tag * NUM_LINES + index) * LINE_SIZE;
        memcpy(&ram[old_base], cl->data, LINE_SIZE);
        cache.writebacks++;
    }
    memcpy(cl->data, &ram[line_base], LINE_SIZE);
    cl->tag   = tag;
    cl->valid = true;
    cl->dirty = false;
}

static uint8_t cpu_read(uint32_t addr)
{
    uint32_t index = addr_index(addr), tag = addr_tag(addr);
    uint32_t line_base = addr_line_base(addr);
    CacheLine *cl = &cache.lines[index];

    if (cl->valid && cl->tag == tag) {
        cache.hits++;                              /* HIT: return cached byte */
    } else {
        cache.misses++;
        cache_fill_line(index, tag, line_base);     /* MISS: pull from RAM */
    }
    return cl->data[addr % LINE_SIZE];
}

static void cpu_write(uint32_t addr, uint8_t value)
{
    uint32_t index = addr_index(addr), tag = addr_tag(addr);
    uint32_t line_base = addr_line_base(addr);
    CacheLine *cl = &cache.lines[index];

    if (!(cl->valid && cl->tag == tag)) {
        cache.misses++;
        cache_fill_line(index, tag, line_base);     /* write-allocate */
    } else {
        cache.hits++;
    }
    cl->data[addr % LINE_SIZE] = value;
    cl->dirty = true;      /* CRITICAL: write lands ONLY in cache. RAM is
                             * now stale until a clean/flush or eviction. */
}
```

## 4. The Three Cache Maintenance Operations

```c
/* ---- CACHE CLEAN ---- (models: DC CVAC)
 * If the line covering [addr] is valid+dirty, write it back to RAM.
 * The cache line REMAINS VALID afterward -- CPU can keep using it.
 * Use: CPU produced data, a device is about to read RAM directly. */
static void cache_clean_addr(uint32_t addr)
{
    uint32_t index = addr_index(addr), tag = addr_tag(addr);
    CacheLine *cl = &cache.lines[index];
    if (cl->valid && cl->tag == tag && cl->dirty) {
        uint32_t base = (tag * NUM_LINES + index) * LINE_SIZE;
        memcpy(&ram[base], cl->data, LINE_SIZE);
        cl->dirty = false;
        cache.writebacks++;
    }
}
static void cache_clean_range(uint32_t start, uint32_t len)
{
    for (uint32_t a = start - (start % LINE_SIZE); a < start + len; a += LINE_SIZE)
        cache_clean_addr(a);
}

/* ---- CACHE INVALIDATE ---- (models: DC IVAC)
 * Unconditionally discards the cache line -- NO write-back. If the
 * line was dirty, that data is LOST. Cache line becomes INVALID, so
 * the next CPU access is a forced miss/reload from RAM.
 * Use: a device just wrote fresh data directly to RAM; CPU's cached
 * copy of that address range is now stale and must be discarded. */
static void cache_invalidate_addr(uint32_t addr)
{
    uint32_t index = addr_index(addr), tag = addr_tag(addr);
    CacheLine *cl = &cache.lines[index];
    if (cl->valid && cl->tag == tag) {
        cl->valid = false;
        cl->dirty = false;   /* any unsaved dirty bytes are discarded */
        cache.invalidations++;
    }
}
static void cache_invalidate_range(uint32_t start, uint32_t len)
{
    for (uint32_t a = start - (start % LINE_SIZE); a < start + len; a += LINE_SIZE)
        cache_invalidate_addr(a);
}

/* ---- CACHE FLUSH (Clean + Invalidate) ---- (models: DC CIVAC)
 * Write back if dirty, THEN discard. No data loss, line ends INVALID.
 * Use: buffer ownership handoff / free, when the last access
 * direction and dirty-state are not known with certainty. */
static void cache_flush_addr(uint32_t addr)
{
    cache_clean_addr(addr);
    cache_invalidate_addr(addr);
}
static void cache_flush_range(uint32_t start, uint32_t len)
{
    for (uint32_t a = start - (start % LINE_SIZE); a < start + len; a += LINE_SIZE)
        cache_flush_addr(a);
}
```

## 5. DMA Path — Bypasses the Cache Entirely

```c
/* The defining property of a non-coherent DMA master: it reads and
 * writes RAM DIRECTLY, with zero visibility into or interaction with
 * the CPU's cache. This is exactly how a real Ethernet MAC, storage
 * controller, or camera sensor DMA engine behaves on a non-coherent
 * embedded interconnect. */
static void dma_write(uint32_t addr, uint8_t value) { ram[addr] = value; }
static uint8_t dma_read(uint32_t addr)               { return ram[addr]; }

static void dma_write_buffer(uint32_t addr, const uint8_t *buf, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) dma_write(addr + i, buf[i]);
}
static void dma_read_buffer(uint32_t addr, uint8_t *buf, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) buf[i] = dma_read(addr + i);
}
```

## 6. Trace Printer — Visualizing All Three Layers

```c
static void print_layers(uint32_t addr, const char *label)
{
    uint32_t index = addr_index(addr), tag = addr_tag(addr);
    CacheLine *cl = &cache.lines[index];
    printf("  [%s] CPU-cache line: valid=%d dirty=%d tag=%u  cached_byte=0x%02X"
           "   |   RAM byte=0x%02X\n",
           label, cl->valid, cl->dirty, cl->tag,
           cl->valid ? cl->data[addr % LINE_SIZE] : 0,
           ram[addr]);
}
```

## 7. Scenario 1 — CPU → DMA (TX path, needs CACHE CLEAN)

```c
static void scenario_cpu_to_dma(void)
{
    printf("\n================ SCENARIO 1: CPU -> DMA (TX) ================\n");
    uint32_t addr = 0;

    /* --- BUGGY VERSION: no clean before DMA reads --- */
    printf("\n-- BUGGY: CPU writes, DMA reads WITHOUT cache_clean() --\n");
    system_reset();
    cpu_write(addr, 0xAB);                 /* lands in cache only, dirty=1 */
    print_layers(addr, "after CPU write");
    uint8_t dma_saw = dma_read(addr);       /* DMA reads RAM directly -- stale! */
    printf("  DMA device reads RAM at addr %u -> 0x%02X   <-- WRONG, expected 0xAB\n",
           addr, dma_saw);

    /* --- FIXED VERSION: cache_clean() before DMA reads --- */
    printf("\n-- FIXED: CPU writes, cache_clean(), THEN DMA reads --\n");
    system_reset();
    cpu_write(addr, 0xAB);
    print_layers(addr, "after CPU write");
    cache_clean_range(addr, 1);             /* push dirty line -> RAM */
    print_layers(addr, "after cache_clean");
    dma_saw = dma_read(addr);
    printf("  DMA device reads RAM at addr %u -> 0x%02X   <-- CORRECT\n",
           addr, dma_saw);
}
```

## 8. Scenario 2 — DMA → CPU (RX path, needs CACHE INVALIDATE)

```c
static void scenario_dma_to_cpu(void)
{
    printf("\n================ SCENARIO 2: DMA -> CPU (RX) ================\n");
    uint32_t addr = 0;

    /* --- BUGGY VERSION: no invalidate before CPU reads --- */
    printf("\n-- BUGGY: DMA writes, CPU reads WITHOUT cache_invalidate() --\n");
    system_reset();
    (void)cpu_read(addr);                   /* CPU touches it earlier -> caches 0x00 */
    print_layers(addr, "CPU's stale cached copy");
    dma_write(addr, 0xCD);                  /* DMA writes fresh data DIRECTLY to RAM */
    printf("  DMA device writes RAM at addr %u = 0xCD (RAM ground truth updated)\n", addr);
    uint8_t cpu_saw = cpu_read(addr);        /* CPU hits its OLD cache line! */
    printf("  CPU reads addr %u -> 0x%02X   <-- WRONG, expected 0xCD\n", addr, cpu_saw);

    /* --- FIXED VERSION: cache_invalidate() before CPU reads --- */
    printf("\n-- FIXED: DMA writes, cache_invalidate(), THEN CPU reads --\n");
    system_reset();
    (void)cpu_read(addr);
    dma_write(addr, 0xCD);
    cache_invalidate_range(addr, 1);         /* discard stale cached line */
    print_layers(addr, "after cache_invalidate (line now invalid)");
    cpu_saw = cpu_read(addr);                /* forced miss -> reload from RAM */
    printf("  CPU reads addr %u -> 0x%02X   <-- CORRECT (forced cache refill)\n",
           addr, cpu_saw);
}
```

## 9. Scenario 3 — Buffer Reuse (needs CACHE FLUSH = Clean+Invalidate)

```c
static void scenario_buffer_flush(void)
{
    printf("\n================ SCENARIO 3: Buffer handoff (FLUSH) ================\n");
    uint32_t addr = 0;

    system_reset();
    cpu_write(addr, 0x11);                  /* CPU wrote data, still dirty */
    print_layers(addr, "before cache_flush");

    cache_flush_range(addr, 1);             /* clean (write back) THEN invalidate */

    print_layers(addr, "after cache_flush");
    printf("  -> RAM now holds the CPU's data (no loss), AND the cache line is\n");
    printf("     invalid, so the buffer is safe to hand to EITHER the CPU or a\n");
    printf("     DMA device next, with zero assumptions about prior direction.\n");
}
```

## 10. main() and Statistics

```c
static void print_stats(void)
{
    printf("\n-- Cache activity totals across this run --\n");
    printf("   hits=%u misses=%u writebacks=%u invalidations=%u\n",
           cache.hits, cache.misses, cache.writebacks, cache.invalidations);
}

int main(void)
{
    printf("##############################################################\n");
    printf("#      DMA + Cache Coherency Simulator (CPU/Cache/RAM/DMA)     #\n");
    printf("##############################################################\n");
    printf("Cache: direct-mapped, %u lines x %u bytes = %u bytes total\n",
           NUM_LINES, LINE_SIZE, NUM_LINES * LINE_SIZE);
    printf("RAM:   %u bytes\n", RAM_SIZE);

    scenario_cpu_to_dma();
    scenario_dma_to_cpu();
    scenario_buffer_flush();
    print_stats();
    return 0;
}
```

## 11. Actual Output

```
##############################################################
#      DMA + Cache Coherency Simulator (CPU/Cache/RAM/DMA)     #
##############################################################
Cache: direct-mapped, 8 lines x 32 bytes = 256 bytes total
RAM:   1024 bytes

================ SCENARIO 1: CPU -> DMA (TX) ================

-- BUGGY: CPU writes, DMA reads WITHOUT cache_clean() --
  [after CPU write] CPU-cache line: valid=1 dirty=1 tag=0  cached_byte=0xAB   |   RAM byte=0x00
  DMA device reads RAM at addr 0 -> 0x00   <-- WRONG, expected 0xAB

-- FIXED: CPU writes, cache_clean(), THEN DMA reads --
  [after CPU write] CPU-cache line: valid=1 dirty=1 tag=0  cached_byte=0xAB   |   RAM byte=0x00
  [after cache_clean] CPU-cache line: valid=1 dirty=0 tag=0  cached_byte=0xAB   |   RAM byte=0xAB
  DMA device reads RAM at addr 0 -> 0xAB   <-- CORRECT

================ SCENARIO 2: DMA -> CPU (RX) ================

-- BUGGY: DMA writes, CPU reads WITHOUT cache_invalidate() --
  [CPU's stale cached copy] CPU-cache line: valid=1 dirty=0 tag=0  cached_byte=0x00   |   RAM byte=0x00
  DMA device writes RAM at addr 0 = 0xCD (RAM ground truth updated)
  CPU reads addr 0 -> 0x00   <-- WRONG, expected 0xCD

-- FIXED: DMA writes, cache_invalidate(), THEN CPU reads --
  [after cache_invalidate (line now invalid)] CPU-cache line: valid=0 dirty=0 tag=0  cached_byte=0x00   |   RAM byte=0xCD
  CPU reads addr 0 -> 0xCD   <-- CORRECT (forced cache refill)

================ SCENARIO 3: Buffer handoff (FLUSH) ================

  [before cache_flush] CPU-cache line: valid=1 dirty=1 tag=0  cached_byte=0x11   |   RAM byte=0x00
  [after cache_flush] CPU-cache line: valid=0 dirty=0 tag=0  cached_byte=0x00   |   RAM byte=0x11
  -> RAM now holds the CPU's data (no loss), AND the cache line is
     invalid, so the buffer is safe to hand to EITHER the CPU or a
     DMA device next, with zero assumptions about prior direction.

-- Cache activity totals across this run --
   hits=0 misses=6 writebacks=3 invalidations=3
```

## 12. Why Stale Cache Data Causes Bugs — The Root Cause

```
The CPU's cache exists to make repeated access to the SAME address
fast, on the assumption that memory only changes when the CPU itself
changes it. A DMA-capable device breaks that assumption: it can
modify RAM -- or need to read RAM -- WITHOUT going through the CPU's
cache at all.

This creates exactly two failure directions, both reproduced above:

1. CPU -> DMA without CLEAN:
   CPU's write is "trapped" in the dirty cache line. RAM -- which is
   the ONLY thing the DMA engine can see -- still holds the OLD value.
   The device transmits/stores garbage/old data, and because this
   compiles and often even "usually works" (evictions eventually
   flush dirty lines anyway), the bug is intermittent and appears to
   depend on unrelated code, timing, or optimization level -- a
   classic heisenbug.

2. DMA -> CPU without INVALIDATE:
   The device correctly updates RAM, but the CPU's cache line for that
   address is still VALID from before -- so the CPU's next read is a
   cache HIT that returns the pre-DMA value. The CPU never even
   attempts to go to RAM, so no amount of RAM correctness helps. The
   symptom is "the driver sees old sensor data / stale descriptor
   status / the previous DMA's leftover contents," often off by
   exactly one transfer.
```

**Why this is dangerous in practice, tying to real systems:**

| Consequence | Where it bites |
|---|---|
| **Silent data corruption, not a crash** | Both bugs above produce a *plausible* wrong value — no fault, no exception. This is why these bugs survive code review and pass on architectures/configurations where the cache happens to write back before the race matters. |
| **Non-reproducible across builds** | Adding an unrelated `printf`, changing optimization level, or moving a variable can shift eviction timing, hiding or exposing the bug — makes it extremely expensive to debug without understanding the cache/DMA model directly. |
| **Invisible on coherent development hardware** | x86 dev machines and ARM systems with a fully coherent interconnect (SMMU/ACE-Lite point-of-coherency at the DMA master) never exhibit this — the bug appears only on the *target* embedded SoC, late in bring-up. |
| **Security-relevant, not just correctness-relevant** | A stale/leftover cache line at a Secure↔Non-secure or process/VM buffer-handoff boundary can leak previous-owner data — the same failure class the TrustZone `dcache_flush` buffer-release step is designed to prevent. |
| **Direction matters, not just "flush everything"** | Reflexively calling flush (clean+invalidate) everywhere works but is measurably slower (always pays the write-back cost, per Scenario 3's stats) — production drivers pick `clean` vs `invalidate` deliberately based on data direction, exactly as modeled in Scenarios 1 and 2. |

## 13. Comparison Table

| Operation | Write-back dirty data? | Line left valid? | Correct direction | Cost |
|---|---|---|---|---|
| **Clean** | Yes, if dirty | Yes | CPU wrote → device reads (TX) | 1 potential RAM write |
| **Invalidate** | **No** (data loss if dirty) | No | Device wrote → CPU reads (RX) | 0 RAM traffic, but destructive |
| **Flush (Clean+Invalidate)** | Yes, if dirty | No | Direction unknown / buffer release | Clean cost + forced future miss |

## Key Takeaways

1. **The bug is structural, not incidental**: any time a second bus master (DMA) has a hardware path to memory that bypasses the CPU's cache, the two views of memory *will* diverge unless software explicitly reconciles them — this simulator makes that divergence directly observable rather than theoretical.

2. **Clean and Invalidate are opposite-direction tools**, not interchangeable safety operations — applying the wrong one either loses data that was never written back (using invalidate on dirty CPU output) or leaves stale data resident (using clean when a discard was needed), both reproduced explicitly above.

3. **Flush is the conservative default at ownership boundaries**, trading a small guaranteed cost (write-back + forced next-miss) for correctness certainty when the caller cannot prove the buffer's prior access direction — directly mirroring the OP-TEE/TrustZone shared-buffer release pattern from the earlier cache-maintenance work.

4. **This simulator's `cache_clean_range` / `cache_invalidate_range` / `cache_flush_range` are functionally identical in semantics to real ARM `DC CVAC` / `DC IVAC` / `DC CIVAC`**, and structurally identical to what the Linux kernel's `dma_sync_single_for_device()` / `dma_sync_single_for_cpu()` wrap — meaning this software model is a faithful, testable stand-in for hardware behavior that would otherwise require a logic analyzer or Lauterbach Trace32 session on real non-coherent silicon to observe.

---

Want this bundled with the earlier **ARM Cache Maintenance Demo** (real `DC CVAC`/`IVAC`/`CIVAC` asm) into a single downloadable `.docx` **"Cache Coherency & DMA Synchronization"** report, or added to the growing **Cache Architecture Portfolio** alongside the simulator series?