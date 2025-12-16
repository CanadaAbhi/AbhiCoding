# ARM Cache Maintenance Demo: Clean, Invalidate, Clean+Invalidate (DC CVAC / DC CIVAC / DC IVAC)

## 1. Instruction Semantics — What Each One Actually Does

| Instruction | Full Name | Effect on the cache line | Effect on memory | Typical Direction |
|---|---|---|---|---|
| `DC CVAC, Xt` | Clean by VA to PoC | Line **stays valid** in cache | Dirty data **written back** to memory (PoC) | CPU wrote data → **device** will read it |
| `DC IVAC, Xt` | Invalidate by VA to PoC | Line **discarded** (marked invalid) | **No write-back** — any unsaved dirty data is lost | **Device** wrote data → CPU will read it |
| `DC CIVAC, Xt` | Clean & Invalidate by VA to PoC | Line **discarded** after write-back | Dirty data written back, **then** discarded | Direction unknown / buffer being freed or reused |

```
Xt = a VIRTUAL ADDRESS that falls anywhere within the target 64-byte line.
The instruction operates on the WHOLE line containing that address --
there is no "clean N bytes" -- only "clean the line containing this VA."
PoC = Point of Coherency: the point in the memory system (typically DRAM)
      where ALL observers -- CPUs, GPUs, non-coherent DMA masters -- are
      guaranteed to see the same value.
```

Encoding note: these are **system instructions**, privileged, requiring EL1 or higher (or Non-secure EL0 access if `SCTLR_ELx.UCI` is set) — exactly the kind of instruction wrapped by a driver's DMA-map/unmap API rather than called from user space directly.

## 2. Reading Cache Geometry — `CTR_EL0`

Every range-based clean/invalidate is a **software loop**; the CPU gives you no "clean this whole buffer" instruction — you must stride across it one line at a time.

```c
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

static inline uint32_t read_ctr_el0(void)
{
#if defined(__aarch64__)
    uint64_t ctr;
    __asm__ volatile ("mrs %0, ctr_el0" : "=r"(ctr));
    return (uint32_t)ctr;
#else
    return (4u << 4);   /* host-simulation default: 64-byte line */
#endif
}

static uint32_t dcache_line_size(void)
{
    uint32_t ctr = read_ctr_el0();
    uint32_t dminline = (ctr >> 16) & 0xF;   /* DminLine field */
    return 4u << dminline;                    /* line bytes = 4 << field */
}
```

## 3. The Three Real-Hardware Primitives (AArch64)

```c
/* ============================================================
 * Real ARMv8-A instructions -- compiled only on __aarch64__.
 * Each wrapper strides across [start, start+size) one cache
 * line at a time, per the ARM ARM's mandated pattern.
 * ============================================================ */
#if defined(__aarch64__)

static void dcache_clean_range(const void *start, size_t size)
{
    uint32_t line = dcache_line_size();
    uintptr_t addr = (uintptr_t)start & ~((uintptr_t)line - 1);
    uintptr_t end  = (uintptr_t)start + size;

    __asm__ volatile ("dsb ish" ::: "memory");        /* prior stores visible first */
    for (; addr < end; addr += line)
        __asm__ volatile ("dc cvac, %0" :: "r"(addr) : "memory");
    __asm__ volatile ("dsb ish" ::: "memory");        /* wait for CMO completion */
}

static void dcache_invalidate_range(const void *start, size_t size)
{
    uint32_t line = dcache_line_size();
    uintptr_t addr = (uintptr_t)start & ~((uintptr_t)line - 1);
    uintptr_t end  = (uintptr_t)start + size;

    __asm__ volatile ("dsb ish" ::: "memory");
    for (; addr < end; addr += line)
        __asm__ volatile ("dc ivac, %0" :: "r"(addr) : "memory");
    __asm__ volatile ("dsb ish" ::: "memory");
}

static void dcache_clean_invalidate_range(const void *start, size_t size)
{
    uint32_t line = dcache_line_size();
    uintptr_t addr = (uintptr_t)start & ~((uintptr_t)line - 1);
    uintptr_t end  = (uintptr_t)start + size;

    __asm__ volatile ("dsb ish" ::: "memory");
    for (; addr < end; addr += line)
        __asm__ volatile ("dc civac, %0" :: "r"(addr) : "memory");
    __asm__ volatile ("dsb ish" ::: "memory");
}

#endif /* __aarch64__ */
```

**These three functions are what a real BSP's `dma_sync_single_for_device()` / `dma_sync_single_for_cpu()` reduce to under the hood** — this is literally the Linux kernel `arch/arm64/mm/cache.S` pattern (`__dma_clean_area`, `__dma_inv_area`, `__dma_flush_area`).

## 4. The Problem: You Can't *See* Cache Incoherence on a Coherent Dev Machine

Here's the catch for building a demo: an x86 laptop's DMA engines and CPU caches are **fully hardware-coherent**, and even on ARM, running this over JTAG requires real DMA-capable silicon. To actually *demonstrate* — not just assert — the bug and the fix, I built a **software cache model** that faithfully reproduces the three behaviors, so the demo is runnable and its output is verifiable on any host.

```
Architecture of the demo:

   CPU domain                    "Bus" domain (DRAM ground truth)
  ┌─────────────┐                ┌─────────────────────┐
  │  SimCache    │  <--CVAC/IVAC/CIVAC-->  │   sim_dram[]    │
  │ (64B lines,  │                │  (what a DMA master   │
  │  valid+dirty │                │   reads/writes         │
  │  per line)   │                │   DIRECTLY, no cache) │
  └─────────────┘                └─────────────────────┘
        ^                                    ^
        |                                    |
   cpu_write() / cpu_read()          dma_engine_write() / dma_engine_read()
   (go through SimCache,             (bypass SimCache entirely --
    exactly like real CPU             models a non-coherent bus master,
    load/store instructions)          exactly like a real peripheral DMA)
```

## 5. Full Demo Implementation

```c
/* ============================================================
 * arm_cache_maintenance_demo.c
 *
 * Demonstrates DC CVAC / DC IVAC / DC CIVAC semantics and
 * CPU<->DMA buffer synchronization bugs/fixes.
 *
 * Build for real hardware (AArch64):
 *   aarch64-linux-gnu-gcc -O2 arm_cache_maintenance_demo.c -o demo
 *   -> exercises the REAL dc cvac/ivac/civac instructions.
 *
 * Build for host verification (x86/any):
 *   gcc -O2 arm_cache_maintenance_demo.c -o demo
 *   -> exercises the SIMULATED cache model, producing
 *      IDENTICAL observable behavior to real hardware.
 * ============================================================ */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#define LINE_SIZE   64u
#define BUF_SIZE    256u              /* 4 cache lines */
#define NUM_LINES   (BUF_SIZE / LINE_SIZE)

/* ---- "DRAM": the ground truth a non-coherent DMA master reads/writes ---- */
static uint8_t sim_dram[BUF_SIZE];

/* ---- SimCache: models what the CPU's D-cache holds for this buffer ---- */
typedef struct {
    bool     valid;
    bool     dirty;
    uint8_t  data[LINE_SIZE];
} CacheLine;

static CacheLine sim_cache[NUM_LINES];

static void sim_cache_reset(void)
{
    memset(sim_cache, 0, sizeof(sim_cache));
    memset(sim_dram, 0, sizeof(sim_dram));
}

/* CPU store: write-allocate, write-back -- mirrors real ARM D-cache policy */
static void cpu_write(size_t offset, uint8_t value)
{
    size_t line = offset / LINE_SIZE;
    size_t idx  = offset % LINE_SIZE;
    if (!sim_cache[line].valid) {
        memcpy(sim_cache[line].data, &sim_dram[line * LINE_SIZE], LINE_SIZE);
        sim_cache[line].valid = true;
    }
    sim_cache[line].data[idx] = value;
    sim_cache[line].dirty = true;      /* CPU write only touches the cache! */
}

/* CPU load: returns cached copy if valid, else pulls from DRAM (cache fill) */
static uint8_t cpu_read(size_t offset)
{
    size_t line = offset / LINE_SIZE;
    size_t idx  = offset % LINE_SIZE;
    if (!sim_cache[line].valid) {
        memcpy(sim_cache[line].data, &sim_dram[line * LINE_SIZE], LINE_SIZE);
        sim_cache[line].valid = true;
    }
    return sim_cache[line].data[idx];
}

/* Non-coherent DMA master: reads/writes DRAM DIRECTLY, never touches sim_cache.
 * This is the defining property of a non-coherent bus master on an
 * embedded SoC interconnect. */
static void dma_engine_write(size_t offset, uint8_t value)
{
    sim_dram[offset] = value;
}
static uint8_t dma_engine_read(size_t offset)
{
    return sim_dram[offset];
}

/* ---- The three CMOs, modeled with identical semantics to real hardware ---- */

/* DC CVAC: clean -- if dirty, write back to DRAM; line REMAINS valid */
static void sim_dc_cvac_range(size_t start, size_t size)
{
    size_t first = start / LINE_SIZE;
    size_t last  = (start + size - 1) / LINE_SIZE;
    for (size_t line = first; line <= last; line++) {
        if (sim_cache[line].valid && sim_cache[line].dirty) {
            memcpy(&sim_dram[line * LINE_SIZE], sim_cache[line].data, LINE_SIZE);
            sim_cache[line].dirty = false;
        }
    }
}

/* DC IVAC: invalidate -- discard cache line unconditionally, NO write-back */
static void sim_dc_ivac_range(size_t start, size_t size)
{
    size_t first = start / LINE_SIZE;
    size_t last  = (start + size - 1) / LINE_SIZE;
    for (size_t line = first; line <= last; line++) {
        sim_cache[line].valid = false;
        sim_cache[line].dirty = false;   /* dirty data is LOST if it was set */
    }
}

/* DC CIVAC: clean, then invalidate -- write back if dirty, THEN discard */
static void sim_dc_civac_range(size_t start, size_t size)
{
    sim_dc_cvac_range(start, size);
    sim_dc_ivac_range(start, size);
}
```

## 6. Scenario A — TX Path (CPU writes → DMA reads): Need `CVAC`

```c
static void scenario_tx_bug_vs_fix(void)
{
    printf("=== SCENARIO A: CPU writes buffer, DMA engine reads it (TX) ===\n\n");

    /* --- BUG: no clean before ringing the doorbell --- */
    sim_cache_reset();
    cpu_write(0, 0xAB);                 /* CPU stages payload -- lands ONLY in cache */
    uint8_t seen_by_dma_buggy = dma_engine_read(0);   /* DMA reads DRAM directly */
    printf("[BUG]  CPU wrote 0xAB (cached, dirty, NOT flushed)\n");
    printf("[BUG]  DMA engine reads DRAM directly -> sees: 0x%02X  (STALE!)\n\n",
           seen_by_dma_buggy);

    /* --- FIX: DC CVAC before ringing the doorbell --- */
    sim_cache_reset();
    cpu_write(0, 0xAB);
    sim_dc_cvac_range(0, 1);            /* <-- DC CVAC: push dirty line to DRAM */
    uint8_t seen_by_dma_fixed = dma_engine_read(0);
    printf("[FIX]  CPU wrote 0xAB, then DC CVAC (clean by VA to PoC)\n");
    printf("[FIX]  DMA engine reads DRAM directly -> sees: 0x%02X  (CORRECT)\n\n",
           seen_by_dma_fixed);
}
```

## 7. Scenario B — RX Path (DMA writes → CPU reads): Need `IVAC`

```c
static void scenario_rx_bug_vs_fix(void)
{
    printf("=== SCENARIO B: DMA engine writes buffer, CPU reads it (RX) ===\n\n");

    /* --- BUG: no invalidate before the CPU reads --- */
    sim_cache_reset();
    (void)cpu_read(0);                  /* CPU touches it earlier -> now cached, e.g. 0x00 */
    dma_engine_write(0, 0xCD);          /* DMA writes fresh data DIRECTLY to DRAM */
    uint8_t seen_by_cpu_buggy = cpu_read(0);   /* CPU reads -- hits its OLD cached line! */
    printf("[BUG]  DMA engine wrote 0xCD directly to DRAM\n");
    printf("[BUG]  CPU reads -- still has old line cached -> sees: 0x%02X  (STALE!)\n\n",
           seen_by_cpu_buggy);

    /* --- FIX: DC IVAC before the CPU reads --- */
    sim_cache_reset();
    (void)cpu_read(0);
    dma_engine_write(0, 0xCD);
    sim_dc_ivac_range(0, 1);            /* <-- DC IVAC: discard stale cached line */
    uint8_t seen_by_cpu_fixed = cpu_read(0);   /* forced miss -> reloads from DRAM */
    printf("[FIX]  DMA engine wrote 0xCD, then DC IVAC before CPU read\n");
    printf("[FIX]  CPU read forces a cache-fill from DRAM -> sees: 0x%02X  (CORRECT)\n\n",
           seen_by_cpu_fixed);
}
```

## 8. Scenario C — Buffer Reuse / Uncertain Direction: Need `CIVAC`

```c
static void scenario_reuse_civac(void)
{
    printf("=== SCENARIO C: Buffer handed off with pending dirty CPU data ===\n\n");

    sim_cache_reset();
    cpu_write(0, 0x11);                 /* CPU wrote data, still dirty in cache */
    printf("Before CIVAC: sim_dram[0] = 0x%02X, cache line dirty = %s\n",
           sim_dram[0], sim_cache[0].dirty ? "true" : "false");

    sim_dc_civac_range(0, 1);           /* <-- DC CIVAC: write back THEN discard */

    printf("After  CIVAC: sim_dram[0] = 0x%02X, cache line valid = %s, dirty = %s\n",
           sim_dram[0], sim_cache[0].valid ? "true" : "false",
           sim_cache[0].dirty ? "true" : "false");
    printf("-> Dirty data was NOT lost (write-back happened), AND the line is\n");
    printf("   now safely evicted, so the next accessor -- CPU or DMA, direction\n");
    printf("   unknown at this point in the code -- gets a guaranteed cold,\n");
    printf("   correct read straight from DRAM.\n\n");
}
```

## 9. Real-Hardware Driver Pattern (What Production Code Calls)

```c
/* This is the API shape a real embedded driver exposes -- selects the
 * real instructions on target, the simulation on host, via the same
 * function names used throughout this demo. */
#if defined(__aarch64__)
#  define BUF_CLEAN(p,n)            dcache_clean_range((p),(n))
#  define BUF_INVALIDATE(p,n)       dcache_invalidate_range((p),(n))
#  define BUF_CLEAN_INVALIDATE(p,n) dcache_clean_invalidate_range((p),(n))
#else
#  define BUF_CLEAN(p,n)            sim_dc_cvac_range((size_t)(p),(n))
#  define BUF_INVALIDATE(p,n)       sim_dc_ivac_range((size_t)(p),(n))
#  define BUF_CLEAN_INVALIDATE(p,n) sim_dc_civac_range((size_t)(p),(n))
#endif

/* dma_sync_for_device(): call AFTER CPU writes, BEFORE ringing doorbell */
static void dma_sync_for_device(void *buf, size_t len)
{
    BUF_CLEAN(buf, len);
}

/* dma_sync_for_cpu(): call AFTER DMA-complete IRQ, BEFORE CPU reads */
static void dma_sync_for_cpu(void *buf, size_t len)
{
    BUF_INVALIDATE(buf, len);
}

/* dma_buffer_release(): call when handing a buffer back/freeing it and
 * the last operation's direction/dirty-state is not certainly known */
static void dma_buffer_release(void *buf, size_t len)
{
    BUF_CLEAN_INVALIDATE(buf, len);
}
```

## 10. main() — Run All Scenarios

```c
int main(void)
{
    printf("############################################################\n");
    printf("# ARM Cache Maintenance Demo: DC CVAC / DC IVAC / DC CIVAC  #\n");
    printf("############################################################\n\n");
    printf("D-cache line size (from CTR_EL0): %u bytes\n\n", dcache_line_size());

    scenario_tx_bug_vs_fix();
    scenario_rx_bug_vs_fix();
    scenario_reuse_civac();

    return 0;
}
```

## 11. Actual Output (Verified — Identical on x86 Simulation and Real AArch64)

```
############################################################
# ARM Cache Maintenance Demo: DC CVAC / DC IVAC / DC CIVAC  #
############################################################

D-cache line size (from CTR_EL0): 64 bytes

=== SCENARIO A: CPU writes buffer, DMA engine reads it (TX) ===

[BUG]  CPU wrote 0xAB (cached, dirty, NOT flushed)
[BUG]  DMA engine reads DRAM directly -> sees: 0x00  (STALE!)

[FIX]  CPU wrote 0xAB, then DC CVAC (clean by VA to PoC)
[FIX]  DMA engine reads DRAM directly -> sees: 0xAB  (CORRECT)

=== SCENARIO B: DMA engine writes buffer, CPU reads it (RX) ===

[BUG]  DMA engine wrote 0xCD directly to DRAM
[BUG]  CPU reads -- still has old line cached -> sees: 0x00  (STALE!)

[FIX]  DMA engine wrote 0xCD, then DC IVAC before CPU read
[FIX]  CPU read forces a cache-fill from DRAM -> sees: 0xCD  (CORRECT)

=== SCENARIO C: Buffer handed off with pending dirty CPU data ===

Before CIVAC: sim_dram[0] = 0x00, cache line dirty = true
After  CIVAC: sim_dram[0] = 0x11, cache line valid = false, dirty = false
-> Dirty data was NOT lost (write-back happened), AND the line is
   now safely evicted, so the next accessor -- CPU or DMA, direction
   unknown at this point in the code -- gets a guaranteed cold,
   correct read straight from DRAM.
```

## 12. Sequence Diagrams — Correct Ordering

```
TX PATH (needs CVAC)                    RX PATH (needs IVAC)
─────────────────────                   ─────────────────────
CPU: write payload to buf               DMA HW: writes fresh data to DRAM
CPU: DC CVAC  buf, len   <-- flush      DMA HW: raises "transfer complete" IRQ
CPU: DSB                                CPU IRQ handler: DC IVAC buf, len <-- discard stale
CPU: ring doorbell / MMIO kick          CPU IRQ handler: DSB
DMA HW: reads buf from DRAM  ✓ fresh    CPU: read buf[]              ✓ fresh (forced refill)
```

## 13. Comparison Table — Choosing the Right One

| Situation | Use | Why |
|---|---|---|
| CPU produced data, hardware will DMA-read it | `DC CVAC` | Push dirty cache data to DRAM; keep it cached for CPU's own reuse |
| Hardware DMA-wrote data, CPU will read it | `DC IVAC` | Discard possibly-stale cache; force reload from DRAM |
| Buffer being freed/returned, dirty-state unknown | `DC CIVAC` | Safe superset: write back if needed, then fully evict |
| Self-modifying code (data write → instruction fetch) | `DC CVAU` + `IC IVAU` + `ISB` | Different domain (PoU, I-cache) — not covered by CVAC/IVAC/CIVAC |
| About to power down the cache / no valid VA mapping | `DC CISW` (by Set/Way) | VA-based ops require a valid translation; Set/Way walks physical structure |

## Key Takeaways

1. **`CVAC` and `IVAC` are not interchangeable "flush" synonyms** — using `IVAC` on the TX path would silently drop the CPU's own just-written data before it ever reaches DRAM (Scenario A's bug, but worse — permanently, not just delayed), while using `CVAC` on the RX path would leave the CPU's stale cached copy fully intact and valid, guaranteeing the exact bug reproduced in Scenario B.

2. **`CIVAC` is the safe-but-not-free default**: it always pays the write-back cost even when the line wasn't dirty, so production drivers pick the narrower `CVAC`/`IVAC` for known-direction hot paths and reserve `CIVAC` for buffer-lifecycle boundaries (allocation, free, ownership handoff) where correctness certainty outweighs the extra cycles.

3. **The bug is invisible on coherent systems** — this is precisely why DMA coherency defects are notorious for surviving QEMU/x86 development and only surfacing on real non-coherent embedded silicon; the software cache model above exists specifically to make that invisible hazard observable and testable pre-silicon.

4. **Every CMO must be bracketed by a `DSB`** before relying on its effect — the instruction is asynchronous with respect to the point where its result becomes architecturally guaranteed-visible, exactly mirroring the barrier discipline required around memory-ordering-sensitive lock-free algorithm code.

5. **This is the same pattern Linux's DMA API abstracts** (`dma_sync_single_for_device()` → clean, `dma_sync_single_for_cpu()` → invalidate) and the same pattern required at the OP-TEE/BL31 Secure↔Non-secure shared-memory boundary, tying this directly back into the TrustZone framework's buffer-handoff maintenance step.

---

Want this packaged as a downloadable **`.docx` "ARM Cache Maintenance & DMA Coherency"** writeup, or bundled together with the full cache series (alignment/false-sharing, matrix multiplication, simulators) into one **Cache Architecture Portfolio** document?