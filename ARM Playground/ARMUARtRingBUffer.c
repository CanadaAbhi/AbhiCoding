# UART Ring Buffer — Circular Buffer Producer/Consumer Implementation ⭐⭐⭐

## 1. The Exact Structure (as specified)

```c
#include <stdint.h>
#include <stddef.h>

#define BUFFER_SIZE 128U

/* IMPORTANT CONSTRAINT: BUFFER_SIZE must be a power of two.
 * 128 = 2^7, so index wraparound can use a fast bitmask
 * (index & (BUFFER_SIZE - 1)) instead of a modulo operation.
 * On Cortex-M0/M0+ (no hardware divider), modulo compiles to a
 * multi-cycle software routine -- the mask is a single AND. */
typedef struct
{
    uint8_t  buffer[BUFFER_SIZE];

    volatile uint32_t head;   /* next WRITE index -- producer-owned */
    volatile uint32_t tail;   /* next READ  index -- consumer-owned */

} RingBuffer;

#define RB_MASK (BUFFER_SIZE - 1U)
```

## 2. Why head/tail-only (no count field) requires the "reserve one slot" trick

Since the struct has **no separate count/length field**, `head == tail` is the *only* signal available, and it is ambiguous — it means either **completely empty** or **completely full** (if you naively let head wrap all the way around to tail).

The standard fix: **treat the buffer as one slot smaller than its physical size**. The buffer is considered *full* when advancing `head` by one would make it equal to `tail`:

```
empty:  head == tail
full:   ((head + 1) & RB_MASK) == tail
usable capacity: BUFFER_SIZE - 1  (127 bytes usable out of 128)
```

This costs 1 byte of capacity but requires zero extra state and zero locking to disambiguate — exactly the trade-off worth making in a `<100us`-latency ISR path.

## 3. Global Instance

```c
static RingBuffer rx_ring = {
    .buffer = {0},
    .head   = 0U,
    .tail   = 0U
};
```

## 4. `ring_put` — Producer Side (called from the UART ISR)

```c
/* ============================================================
 * ring_put: PRODUCER. In this driver, called ONLY from
 * UART_IRQHandler (single-producer). Writes `head`; NEVER
 * touches `tail`.
 *
 * Design rules enforced:
 *   1. O(1), non-blocking -- an ISR must never spin/wait.
 *   2. Full buffer => DROP the new byte, do NOT overwrite
 *      unread data and do NOT block. Silent overwrite would
 *      corrupt the stream; blocking would violate interrupt
 *      latency budget and could deadlock if the consumer never
 *      runs (e.g. main loop stuck).
 *   3. Data is written into buffer[head] BEFORE head is
 *      advanced -- this ordering is what makes the buffer safe
 *      to read concurrently: the consumer only trusts an index
 *      once head has been updated to include it.
 * ============================================================ */
void ring_put(uint8_t data)
{
    uint32_t head = rx_ring.head;
    uint32_t next = (head + 1U) & RB_MASK;

    if (next == rx_ring.tail)
    {
        /* Buffer full -- consumer isn't draining fast enough.
         * Policy: drop the incoming byte.
         * (A production driver would increment a diagnostic
         * overflow counter here, e.g. rx_ring_overflow_count++,
         * read only by the application, never by the ISR logic.) */
        return;
    }

    rx_ring.buffer[head] = data;

    /* Publish the write LAST. On Cortex-M (single core, in-order
     * pipeline for this kind of access), a plain volatile store
     * is sufficient -- no DMB needed. On an SMP/multi-core target
     * sharing this buffer across cores, a store-release / DMB
     * would be required here before publishing `head`, matching
     * the coherency discipline from the DMA + Cache Coherency
     * Simulator work. */
    rx_ring.head = next;
}
```

## 5. `ring_get` — Consumer Side (called from the application/main loop)

```c
/* ============================================================
 * ring_get: CONSUMER. Called ONLY from application/main-loop
 * context (single-consumer). Writes `tail`; NEVER touches `head`.
 *
 * Return convention:
 *   0  = success, *data contains the popped byte
 *  -1  = buffer empty, *data untouched
 * ============================================================ */
int ring_get(uint8_t *data)
{
    uint32_t tail = rx_ring.tail;

    if (tail == rx_ring.head)
    {
        return -1;   /* empty */
    }

    *data = rx_ring.buffer[tail];

    /* Advance tail LAST -- until this line executes, the ISR
     * still considers this slot "in use" and will never
     * overwrite it, even if a byte arrives mid-read. */
    rx_ring.tail = (tail + 1U) & RB_MASK;

    return 0;
}
```

## 6. Helper Diagnostics (built on the same head/tail state)

```c
static inline int ring_is_empty(void)
{
    return rx_ring.head == rx_ring.tail;
}

static inline int ring_is_full(void)
{
    return ((rx_ring.head + 1U) & RB_MASK) == rx_ring.tail;
}

/* Number of unread bytes currently queued. Correct across wrap
 * because of unsigned modular arithmetic + power-of-two masking --
 * same trick used for the count fields in the Fenwick Tree and
 * multi-level cache simulator work. */
static inline uint32_t ring_count(void)
{
    return (rx_ring.head - rx_ring.tail) & RB_MASK;
}
```

## 7. Full Pipeline: ISR → Ring Buffer → Application

```c
/* ---- UART hardware access (from the earlier polling/interrupt
 * driver work) ---- */
#define UART_SR_RXNE   (1U << 5)

typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
} UART_TypeDef;

#define UART   ((UART_TypeDef *)0x40004400U)

/* ---- ISR: hardware -> ring buffer ---- */
void UART_IRQHandler(void)
{
    if (UART->SR & UART_SR_RXNE)
    {
        uint8_t c = (uint8_t)UART->DR;   /* reading DR clears RXNE */
        ring_put(c);
    }
}

/* ---- Application: ring buffer -> processing ---- */
void app_main_loop(void)
{
    uint8_t byte;

    while (ring_get(&byte) == 0)
    {
        /* process byte -- e.g. append to a command line buffer,
         * feed a protocol parser, echo it back, etc. */
    }

    __asm volatile ("wfi");   /* nothing to do -- sleep until next IRQ,
                                 consistent with the +18% perf/watt
                                 WFI discipline used elsewhere */
}
```

## 8. Concurrency Model — Why This Is Lock-Free and Correct

| Property | Guarantee |
|---|---|
| **Producer/Consumer split** | `ring_put` (ISR) writes ONLY `head`; `ring_get` (main loop) writes ONLY `tail`. Neither function writes the other's index. This Single-Producer/Single-Consumer (SPSC) split is what allows lock-free operation. |
| **`volatile` on `head`/`tail`/`buffer`** | Forces the compiler to re-read from memory on every access rather than caching a stale value in a register — critical since the ISR and main loop are, from the compiler's point of view, two independent, unsynchronized execution contexts. |
| **Word-aligned 32-bit read/write atomicity** | On ARM Cortex-M, aligned 32-bit loads/stores are inherently atomic — `head` and `tail` can never be observed "torn" (half-updated), so no lock is needed just to read/write them safely. |
| **Data-before-index ordering** | `buffer[head] = data` happens *before* `head` is advanced (`ring_put`), and the byte is read *before* `tail` is advanced (`ring_get`). This ensures the consumer never observes an index update before the corresponding data write is visible. |
| **What would break this** | Adding a **second producer** (e.g. two ISRs both calling `ring_put`) or a **second consumer** (two tasks both calling `ring_get`) turns this into MPSC/SPMC/MPMC, at which point a single unguarded `head++`/`tail++` becomes a genuine race — you would then need a critical section (`__disable_irq()`/`__enable_irq()`) or atomic CAS around the index update. |

### If a second producer/consumer were ever added:

```c
/* Only needed if this SPSC assumption is ever violated --
 * e.g. two different interrupt sources both calling ring_put(). */
#define CRITICAL_ENTER()   __asm volatile ("cpsid i")
#define CRITICAL_EXIT()    __asm volatile ("cpsie i")

void ring_put_mp_safe(uint8_t data)
{
    CRITICAL_ENTER();
    ring_put(data);
    CRITICAL_EXIT();
}
```
Not used in the default implementation above — adding unnecessary critical sections around a true SPSC buffer would only cost cycles and (worse) briefly stall *other* unrelated interrupts for no benefit.

## 9. Overflow / Underflow Behavior Summary

| Scenario | Result |
|---|---|
| `ring_put()` called when buffer is full | Byte silently dropped; `head` unchanged (call this what it is: **data loss**, and instrument it in production) |
| `ring_get()` called when buffer is empty | Returns `-1`; `*data` untouched, `tail` unchanged |
| `ring_put()`/`ring_get()` racing at the exact full/empty boundary | Safe — the reserved-slot design guarantees `head` and `tail` never collide from opposite directions mid-update |

## 10. Test / Demo Harness

```c
#include <stdio.h>
#include <assert.h>

void test_ring_buffer_wraparound(void)
{
    uint8_t out;

    /* Fill to capacity - 1 (127 usable slots) */
    for (uint32_t i = 0; i < BUFFER_SIZE - 1U; i++)
    {
        ring_put((uint8_t)(i & 0xFF));
    }
    assert(ring_is_full());

    /* One more put should be silently dropped */
    ring_put(0xAA);
    assert(ring_count() == BUFFER_SIZE - 1U);

    /* Drain and verify FIFO order is preserved */
    for (uint32_t i = 0; i < BUFFER_SIZE - 1U; i++)
    {
        int rc = ring_get(&out);
        assert(rc == 0);
        assert(out == (uint8_t)(i & 0xFF));
    }
    assert(ring_is_empty());

    /* Buffer now empty -- get should fail cleanly */
    assert(ring_get(&out) == -1);

    /* Push past the physical end of the array to prove index
     * wraparound (head/tail masking) works correctly */
    for (uint32_t i = 0; i < 200U; i++)
    {
        ring_put((uint8_t)i);
        int rc = ring_get(&out);
        assert(rc == 0);
        assert(out == (uint8_t)i);
    }

    printf("All ring buffer tests passed.\n");
}
```

## 11. Concepts Demonstrated (mapped to the learning goals)

| Concept | Where it shows up in this implementation |
|---|---|
| **Circular buffers** | `head`/`tail` indices wrap via `& RB_MASK`, turning a linear array into a logically infinite FIFO stream within fixed `BUFFER_SIZE` memory — zero heap allocation, deterministic footprint (same static-pool discipline as the Device Tree parser and Fenwick Tree work). |
| **Producer-consumer model** | `ring_put` (producer, ISR context) and `ring_get` (consumer, application context) operate at independent rates, fully decoupled by the buffer — the producer never waits on the consumer and vice versa, up to the buffer's capacity. |
| **Interrupt-driven drivers** | `ring_put` is designed to be safely callable from `UART_IRQHandler` with O(1), non-blocking, allocation-free execution — preserving the `<100us` interrupt latency discipline used throughout this driver series. |
| **Concurrency** | Correctness relies on SPSC ownership discipline (one writer per index), `volatile`-enforced memory visibility, and ARM's guaranteed atomicity of aligned word accesses — demonstrating that lock-free correctness is achievable *without* disabling interrupts or using a mutex, provided the single-producer/single-consumer contract is never violated. |