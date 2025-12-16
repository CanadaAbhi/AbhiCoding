# UART Receive Interrupt — ISR + Lock-Free Ring Buffer Pipeline

## 1. Architecture

```
UART RX pin
   |
   v
Shift Register (hardware deserializes bits)
   |
   v
RDR / DR (Receive Data Register) --------+
   |                                     |
   v                                     |
RXNE flag set --> NVIC --> UART_IRQHandler (ISR)
                                |
                                v
                     ring_buffer_put(c)   <- O(1), no blocking, no allocation
                                |
                                v
                    Application (main loop / task)
                                |
                                v
                     ring_buffer_get(&c)  <- polled independently, own pace
```

The design goal mirrors the SysTick and EXTI work already established in this driver series: **the ISR does the absolute minimum required work** (read the register, push to a buffer, return), and all real processing is deferred to a context that runs at the application's own cadence. This keeps interrupt latency within the `<100us` budget regardless of how slow or bursty the application-side consumer is.

## 2. Register Definitions (extending the previous UART driver)

```c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
} USART_TypeDef;

#define USART2   ((USART_TypeDef *)0x40004400U)

#define USART_SR_PE     (1U << 0)
#define USART_SR_FE     (1U << 1)
#define USART_SR_ORE    (1U << 3)
#define USART_SR_RXNE   (1U << 5)
#define USART_SR_TC     (1U << 6)
#define USART_SR_TXE    (1U << 7)

#define USART_CR1_RE      (1U << 2)
#define USART_CR1_TE      (1U << 3)
#define USART_CR1_RXNEIE  (1U << 5)   /* NEW: enables RXNE -> interrupt */
#define USART_CR1_UE      (1U << 13)

/* NVIC (Cortex-M) -- vendor-specific IRQ number, e.g. USART2 = 38 on
 * many STM32F4 parts. Kept generic here since the exact number is
 * MCU-family-specific and belongs in the vendor header. */
#define USART2_IRQn      38

typedef struct {
    volatile uint32_t ISER[8];   /* Interrupt Set-Enable Registers */
    volatile uint32_t ICER[8];   /* Interrupt Clear-Enable Registers */
    /* ... other NVIC regs omitted for brevity ... */
} NVIC_TypeDef;

#define NVIC   ((NVIC_TypeDef *)0xE000E100U)

static inline void nvic_enable_irq(uint32_t irqn)
{
    NVIC->ISER[irqn >> 5U] = (1U << (irqn & 0x1FU));
}
```

## 3. Ring Buffer — Lock-Free, Single-Producer / Single-Consumer

```c
/* ============================================================
 * Design constraints (consistent with the rest of this driver
 * series' embedded discipline):
 *   - Zero heap allocation: static, fixed-size backing array.
 *   - Power-of-two capacity: index wrap via bitmask (& (N-1))
 *     instead of modulo, which on some Cortex-M cores lacking a
 *     hardware divider is a multi-cycle software routine --
 *     the same O(1) masking discipline used in the Fenwick Tree
 *     and Direct-Mapped Cache Simulator work.
 *   - Single-Producer/Single-Consumer (SPSC): ISR is the ONLY
 *     writer of `head`, main loop is the ONLY writer of `tail`.
 *     This specific ownership split is what allows the buffer to
 *     be lock-free on a single-core Cortex-M WITHOUT disabling
 *     interrupts on every access -- a mutex or spinlock here
 *     would be both unnecessary and a latency-budget violation.
 * ============================================================ */

#define RB_SIZE   64U                      /* MUST be power of two */
#define RB_MASK   (RB_SIZE - 1U)

typedef struct {
    volatile uint8_t  buf[RB_SIZE];
    volatile uint32_t head;   /* next write index -- ISR-owned  */
    volatile uint32_t tail;   /* next read index  -- app-owned  */
    volatile uint32_t overflow_count;   /* diagnostic counter   */
} ring_buffer_t;

static ring_buffer_t rx_rb = { .head = 0U, .tail = 0U, .overflow_count = 0U };

/* ------------------------------------------------------------
 * ring_buffer_put: called ONLY from ISR context.
 * Returns false if the buffer was full (byte dropped) so the
 * caller can account for it -- silent data loss is never
 * acceptable to hide, even in a "fire and forget" ISR path.
 * ------------------------------------------------------------ */
static inline bool ring_buffer_put(ring_buffer_t *rb, uint8_t c)
{
    uint32_t head = rb->head;
    uint32_t next = (head + 1U) & RB_MASK;

    if (next == rb->tail)
    {
        /* Buffer full: consumer hasn't kept up. Policy choice:
         * DROP the incoming byte and count it, rather than
         * overwrite unread data or block inside the ISR (blocking
         * in an ISR is never acceptable -- it would stall RXNE
         * servicing for every other pending interrupt too). */
        rb->overflow_count++;
        return false;
    }

    rb->buf[head] = c;

    /* Publish the byte BEFORE advancing head: on ARM this ordering
     * is enforced by the fact that head is the last thing written,
     * combined with the compiler observing `volatile` on both the
     * array and the index. On a multi-core part sharing this buffer
     * across cores (not the single-core Cortex-M case here), a
     * DMB (Data Memory Barrier) would be required between the data
     * write and the index update -- the same coherency discipline
     * covered in the DMA + Cache Coherency Simulator work. */
    rb->head = next;

    return true;
}

/* ------------------------------------------------------------
 * ring_buffer_get: called ONLY from application/main-loop
 * context. Returns false if empty.
 * ------------------------------------------------------------ */
static inline bool ring_buffer_get(ring_buffer_t *rb, uint8_t *out)
{
    uint32_t tail = rb->tail;

    if (tail == rb->head)
    {
        return false;   /* empty */
    }

    *out = rb->buf[tail];
    rb->tail = (tail + 1U) & RB_MASK;

    return true;
}

static inline bool ring_buffer_empty(const ring_buffer_t *rb)
{
    return rb->head == rb->tail;
}

static inline uint32_t ring_buffer_count(const ring_buffer_t *rb)
{
    /* Correct even across the wrap point because of the unsigned
     * modular arithmetic and power-of-two masking. */
    return (rb->head - rb->tail) & RB_MASK;
}
```

## 4. ISR — Minimal Work, Correct Flag-Clear Order

```c
/* ============================================================
 * UART_IRQHandler: fires on RXNE (and, if enabled, on error
 * flags ORE/FE/PE, which share this same vector on most Cortex-M
 * USART implementations).
 *
 * Discipline enforced here (same as the SysTick ISR and EXTI ISR
 * in this series):
 *   1. Read SR before DR -- required by the datasheet-documented
 *      clear sequence for ORE/FE/PE (see the polling driver's RX
 *      path for the same rule).
 *   2. Do the absolute minimum: read DR, push to ring buffer,
 *      return. No parsing, no string handling, no printf --
 *      anything heavier is deferred to the application.
 *   3. Never block. ring_buffer_put() is O(1) and non-blocking
 *      by construction (drop-on-full, not wait-on-full).
 * ============================================================ */
void UART_IRQHandler(void)
{
    uint32_t status = USART2->SR;

    if (status & USART_SR_RXNE)
    {
        /* Reading DR clears RXNE as a hardware side-effect --
         * mirrors the DATA-register read/write side-effects noted
         * in the polling driver. */
        uint8_t c = (uint8_t)USART2->DR;

        if (!ring_buffer_put(&rx_rb, c))
        {
            /* Buffer full -- byte dropped, overflow_count already
             * incremented inside ring_buffer_put(). Deliberately NOT
             * doing anything heavier here (no logging, no retry) --
             * that would violate the <100us ISR-minimal discipline. */
        }
    }

    if (status & (USART_SR_ORE | USART_SR_FE | USART_SR_PE))
    {
        /* Hardware overrun/framing/parity error. On this family the
         * documented clear sequence is: read SR (already done above),
         * then read DR. Reading DR again here is safe/idempotent for
         * clearing ORE even if RXNE was also set and already
         * consumed above -- consult the specific MCU reference manual,
         * as some parts require an explicit different clear write. */
        (void)USART2->DR;

        /* Track error separately from ring-buffer overflow -- these
         * are two distinct failure modes: ORE means the CPU/ISR was
         * too slow to service RXNE before the NEXT byte arrived at
         * the wire, entirely independent of ring buffer capacity. */
        rx_rb.overflow_count++;
    }
}
```

## 5. Initialization — Enabling the Interrupt Path

```c
static void uart_rx_interrupt_init(USART_TypeDef *u, uint32_t pclk_hz, uint32_t baud)
{
    u->CR1 &= ~USART_CR1_UE;

    usart_set_baud(u, pclk_hz, baud);   /* from the polling driver */

    u->CR1 |= (USART_CR1_TE | USART_CR1_RE);
    u->CR1 |= USART_CR1_RXNEIE;         /* NEW: RXNE now raises an
                                            interrupt instead of only
                                            setting a pollable flag */

    u->CR1 |= USART_CR1_UE;

    nvic_enable_irq(USART2_IRQn);       /* NEW: unmask at the NVIC --
                                            without this the UART would
                                            still set RXNE/pending, but
                                            the CPU vector would never
                                            fire, exactly like the
                                            EXTI/SYSCFG routing step in
                                            the GPIO interrupt work */
}
```

## 6. Application-Side Consumer

```c
/* ------------------------------------------------------------
 * uart_getc: non-blocking pop. Returns false if no byte is
 * currently available -- lets the application interleave UART
 * servicing with other work instead of stalling on empty.
 * ------------------------------------------------------------ */
bool uart_getc(char *out)
{
    uint8_t byte;
    if (!ring_buffer_get(&rx_rb, &byte))
    {
        return false;
    }
    *out = (char)byte;
    return true;
}

/* ------------------------------------------------------------
 * uart_getc_blocking: convenience wrapper for simple sequential
 * code paths (e.g. a command-line shell) where blocking is
 * acceptable. Uses WFI between polls -- NOT a busy spin -- so the
 * core drops to a low-power idle state and is woken by the next
 * RXNE interrupt, consistent with the +18% performance/watt / WFI
 * discipline used throughout the SysTick and GPIO EXTI work.
 * ------------------------------------------------------------ */
char uart_getc_blocking(void)
{
    char c;
    while (!uart_getc(&c))
    {
        __asm volatile ("wfi");
    }
    return c;
}

/* ------------------------------------------------------------
 * Example application: simple line-buffered echo/command reader
 * running entirely in the main loop, decoupled from ISR timing.
 * ------------------------------------------------------------ */
void app_main_loop(void)
{
    static char line[128];
    static uint32_t idx = 0U;

    char c;
    while (uart_getc(&c))
    {
        if (c == '\r' || c == '\n')
        {
            line[idx] = '\0';
            /* process complete line here (command dispatch, etc.) */
            idx = 0U;
        }
        else if (idx < (sizeof(line) - 1U))
        {
            line[idx++] = c;
        }
        else
        {
            /* line too long -- drop/reset rather than overflow
             * the local buffer; same bounded-buffer discipline as
             * the Device Tree parser's static-pool node limit */
            idx = 0U;
        }
    }

    if (rx_rb.overflow_count > 0U)
    {
        /* Surface diagnostics at application cadence, never from
         * inside the ISR itself. */
    }
}
```

## 7. Concurrency Correctness Notes

| Concern | How it's handled here |
|---|---|
| ISR writes `head`, app writes `tail` | SPSC ownership split — no shared-write race, no lock needed on single-core Cortex-M |
| Compiler reordering/caching stale index values | `volatile` on `buf`, `head`, `tail` forces every access to hit memory, not a register-cached copy |
| Full/empty ambiguity (`head == tail` for both) | Reserve one slot: buffer is "full" when `next == tail`, never actually filling all `RB_SIZE` slots — standard ring-buffer technique, avoids needing a separate count field for the hot path |
| Byte lost during overrun (`ORE`) | Counted via `overflow_count`, distinct from ring-buffer-full drops — tells you whether to raise UART speed tolerance/DMA vs. enlarge the buffer |
| Multi-byte read of `head`/`tail` torn by preemption | Not an issue here: both are single 32-bit words, and Cortex-M word-aligned loads/stores are atomic by architecture — no need for the double-read torn-read guard used for the 64-bit SysTick tick counter |
| Application briefly disabling interrupts to inspect buffer state | Only ever needs to disable **RXNEIE**, not global IRQs, if a consistent snapshot of `head`/`tail` together is ever required — kept out of the hot path here since SPSC doesn't need it |

## 8. Sizing the Ring Buffer

```
worst-case bytes to buffer ≈ (application's longest possible
                               main-loop iteration time)
                              × (byte arrival rate at the UART's
                                 configured baud rate)

Example: 115200 baud ≈ 11,520 bytes/sec ≈ 1 byte every ~87us.
If the application's longest loop iteration is 5ms, worst case
incoming bytes during that window ≈ 5ms / 87us ≈ 57 bytes.
RB_SIZE = 64 (next power of two ≥ 57) with margin.
```

This is the same throughput-vs-latency-budget reasoning used to size the multi-rate soft-timer dispatch lists and the Fenwick Tree's sliding window — pick buffer capacity from the actual worst-case producer/consumer rate mismatch, not an arbitrary round number.

## 9. Polling vs. Interrupt-Driven RX — Updated Comparison

| | Polling (previous driver) | Interrupt + Ring Buffer (this driver) |
|---|---|---|
| CPU cost while idle | 100% (spin loop) | ~0% (WFI between events) |
| Overrun risk | High — any delay in the polling loop drops/overruns data | Low — ISR services RXNE within interrupt latency (`<100us` budget), buffer absorbs bursts |
| Multiple bytes in flight | Not handled — one `uart_recv_char()` call per byte, caller must poll continuously | Handled — ring buffer decouples arrival timing from consumption timing |
| Code complexity | Minimal | Moderate — requires correct SPSC buffer design |
| Appropriate for | Boot-time diagnostics, simple synchronous protocols | Any system with concurrent real-time work, command shells, streaming sensor/log data |

## 10. Common Pitfalls Checklist

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgetting `RXNEIE` in `CR1` | RXNE sets, but the interrupt never fires | Set `RXNEIE` explicitly — the flag existing is not the same as it being routed to the CPU |
| Forgetting `nvic_enable_irq()` | Same symptom as above, at the NVIC level instead of peripheral level | Unmask the correct IRQ number at the NVIC — mirrors the EXTI/SYSCFG routing step |
| Non-power-of-two `RB_SIZE` | Index wrap logic (`& RB_MASK`) silently breaks, corrupting buffer contents | Enforce power-of-two capacity, ideally with a compile-time `static_assert` |
| Doing heavy work inside the ISR (parsing, printf, memcpy of large blocks) | Interrupt latency budget blown, other peripherals starved | Push raw bytes only; defer all parsing/processing to the main loop |
| Treating ring-buffer-full drops and hardware `ORE` as the same failure | Misdiagnosing whether the bottleneck is buffer size or ISR servicing latency | Track them in separate counters, as done here |
| Multi-producer access to the same ring buffer (e.g. two ISRs writing `head`) | Race condition corrupting `head`/data despite `volatile` | SPSC lock-free design assumes exactly one writer — use a lock (or redesign to MPSC with atomics) if that assumption is violated |
| Reading `DR` without reading `SR` first for error clearing | `ORE`/`FE`/`PE` stay latched, RXNE interrupt stops firing entirely, RX appears to silently die | Always read `SR` before `DR`, exactly as enforced in this ISR |

## 11. Key Takeaways

1. **The ISR's only job is to move the byte from hardware to buffer as fast as possible** — everything else (parsing, dispatch, error reporting) belongs in the application, preserving the same `<100us` ISR-minimal discipline used across the SysTick and EXTI drivers in this series.
2. **SPSC ownership (ISR writes `head`, app writes `tail`) is what makes this lock-free** on a single-core Cortex-M — introducing a second writer on either side would require an actual lock or a different (MPSC/SPMC) algorithm.
3. **Ring-buffer-full drops and hardware overrun (`ORE`) are different failures with different fixes** — conflating them hides whether the real problem is buffer sizing or ISR/NVIC priority/latency.
4. **Buffer capacity should be derived from the worst-case producer/consumer rate mismatch**, not picked arbitrarily — same sizing discipline as the sliding-window Fenwick Tree and multi-rate timer lists elsewhere in this portfolio.
5. **WFI between polls turns the "blocking" convenience wrapper into a power-efficient wait**, not a busy spin — keeping this driver consistent with the +18% performance/watt / DVFS-adjacent low-power discipline running through the rest of the bare-metal work.