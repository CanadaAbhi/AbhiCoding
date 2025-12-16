# SysTick Timer — Periodic Interrupts for Deterministic Timing

## 1. Concept Map

```
System Clock (SystemCoreClock, e.g. 168MHz)
     |
     v
SysTick Timer (24-bit down-counter, built into every ARM Cortex-M core --
     |          NOT a peripheral, it's part of the NVIC/core itself)
     |
     |  counts down from LOAD to 0, then reloads and asserts exception
     v
Interrupt (Exception #15, fixed in the vector table on every Cortex-M)
     |
     v
SysTick_Handler() -- ISR, runs at configured priority
     |
     v
Periodic application logic (ticks++, soft-timer dispatch, RTOS tick,
                             scheduler quantum, etc.)
```

## 2. SysTick Register Map (what `SysTick_Config()` actually does under the hood)

```c
/* ============================================================
 * SysTick is memory-mapped at a FIXED address on every Cortex-M
 * core (part of the System Control Space, not vendor-specific) --
 * this is one of the few peripherals whose address is identical
 * across STM32, NXP, TI, Nordic, etc.
 * ============================================================ */

#include <stdint.h>

#define SYSTICK_BASE   0xE000E010U

typedef struct {
    volatile uint32_t CTRL;    /* 0x00: Control and Status */
    volatile uint32_t LOAD;    /* 0x04: Reload Value (24-bit) */
    volatile uint32_t VAL;     /* 0x08: Current Value */
    volatile uint32_t CALIB;   /* 0x0C: Calibration (read-only) */
} SysTick_TypeDef;

#define SysTick   ((SysTick_TypeDef *)SYSTICK_BASE)

/* CTRL register bit fields */
#define SYSTICK_CTRL_ENABLE_Pos     0U
#define SYSTICK_CTRL_TICKINT_Pos    1U   /* enable the interrupt        */
#define SYSTICK_CTRL_CLKSOURCE_Pos  2U   /* 0=external ref, 1=processor */
#define SYSTICK_CTRL_COUNTFLAG_Pos 16U   /* read: set if counted to 0
                                             since last read; clears on read */

#define SYSTICK_LOAD_RELOAD_Msk    0x00FFFFFFU   /* only 24 bits valid! */
```

## 3. Manual `SysTick_Config()` — no CMSIS dependency

```c
/* ============================================================
 * This is exactly what CMSIS's SysTick_Config() does internally.
 * Understanding it cold is a common ARM interview ask.
 * ============================================================ */

#define SYSTICK_MAX_RELOAD   0x00FFFFFFU   /* 24-bit counter limit */

/* Returns 0 on success, 1 if the requested period exceeds what
 * the 24-bit counter can hold at the given clock frequency. */
static uint32_t SysTick_Config_manual(uint32_t ticks)
{
    if ((ticks - 1U) > SYSTICK_MAX_RELOAD)
    {
        /* INTERVIEW TRAP: at 168MHz, SystemCoreClock/1000 = 168000
         * fits easily, but SystemCoreClock/1 (a 1-second period)
         * would need 168,000,000 ticks -- far beyond 0xFFFFFF
         * (16,777,215). Always validate the reload value; silently
         * truncating it produces a MUCH faster-than-intended
         * interrupt rate, which is a nasty silent bug. */
        return 1U;
    }

    SysTick->LOAD = (ticks - 1U);   /* reload = period_in_ticks - 1,
                                        since counting is 0-inclusive */
    SysTick->VAL  = 0U;             /* writing ANY value clears VAL
                                        to 0 (hardware behavior, not
                                        actually "writing zero") and
                                        also clears COUNTFLAG */

    /* Priority: SysTick priority register lives in NVIC's
     * SHPR3 (System Handler Priority Register 3), byte 3.
     * Deliberately left at default/lowest here -- production
     * code typically sets this explicitly relative to other
     * ISRs (e.g. lower priority than a hard real-time DMA/EXTI
     * handler, higher than PendSV in an RTOS). */

    SysTick->CTRL = (1U << SYSTICK_CTRL_CLKSOURCE_Pos) |  /* use core clock */
                    (1U << SYSTICK_CTRL_TICKINT_Pos)   |  /* enable interrupt */
                    (1U << SYSTICK_CTRL_ENABLE_Pos);      /* start counting */

    return 0U;
}
```

## 4. Baseline Example — Annotated, With the Hidden Bug Fixed

```c
/* ============================================================
 * The baseline `if (ticks % 1000 == 0)` pattern has a subtle
 * but real bug: the condition stays TRUE for the ENTIRE 1ms
 * tick period (until the next increment), so depending on how
 * fast the main loop spins, the "once per second" block can
 * execute HUNDREDS or THOUSANDS of times during that single
 * millisecond window -- not once.
 * ============================================================ */

#include <stdint.h>

extern uint32_t SystemCoreClock;
static uint32_t SysTick_Config_manual(uint32_t ticks);

volatile uint32_t g_ticks = 0;   /* volatile: written in ISR context,
                                     read in main() -- without volatile
                                     the compiler is free to cache the
                                     value in a register and NEVER
                                     re-read it, causing an infinite
                                     loop that never sees the update */

void SysTick_Handler(void)
{
    g_ticks++;   /* single 32-bit increment: on a 32-bit ARM core this
                     compiles to a single LDR/ADD/STR sequence that
                     cannot be interrupted mid-instruction (no other
                     interrupt can preempt SysTick's own ISR body at
                     a finer grain than an instruction), so this
                     specific increment is safe without extra locking.
                     This changes once you widen the counter -- see
                     Section 5. */
}

int main(void)
{
    /* WRONG (baseline): fires the "once per second" block
     * repeatedly for the whole 1ms window it stays true. */
    /*
    while (1)
    {
        if (g_ticks % 1000 == 0)
        {
            // executes ~thousands of times per second, not once
        }
    }
    */

    SysTick_Config_manual(SystemCoreClock / 1000U);  /* 1ms tick */

    uint32_t last_second_mark = 0;

    while (1)
    {
        uint32_t now = g_ticks;   /* single volatile read, snapshot
                                      to avoid re-reading a changing
                                      value multiple times in this
                                      iteration */

        /* CORRECT: edge-detect the transition into a new second,
         * rather than testing a level condition that stays true
         * for a whole tick period. */
        if ((now - last_second_mark) >= 1000U)
        {
            last_second_mark = now;
            /* Approximately every 1 second -- executes EXACTLY
             * once per second, regardless of main-loop speed. */
        }

        /* Subtraction-based comparison (now - last >= period)
         * instead of equality is also the standard defense
         * against MISSED ticks: if the main loop happens to be
         * busy for 3ms and skips checking during that window,
         * `now - last_second_mark` will be >= 1000 on the next
         * check and still fire correctly -- exact equality
         * (now == last + 1000) could be skipped entirely. */
    }
}
```

## 5. Atomic Multi-Word Tick Reads (64-bit millisecond counter on a 32-bit core)

```c
/* ============================================================
 * A 32-bit millisecond counter wraps in ~49.7 days
 * (2^32 / 1000 / 86400). Long-running embedded systems (fleet
 * telemetry, automotive ECUs) often need a 64-bit tick counter,
 * but a 64-bit read on a 32-bit core is TWO separate 32-bit
 * loads -- NOT atomic. The ISR can increment between the two
 * loads and produce a torn read (e.g. reading a stale high word
 * with a wrapped-around low word).
 * ============================================================ */

#include <stdint.h>

static volatile uint64_t g_ticks64 = 0;

void SysTick_Handler(void)
{
    g_ticks64++;   /* still fine to increment in ISR -- the ISR
                       itself cannot be re-entered by SysTick again
                       before it completes (same exception can't
                       preempt itself), so no lock needed HERE */
}

/* Reading from main() DOES need protection, since main() can be
 * interrupted mid-read by the very ISR that's updating the value. */
static uint64_t read_ticks64_atomic(void)
{
    uint64_t snapshot;

    __asm volatile ("cpsid i" : : : "memory");  /* disable IRQs --
                                                     PRIMASK = 1 */
    snapshot = g_ticks64;
    __asm volatile ("cpsie i" : : : "memory");  /* re-enable IRQs */

    return snapshot;

    /* Alternative without a global IRQ disable: read high word,
     * read low word, read high word again; if the two high-word
     * reads differ, a rollover happened mid-read -- retry. This
     * avoids the (short but nonzero) interrupt-latency hit of
     * cpsid/cpsie, which matters given the <100us interrupt
     * latency budgets discussed elsewhere in this portfolio. */
}
```

## 6. Multi-Rate Soft-Timer Dispatch (the realistic production pattern)

```c
/* ============================================================
 * Real firmware rarely needs just "once per second" -- it needs
 * N independent periodic tasks (e.g. 10ms sensor poll, 100ms
 * control loop, 1000ms heartbeat) all driven off a SINGLE
 * hardware SysTick, since most Cortex-M parts only have one.
 * This is the software-timer-list pattern used to build a
 * deterministic 100ms event loop on top of a 1ms hardware tick.
 * ============================================================ */

#include <stdint.h>
#include <stdbool.h>

#define MAX_SOFT_TIMERS   8U

typedef void (*timer_callback_t)(void);

typedef struct {
    uint32_t period_ms;
    uint32_t last_fire_ms;
    timer_callback_t callback;
    bool active;
} soft_timer_t;

static soft_timer_t g_timers[MAX_SOFT_TIMERS];
static volatile uint32_t g_ticks = 0;   /* incremented by SysTick_Handler */

/* Registration is a plain function, not ISR-context -- safe to
 * use normal (non-volatile-qualified) local reasoning here. */
static bool soft_timer_register(uint32_t period_ms, timer_callback_t cb)
{
    for (uint32_t i = 0; i < MAX_SOFT_TIMERS; i++)
    {
        if (!g_timers[i].active)
        {
            g_timers[i].period_ms    = period_ms;
            g_timers[i].last_fire_ms = g_ticks;
            g_timers[i].callback     = cb;
            g_timers[i].active       = true;
            return true;
        }
    }
    return false;   /* table full -- fixed-size, zero heap allocation,
                        same discipline as the Fenwick-tree and
                        cache-simulator static-pool designs */
}

/* Call this once per main-loop iteration, or once per SysTick
 * tick from a lightweight ISR-deferred flag -- NOT directly from
 * inside SysTick_Handler itself if callbacks might take a while,
 * to keep ISR execution time bounded and deterministic. */
static void soft_timer_dispatch(void)
{
    uint32_t now = g_ticks;   /* single snapshot per dispatch pass */

    for (uint32_t i = 0; i < MAX_SOFT_TIMERS; i++)
    {
        if (g_timers[i].active &&
            (now - g_timers[i].last_fire_ms) >= g_timers[i].period_ms)
        {
            /* Accumulate the period rather than resetting to `now`,
             * so a late dispatch (main loop briefly busy) doesn't
             * permanently shift the timer's phase -- keeps long-run
             * average rate accurate even under jitter. */
            g_timers[i].last_fire_ms += g_timers[i].period_ms;
            g_timers[i].callback();
        }
    }
}

/* --- Example usage --- */
static void task_10ms(void)  { /* sensor poll */ }
static void task_100ms(void) { /* control loop */ }
static void task_1000ms(void){ /* heartbeat/LED */ }

int main(void)
{
    /* ... SysTick_Config_manual(SystemCoreClock / 1000U) ... */

    soft_timer_register(10U,   task_10ms);
    soft_timer_register(100U,  task_100ms);
    soft_timer_register(1000U, task_1000ms);

    while (1)
    {
        soft_timer_dispatch();
        __asm volatile ("wfi");   /* sleep between dispatches --
                                      same power-efficiency pattern
                                      as the GPIO SysTick-delay/WFI
                                      work; core wakes automatically
                                      on the next SysTick exception */
    }
}
```

## 7. Common Pitfalls Checklist

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgetting `volatile` on the tick counter | Infinite loop / never sees update — compiler caches value in a register | Always `volatile` on ISR-shared globals |
| `ticks % N == 0` level check | Block fires many times per tick period, not once | Edge-detect with `(now - last) >= period`, update `last` |
| Reload value `SystemCoreClock` (not `/1000`) exceeding 24-bit `LOAD` | Silent truncation → interrupt fires far faster than intended | Validate `(ticks - 1) <= 0xFFFFFF` before configuring |
| 64-bit tick counter read torn by ISR preemption | Rare, hard-to-repro glitch in long-uptime timestamps | Disable-IRQ snapshot or double-read-high-word retry |
| Heavy work directly inside `SysTick_Handler()` | Increases interrupt latency for *other* ISRs, breaks `<100us` budgets | Keep ISR to `ticks++` / flag-set only; defer real work to main loop or soft-timer dispatch |
| Using `while(1);` busy-wait instead of `WFI` | Wastes power, defeats DVFS/idle-power strategies | Sleep with `WFI` between dispatch passes |
| Not resetting `VAL` before reconfiguring | Stale count causes first interrupt to fire early or late | Always write `SysTick->VAL = 0` when reconfiguring `LOAD`/`CTRL` |

## 8. Key Takeaways

1. **SysTick is core-integrated, not a vendor peripheral** — its register map (`CTRL`/`LOAD`/`VAL`/`CALIB` at `0xE000E010`) is identical across every Cortex-M vendor, making it a favorite "explain the registers" interview target.
2. **The 24-bit reload limit is a real constraint** — always validate that `SystemCoreClock / desired_rate - 1` fits in 24 bits; silent overflow produces a dramatically wrong tick rate.
3. **Level checks (`% N == 0`) on a monotonically-updating counter are a classic bug** — always use edge detection via subtraction (`now - last >= period`) which is also naturally robust to missed/late checks.
4. **`volatile` is necessary but not sufficient for correctness** — it prevents the compiler from optimizing away memory access, but does *not* provide atomicity for multi-word reads; those need explicit interrupt-disable or retry-on-rollover patterns.
5. **A single hardware SysTick can drive many independent periodic tasks** via a software timer-list dispatched from the main loop — this is the standard way to build deterministic 10ms/100ms/1000ms multi-rate event loops on hardware with only one systick, directly analogous to the 100ms deterministic event-loop work elsewhere in this portfolio.
6. **Keep the ISR itself minimal** (`ticks++` or a flag set) and defer real work to the main loop or a soft-timer dispatcher — this preserves tight, predictable interrupt latency for higher-priority time-critical ISRs sharing the same NVIC.