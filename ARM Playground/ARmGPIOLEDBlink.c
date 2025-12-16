# GPIO LED Blink — Memory-Mapped I/O Control

## 1. Concept Map

```
CPU Core
   |
   |  (Load/Store instructions, NOT special I/O instructions --
   |   ARM has no IN/OUT like x86; everything is memory-mapped)
   v
System Bus (AHB/APB)
   |
   +-- 0x40020000  GPIOA_MODER   (mode: input/output/AF/analog)
   +-- 0x40020004  GPIOA_OTYPER  (output type: push-pull/open-drain)
   +-- 0x40020008  GPIOA_OSPEEDR (slew rate)
   +-- 0x4002000C  GPIOA_PUPDR   (pull-up/pull-down)
   +-- 0x40020010  GPIOA_IDR     (input data, read-only)
   +-- 0x40020014  GPIOA_ODR     (output data, read-write, NON-atomic)
   +-- 0x40020018  GPIOA_BSRR    (bit set/reset, WRITE-ONLY, ATOMIC)
   +-- 0x4002001C  GPIOA_LCKR    (config lock)
```

## 2. Baseline Version (as given) — annotated

```c
#include <stdint.h>

#define GPIO_BASE       0x40020000U
#define GPIO_MODER      (*(volatile uint32_t *)(GPIO_BASE + 0x00))
#define GPIO_ODR        (*(volatile uint32_t *)(GPIO_BASE + 0x14))

#define LED_PIN         5U

void delay(void)
{
    /* volatile on the loop counter is mandatory here: without it,
     * an empty loop body is dead code to the optimizer and GCC at
     * -O1+ will delete the entire loop, turning a "blink" into a
     * permanently-lit or dark LED. This is a *software* busy-wait,
     * not cycle-accurate -- see Section 5 for SysTick-based timing. */
    for (volatile int i = 0; i < 100000; i++);
}

int main(void)
{
    /* Configure GPIO pin as output.
     * BUG (left in intentionally to discuss): MODER is a 2-bit-per-pin
     * field (00=input, 01=output, 10=AF, 11=analog). Pin 5's field is
     * bits [11:10]. `|= (1 << 10)` sets bit10=1, bit11=0 -> value 01
     * (output) ONLY IF bit11 was already 0 at reset. Relying on reset
     * state is fragile -- see the hardened version in Section 3. */
    GPIO_MODER |= (1U << (LED_PIN * 2));

    while (1)
    {
        /* Read-modify-write toggle: read ODR, flip bit 5, write ODR
         * back. This is NOT atomic -- an interrupt between the read
         * and write that also touches ODR will corrupt the other
         * bits' intended state (see Section 4). */
        GPIO_ODR ^= (1U << LED_PIN);

        delay();
    }
}
```

## 3. Hardened, Production-Style Version (STM32F4xx-class MCU)

```c
/* ============================================================
 * gpio_led_blink.c
 * Bare-metal GPIO LED blink via direct memory-mapped registers.
 * Target: STM32F4-series-style peripheral map (illustrative --
 * confirm exact offsets/bit widths against your MCU's reference
 * manual, e.g. RM0090 for STM32F4, before deploying to silicon).
 * ============================================================ */

#include <stdint.h>

/* ---------------------------------------------------------------
 * RCC (Reset and Clock Control) -- MUST be configured BEFORE
 * touching GPIOA registers. On reset, most peripheral clocks are
 * gated off to save power; writing to an unclocked peripheral's
 * registers is a classic "why does my GPIO do nothing" bug and,
 * on some cores, can trigger a bus fault.
 * --------------------------------------------------------------- */
#define RCC_BASE            0x40023800U
#define RCC_AHB1ENR         (*(volatile uint32_t *)(RCC_BASE + 0x30))
#define RCC_AHB1ENR_GPIOAEN (1U << 0)

/* ---------------------------------------------------------------
 * GPIOA register block, modeled as a struct overlay.
 * This is the idiomatic CMSIS-style pattern: one struct definition,
 * one pointer cast, instead of a #define per register+offset. It
 * scales cleanly to GPIOB/C/D/... by just re-basing the pointer.
 * --------------------------------------------------------------- */
typedef struct {
    volatile uint32_t MODER;    /* 0x00 : mode register            */
    volatile uint32_t OTYPER;   /* 0x04 : output type register     */
    volatile uint32_t OSPEEDR;  /* 0x08 : output speed register    */
    volatile uint32_t PUPDR;    /* 0x0C : pull-up/pull-down reg    */
    volatile uint32_t IDR;      /* 0x10 : input data register (RO) */
    volatile uint32_t ODR;      /* 0x14 : output data register     */
    volatile uint32_t BSRR;     /* 0x18 : bit set/reset (WO, atomic)*/
    volatile uint32_t LCKR;     /* 0x1C : configuration lock reg   */
    volatile uint32_t AFR[2];   /* 0x20/0x24 : alternate function  */
} GPIO_TypeDef;

#define GPIOA_BASE          0x40020000U
#define GPIOA               ((GPIO_TypeDef *)GPIOA_BASE)

#define LED_PIN             5U
#define LED_PIN_MODE_MASK   (0x3U << (LED_PIN * 2U))   /* 2 bits/pin field */
#define LED_PIN_MODE_OUTPUT (0x1U << (LED_PIN * 2U))   /* 01 = general output */

/* BSRR encoding: writing bit[N] sets pin N; writing bit[N+16] resets
 * pin N. This is a WRITE-ONLY, hardware-atomic operation -- the
 * peripheral itself resolves the bit, so no read-modify-write race
 * is possible, unlike ODR. */
#define LED_SET()    (GPIOA->BSRR = (1U << LED_PIN))
#define LED_RESET()  (GPIOA->BSRR = (1U << (LED_PIN + 16U)))
#define LED_TOGGLE() (GPIOA->ODR ^= (1U << LED_PIN))  /* fine for main-loop-only access */

/* ---------------------------------------------------------------
 * Busy-wait delay. `volatile` on the loop variable prevents the
 * optimizer from eliminating the empty loop body (see Section 2).
 * Duration is core-clock-dependent and NOT calibrated -- treat as
 * illustrative. Section 5 replaces this with a SysTick-based delay
 * for deterministic, clock-independent timing.
 * --------------------------------------------------------------- */
static void delay(volatile uint32_t count)
{
    while (count--)
    {
        __asm volatile ("nop");  /* prevents loop collapse even at -O2,
                                     and gives the compiler an explicit
                                     side-effecting instruction per
                                     iteration instead of relying solely
                                     on the volatile qualifier */
    }
}

int main(void)
{
    /* Step 1: Enable the GPIOA peripheral clock.
     * Read-modify-write via |= is safe here because this executes
     * once at startup before interrupts are enabled -- no
     * concurrent access is possible yet. */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

    /* Step 2: Configure PA5 as general-purpose output.
     * Correct pattern: CLEAR the 2-bit field first, THEN set the
     * desired mode bits. Relying on reset-default zeros (as the
     * baseline version implicitly does) is fragile if this GPIO
     * was previously configured by bootloader/other firmware. */
    GPIOA->MODER &= ~LED_PIN_MODE_MASK;   /* clear bits [11:10] -> 00 */
    GPIOA->MODER |=  LED_PIN_MODE_OUTPUT; /* set bits [11:10]  -> 01 */

    /* Optional but recommended for real hardware: explicitly select
     * push-pull output type (OTYPER bit = 0) and pull-up/down = none
     * (PUPDR field = 00). Reset defaults are usually already this,
     * but explicit configuration documents intent and survives
     * silicon variants where reset state differs. */
    GPIOA->OTYPER &= ~(1U << LED_PIN);              /* push-pull      */
    GPIOA->PUPDR  &= ~(0x3U << (LED_PIN * 2U));      /* no pull-up/dn  */

    while (1)
    {
        LED_TOGGLE();
        delay(1000000U);
    }
}
```

## 4. The Read-Modify-Write Hazard (ODR) vs. Atomic Set/Reset (BSRR)

```c
/* ============================================================
 * Why ODR toggling is dangerous in interrupt-driven systems,
 * and why BSRR exists in real silicon.
 * ============================================================ */

/* DANGEROUS pattern if an ISR also touches ODR: */
void main_loop_bad(void)
{
    GPIOA->ODR ^= (1U << LED_PIN);
    /*   Instruction sequence generated:
     *     LDR  R0, [ODR_ADDR]      ; read
     *     EOR  R0, R0, #(1<<5)     ; modify
     *     STR  R0, [ODR_ADDR]      ; write
     *
     *   If an interrupt fires between LDR and STR, and the ISR
     *   also writes ODR (e.g. to drive a different pin), the ISR's
     *   write is LOST the moment main_loop_bad()'s stale R0 is
     *   stored back -- classic lost-update race, identical in
     *   spirit to the false-sharing/lock-contention races addressed
     *   in the cache-coherency work, just at register-file scope
     *   instead of cache-line scope.
     */
}

/* SAFE pattern: BSRR is hardware-atomic by construction -- the
 * write itself IS the set-or-clear operation, with no read phase
 * to race against. This is why production STM32 GPIO drivers (and
 * the CMSIS HAL) use BSRR, never ODR |=/&=/ ^=, for ISR-shared pins. */
void main_loop_safe(void)
{
    GPIOA->BSRR = (1U << (LED_PIN + 16U));  /* atomically clear pin */
    GPIOA->BSRR = (1U << LED_PIN);          /* atomically set pin   */
}
```

## 5. Deterministic Delay via SysTick (replacing the busy-wait)

```c
/* ============================================================
 * A busy-wait loop's duration depends on core clock speed, cache
 * state, and pipeline effects -- it is NOT portable or precise.
 * Production firmware uses a hardware timer. SysTick is present
 * on every Cortex-M core, making it the natural baseline choice.
 * ============================================================ */

#define SYSTICK_BASE    0xE000E010U
typedef struct {
    volatile uint32_t CTRL;
    volatile uint32_t LOAD;
    volatile uint32_t VAL;
    volatile uint32_t CALIB;
} SysTick_TypeDef;
#define SysTick ((SysTick_TypeDef *)SYSTICK_BASE)

static volatile uint32_t g_tick_ms = 0;

void SysTick_Handler(void)
{
    g_tick_ms++;   /* incremented every 1ms once configured below */
}

static void systick_init(uint32_t core_clock_hz)
{
    SysTick->LOAD = (core_clock_hz / 1000U) - 1U;  /* 1ms reload value */
    SysTick->VAL  = 0U;
    SysTick->CTRL = (1U << 2) |  /* CLKSOURCE = processor clock */
                    (1U << 1) |  /* TICKINT   = enable exception */
                    (1U << 0);   /* ENABLE                       */
}

static void delay_ms(uint32_t ms)
{
    uint32_t start = g_tick_ms;
    while ((g_tick_ms - start) < ms) {
        __asm volatile ("wfi");  /* sleep until next interrupt --
                                     saves power vs. spinning, same
                                     philosophy as the +18% perf/watt
                                     DVFS/idle-state work */
    }
}

int main(void)
{
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER &= ~LED_PIN_MODE_MASK;
    GPIOA->MODER |=  LED_PIN_MODE_OUTPUT;

    systick_init(16000000U);  /* e.g. 16MHz HSI default clock */

    while (1)
    {
        LED_TOGGLE();
        delay_ms(500);   /* precise, clock-referenced 500ms period,
                             independent of compiler optimization
                             level or CPU pipeline behavior */
    }
}
```

## 6. Why `volatile` Is Non-Negotiable Here

| Without `volatile` | With `volatile` |
|---|---|
| Compiler may cache the register value in a CPU register across accesses, assuming memory doesn't change "on its own" | Every access compiles to an actual load/store instruction touching the real address |
| `GPIO_ODR ^= (1<<5)` in a loop can be **hoisted out entirely** at `-O2` if the compiler proves no visible effect on program state (it doesn't know memory-mapped I/O has externally-visible side effects) | Toggle instruction sequence is preserved exactly as written, every iteration |
| Reads/writes may be **reordered** relative to other memory operations | Access ordering relative to other volatile accesses is preserved (though *not* relative to non-volatile memory — that still needs `DMB`/`DSB` on multi-core/DMA systems, per the CMO work) |
| Empty delay loops get deleted as dead code | Loop body with observable (volatile) side effect is retained |

`volatile` guarantees **access preservation and ordering among volatile operations**, but on multi-core or DMA-coherent SoCs it is **not** a substitute for memory barriers — that distinction is exactly what the earlier `DC CVAC`/`DSB`/`ISB` cache-maintenance work addresses for non-CPU bus masters; GPIO toggling from a single core in a simple loop is the case where `volatile` alone is sufficient.

## 7. Summary

| Register | Purpose | Access pattern | Atomicity |
|---|---|---|---|
| `RCC_AHB1ENR` | Gate peripheral clock | RMW `\|=` at startup only | Safe (no concurrency yet) |
| `GPIOx_MODER` | Pin mode (2 bits/pin) | Clear-then-set RMW | Safe if single-threaded config phase |
| `GPIOx_ODR` | Output data | RMW toggle/set/clear | **Unsafe** if shared with ISR |
| `GPIOx_BSRR` | Atomic set/reset | Write-only, self-atomic | **Safe** under interrupts, preferred for shared pins |
| `SysTick` | Deterministic timing | ISR-incremented counter + `WFI` wait | Standard low-power delay primitive |

This mirrors the same layered discipline seen across the earlier TrustZone/TZASC and cache-maintenance work: **Tier 1** (raw register access, analogous to basic SMC calls), **Tier 2** (correct read-modify-write sequencing, analogous to SCR NS-bit switching), and **Tier 3** (hardware-atomic primitives like BSRR, analogous to TZASC's hardware-enforced partitioning) — moving progressively from "it compiles" to "it's correct under concurrency and interrupts," which is the bar production embedded firmware must clear.