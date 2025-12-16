# External GPIO Interrupt — NVIC-Driven ISR Replacing Polling

## 1. Concept Map

```
Button (mechanical switch, active-LOW to GND via PUPDR pull-up)
   |
GPIO (IDR bit latched by hardware; EXTI line multiplexed onto the pin
   |   via SYSCFG_EXTICR — each EXTI line maps to ONE GPIO port at a time)
   |
EXTI (Edge Detect Circuit -- rising/falling/both, independent of CPU;
   |   sets Pending Register PR bit the instant the edge occurs,
   |   completely asynchronously to whatever the CPU is doing)
   |
NVIC (Nested Vectored Interrupt Controller -- routes the EXTI line's
   |   IRQ number to the CPU, applies priority, handles preemption)
   |
ISR  (EXTIx_IRQHandler -- vector-table entry, runs at configured priority)
   |
LED (toggled via GPIO_BSRR -- atomic set/reset, no RMW hazard)
```

**Why this beats polling:** polling burns 100% CPU checking `button_pressed()` in a tight loop, has latency bounded only by loop speed (and by whatever else is competing for CPU time), and prevents the core from ever sleeping. Interrupt-driven GPIO lets the core execute `WFI` and consume near-zero power between events, while guaranteeing sub-microsecond reaction time because the EXTI edge-detect hardware is asynchronous to program flow — this is the same discipline used for the `<100us` interrupt latency budgets and `+18%` performance/watt work elsewhere in this portfolio.

## 2. Register-Level Definitions

```c
/* ============================================================
 * STM32F4-class register map (illustrative addresses -- exact
 * offsets vary slightly by family, but the STRUCTURE and the
 * hazards below are universal across ARM Cortex-M EXTI designs).
 * ============================================================ */

#include <stdint.h>
#include <stdbool.h>

/* --- GPIO (Port A used here for button + LED) --- */
typedef struct {
    volatile uint32_t MODER;    /* 0x00: mode (in/out/alt/analog), 2b/pin */
    volatile uint32_t OTYPER;   /* 0x04: output type (push-pull/open-drain) */
    volatile uint32_t PUPDR;    /* 0x0C: pull-up/pull-down, 2b/pin */
    volatile uint32_t IDR;      /* 0x10: Input Data Register (read-only,
                                    hardware-latched -- no RMW hazard) */
    volatile uint32_t ODR;      /* 0x14: Output Data Register (RMW hazard
                                    on non-atomic access -- avoid, use BSRR) */
    volatile uint32_t BSRR;     /* 0x18: Bit Set/Reset Register (atomic) */
} GPIO_TypeDef;

#define GPIOA   ((GPIO_TypeDef *)0x40020000U)

/* --- SYSCFG: multiplexes EXTI lines 0-15 onto a GPIO port --- */
typedef struct {
    volatile uint32_t MEMRMP;
    volatile uint32_t PMC;
    volatile uint32_t EXTICR[4];  /* EXTICR[0] covers EXTI0-3,
                                      EXTICR[1] covers EXTI4-7, etc.
                                      Each EXTI line is a 4-bit field
                                      selecting which port (A=0000,
                                      B=0001, ...) drives that line. */
} SYSCFG_TypeDef;

#define SYSCFG  ((SYSCFG_TypeDef *)0x40013800U)

/* --- EXTI: edge-detect + pending-flag hardware --- */
typedef struct {
    volatile uint32_t IMR;    /* 0x00: Interrupt Mask Register
                                  (1 = NOT masked, i.e. enabled) */
    volatile uint32_t EMR;    /* 0x04: Event Mask Register (wake w/o ISR) */
    volatile uint32_t RTSR;   /* 0x08: Rising Trigger Selection */
    volatile uint32_t FTSR;   /* 0x0C: Falling Trigger Selection */
    volatile uint32_t SWIER;  /* 0x10: Software Interrupt Event (debug) */
    volatile uint32_t PR;     /* 0x14: Pending Register -- WRITE 1 TO CLEAR,
                                  writing 0 does nothing (this is NOT a
                                  normal RMW register -- critical detail) */
} EXTI_TypeDef;

#define EXTI    ((EXTI_TypeDef *)0x40013C00U)

/* --- NVIC: core-integrated, fixed address on every Cortex-M --- */
typedef struct {
    volatile uint32_t ISER[8];   /* Interrupt Set-Enable */
    uint32_t RESERVED0[24];
    volatile uint32_t ICER[8];   /* Interrupt Clear-Enable */
    uint32_t RESERVED1[24];
    volatile uint32_t ISPR[8];   /* Interrupt Set-Pending */
    uint32_t RESERVED2[24];
    volatile uint32_t ICPR[8];   /* Interrupt Clear-Pending */
    uint32_t RESERVED3[24];
    /* ... IABR, IPR (priority) omitted for brevity ... */
    volatile uint8_t  IP[240];   /* Interrupt Priority (byte-addressable) */
} NVIC_TypeDef;

#define NVIC        ((NVIC_TypeDef *)0xE000E100U)
#define EXTI0_IRQn  6           /* vendor-specific IRQ number for EXTI0 */

#define BUTTON_PIN_POS   0U     /* PA0 */
#define LED_PIN_POS      5U     /* PA5 */
#define BUTTON_PIN       (1U << BUTTON_PIN_POS)
#define LED_PIN          (1U << LED_PIN_POS)
```

## 3. Initialization — GPIO → SYSCFG → EXTI → NVIC

```c
/* ============================================================
 * This sequence mirrors the diagram top-to-bottom: configure
 * the GPIO electrical properties first, wire the EXTI line to
 * that GPIO port, arm the edge trigger, THEN enable at the NVIC
 * last -- enabling NVIC before the source is fully configured
 * risks a spurious/garbage interrupt firing on an unconfigured line.
 * ============================================================ */

static void button_led_gpio_exti_init(void)
{
    /* --- 1. GPIO electrical config --- */

    /* PA0 = input (MODER = 00), PA5 = output (MODER = 01) */
    GPIOA->MODER &= ~(0x3U << (BUTTON_PIN_POS * 2U));  /* clear -> input */
    GPIOA->MODER &= ~(0x3U << (LED_PIN_POS    * 2U));
    GPIOA->MODER |=  (0x1U << (LED_PIN_POS    * 2U));  /* set -> output */

    /* Mandatory pull-up on the button input: without this, PA0
     * floats when the switch is open, and stray capacitive
     * coupling / EMI will trigger phantom edge interrupts.
     * Active-LOW convention: pressed = pulled to GND = falling edge. */
    GPIOA->PUPDR &= ~(0x3U << (BUTTON_PIN_POS * 2U));
    GPIOA->PUPDR |=  (0x1U << (BUTTON_PIN_POS * 2U));  /* 01 = pull-up */

    /* --- 2. SYSCFG: route EXTI0 to GPIO port A --- */

    SYSCFG->EXTICR[0] &= ~(0xFU << 0U);   /* EXTI0 field = bits [3:0] of
                                              EXTICR[0]; clear then set,
                                              same clear-then-set discipline
                                              as MODER field writes */
    SYSCFG->EXTICR[0] |=  (0x0U << 0U);   /* 0000 = Port A */

    /* --- 3. EXTI: falling-edge trigger, unmask, ensure no stale flag --- */

    EXTI->FTSR |= BUTTON_PIN;   /* enable falling-edge detection on line 0 */
    EXTI->RTSR &= ~BUTTON_PIN;  /* explicitly disable rising-edge (defensive --
                                    don't assume reset state) */
    EXTI->IMR  |= BUTTON_PIN;   /* unmask line 0 -> generates CPU interrupt
                                    (as opposed to EMR, which would only
                                    wake the core via the Event mechanism
                                    without entering an ISR) */

    EXTI->PR = BUTTON_PIN;      /* clear any stale pending bit accumulated
                                    during power-up/config BEFORE enabling
                                    at the NVIC -- prevents an immediate
                                    spurious interrupt the instant it's
                                    unmasked */

    /* --- 4. NVIC: priority THEN enable, in that order --- */

    NVIC->IP[EXTI0_IRQn] = (2U << 4U);   /* priority group encoding varies;
                                             set explicitly rather than
                                             trusting reset default, so this
                                             ISR's priority relative to
                                             SysTick/other EXTI lines is a
                                             deliberate choice, not an
                                             accident */

    NVIC->ISER[EXTI0_IRQn / 32U] |= (1U << (EXTI0_IRQn % 32U));
    /* ISER is a SET-enable register: writing 1 enables that IRQ,
     * writing 0 is a no-op (mirrors EXTI->PR's "write-1" semantics --
     * NVIC has a SEPARATE ICER register for clearing/disabling,
     * you cannot disable by writing 0 to ISER) */
}
```

## 4. The ISR — Flag Check, Correct Clear, Atomic LED Toggle

```c
/* ============================================================
 * volatile shared state between ISR and main() -- necessary
 * because main() reads it without any memory barrier of its own.
 * ============================================================ */
static volatile uint32_t g_button_press_count = 0;
static volatile bool     g_button_event_pending = false;

void EXTI0_IRQHandler(void)
{
    /* --- Step 1: check the flag for THIS specific line --- */
    if (EXTI->PR & BUTTON_PIN)
    {
        /* --- Step 2: clear the pending bit FIRST ---
         * EXTI->PR is write-1-to-clear: writing back the SAME bit
         * pattern you read clears only that bit and leaves every
         * other line's pending status untouched. This is NOT a
         * normal read-modify-write register -- do NOT do
         * `EXTI->PR &= ~BUTTON_PIN` (that's a read-AND-write which
         * works here too by coincidence of hardware design, but the
         * canonical/correct idiom for write-1-to-clear registers is
         * a plain store of the bit you want cleared: `EXTI->PR = BUTTON_PIN;`
         * -- never `EXTI->PR = 0xFFFFFFFF` which would silently
         * clear and lose OTHER lines' pending interrupts that may
         * have been latched between your read and this write. */
        EXTI->PR = BUTTON_PIN;

        /* --- Step 3: clear BEFORE handling the event, not after ---
         * If the flag is cleared AFTER the LED toggle below and the
         * button bounces again (or a second real press occurs)
         * during that window, the EDGE-DETECT HARDWARE still latches
         * the new pending bit correctly regardless of ordering --
         * BUT clearing first minimizes the ISR's "blind window" and
         * is the standard defensive idiom: it ensures a fresh edge
         * arriving mid-ISR is not silently swallowed by whatever
         * work happens next, since PR already reflects only the
         * NEW edge by the time we act. */

        /* --- Step 4: atomic LED toggle via BSRR, not ODR ^= --- */
        if (GPIOA->ODR & LED_PIN)
        {
            GPIOA->BSRR = (LED_PIN << 16U);  /* upper 16 bits = RESET */
        }
        else
        {
            GPIOA->BSRR = LED_PIN;           /* lower 16 bits = SET */
        }
        /* Using ODR ^= LED_PIN directly (as in the naive example)
         * is a read-modify-write: if ANY other code (another ISR at
         * higher priority, or main() writing a different ODR bit
         * concurrently) touches ODR between the read and the write,
         * the toggle can corrupt an unrelated pin's state. BSRR's
         * set/reset-by-bit-mask design makes each half hardware-atomic
         * with zero RMW hazard -- same discipline as the GPIO BSRR
         * work covered for bare-metal Cortex-M GPIO output elsewhere
         * in this portfolio. */

        g_button_press_count++;
        g_button_event_pending = true;   /* defer any heavier work
                                             (debounce confirmation,
                                             logging, state-machine
                                             update) to main(), keeping
                                             this ISR body minimal and
                                             bounded in execution time */
    }

    /* If NO recognized bit is set (shouldn't happen since NVIC only
     * routed line 0 here), the ISR simply returns -- but note: if
     * a shared vector like EXTI9_5_IRQHandler covers MULTIPLE lines,
     * every line sharing that vector MUST be checked and cleared
     * individually, or an unhandled line's pending bit stays set
     * and the ISR re-fires immediately upon return (interrupt storm). */
}
```

## 5. Software Debounce Layered on Top (ISR Sets Flag, Main Confirms)

```c
/* ============================================================
 * Raw mechanical bounce still produces MULTIPLE falling edges
 * per physical press. The ISR above faithfully reports every
 * one of them. Debounce confirmation is deliberately done in
 * main()/a lower-priority context, using the SysTick millisecond
 * counter from the companion SysTick implementation, to keep the
 * EXTI ISR itself short. */

extern volatile uint32_t g_ticks;   /* from SysTick_Handler, 1ms tick */

#define DEBOUNCE_MS   20U

static uint32_t s_last_confirmed_ms = 0;

static void poll_button_event(void)
{
    if (g_button_event_pending)
    {
        g_button_event_pending = false;   /* consume the flag */

        uint32_t now = g_ticks;
        if ((now - s_last_confirmed_ms) >= DEBOUNCE_MS)
        {
            s_last_confirmed_ms = now;
            /* This is a CONFIRMED, debounced press -- safe to act on
             * for anything beyond the immediate LED toggle already
             * done in the ISR (e.g. incrementing a UI counter,
             * logging, sending a message on a queue). */
        }
        /* else: bounce artifact within 20ms of the last confirmed
         * edge -- ignored. The LED already toggled in the ISR for
         * every raw edge; if a debounced, single-toggle-per-press
         * LED response is required instead, move the BSRR toggle
         * itself out of the ISR and into this confirmed branch --
         * a deliberate design trade-off between "instant visual
         * feedback on every raw edge" vs. "strictly one toggle per
         * physical press." */
    }
}

int main(void)
{
    button_led_gpio_exti_init();
    /* ... SysTick_Config_manual(SystemCoreClock / 1000U) from the
       companion SysTick implementation ... */

    while (1)
    {
        poll_button_event();
        __asm volatile ("wfi");   /* sleep until EITHER SysTick or
                                      EXTI0 wakes the core -- CPU usage
                                      approaches 0% between events,
                                      compared to 100% for the naive
                                      `while(1) if(button_pressed())`
                                      polling loop in the prompt */
    }
}
```

## 6. Polling vs. Interrupt-Driven — Measured Trade-offs

| Aspect | Polling (`while(1) if(button_pressed())`) | Interrupt-Driven (EXTI + NVIC) |
|---|---|---|
| CPU usage while idle | 100% — core spins continuously | ~0% — core sleeps in `WFI` between events |
| Reaction latency | Bounded by loop iteration time; degrades if loop does other work | Sub-microsecond — edge-detect hardware is asynchronous to program flow |
| Power | High — no opportunity for clock/power gating | Low — enables deepest sleep modes between edges |
| Missed events | Possible if loop is busy elsewhere when press occurs | Edge is latched in hardware `PR` regardless of CPU state, until explicitly cleared |
| Determinism | Unpredictable — coupled to whatever else runs in the loop | Deterministic — NVIC priority guarantees response ordering vs. other ISRs |
| Scalability | Every extra input needs its own polled check, linearly increasing loop cost | Each new input is a new EXTI line/ISR, no cost added to existing paths |

## 7. Common Pitfalls Checklist

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgetting to clear `EXTI->PR` in the ISR | ISR fires once, returns, then **immediately fires again forever** (interrupt storm/hang) — pending bit never cleared | Always `EXTI->PR = <line_bit>;` before returning |
| Clearing with `EXTI->PR = 0xFFFFFFFF` | Silently clears and loses pending bits for *other*, unrelated EXTI lines latched concurrently | Clear only the specific bit(s) actually being handled |
| Using `ODR ^= LED_PIN` inside the ISR | Rare bit corruption on other ODR pins if preempted mid-RMW by a higher-priority ISR | Use `BSRR` set/reset halves — hardware-atomic |
| Not configuring `PUPDR` pull-up/pull-down | Floating input triggers phantom edges from EMI/noise, "ghost presses" | Mandatory pull configuration matching active-HIGH/LOW convention |
| Enabling NVIC before EXTI/SYSCFG fully configured | Spurious interrupt fires on an unconfigured/garbage line at startup | Configure GPIO → SYSCFG → EXTI trigger/clear-stale-flag → NVIC, in that order |
| Shared vector (e.g. `EXTI9_5_IRQHandler`) not checking every line it covers | One line's real event is missed, or an unhandled pending bit causes immediate re-entry | Check and clear each line individually inside a shared-vector ISR |
| Heavy work (printf, long computation) directly in the ISR | Blocks other higher-priority ISRs, breaks `<100us` latency budgets elsewhere in the system | Set a flag/counter in the ISR; defer real work to main loop, as shown in `poll_button_event()` |
| Debounce state entirely inside the ISR | ISR execution time grows and becomes press-pattern-dependent, hurting determinism | Keep ISR minimal; do debounce timing confirmation in main() against the SysTick reference |

## 8. Key Takeaways

1. **The NVIC + EXTI pipeline replaces CPU-bound polling with hardware-latched, asynchronous edge detection** — the `PR` (pending) register captures the event the instant it happens, completely independent of what the CPU is currently executing, which is what makes sub-microsecond reaction times possible.
2. **`EXTI->PR` is write-1-to-clear, not a normal RMW register** — always store back exactly the bit(s) being handled; never blanket-clear with `0xFFFFFFFF`, and never forget to clear at all (the single most common bug, causing an infinite re-fire).
3. **GPIO output toggling inside an ISR must use `BSRR`, not `ODR ^=`** — `BSRR`'s dedicated set/reset halves are hardware-atomic, eliminating the RMW hazard that a shared `ODR` register otherwise has under interrupt preemption.
4. **Initialization order matters**: GPIO electrical config → SYSCFG line routing → EXTI trigger config and stale-flag clear → NVIC priority and enable, strictly in that order, to avoid spurious interrupts on an incompletely configured line.
5. **Keep the ISR minimal** — clear the flag, do the truly time-critical action (LED toggle here), set a deferred-work flag, and return; push debounce confirmation and any heavier logic into the main loop, preserving tight and predictable interrupt latency for every other ISR sharing the NVIC.
6. **`WFI` in the main loop is the payoff** — once the reactive path is fully interrupt-driven, the core can sleep between real events, directly realizing the same power-efficiency discipline behind the `+18%` performance/watt and DVFS-tuning work elsewhere in this portfolio.