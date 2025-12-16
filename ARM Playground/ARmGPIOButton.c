# GPIO Input/Button — Reading External State via Memory-Mapped Registers

## 1. Concept Map

```
Button (mechanical switch)
   |
   |  Physical bounce: contact makes/breaks multiple times
   |  over 1-20ms before settling -- NOT an instant clean edge
   v
GPIO Input Pin (PA0, e.g.)
   |
   |  Needs a defined electrical level when button is OPEN:
   |  external or internal PULL-UP/PULL-DOWN resistor required,
   |  otherwise the pin floats and reads noise
   v
GPIO_IDR (Input Data Register) -- read-only, hardware-latched
   |
   |  CPU polls or is interrupted (EXTI) on level/edge
   v
CPU decision logic (with debounce filtering)
   |
   v
GPIO_ODR / BSRR (Output Data Register) -- write
   |
   v
LED
```

## 2. Baseline Version (as given) — annotated

```c
#include <stdint.h>

#define GPIO_BASE    0x40020000U
#define GPIO_MODER   (*(volatile uint32_t *)(GPIO_BASE + 0x00))
#define GPIO_PUPDR   (*(volatile uint32_t *)(GPIO_BASE + 0x0C))
#define GPIO_IDR     (*(volatile uint32_t *)(GPIO_BASE + 0x10))
#define GPIO_ODR     (*(volatile uint32_t *)(GPIO_BASE + 0x14))

#define BUTTON_PIN   0U
#define LED_PIN      5U

int main(void)
{
    /* BUG (intentional, discussed below): MODER for BUTTON_PIN is
     * never explicitly cleared to 00 (input mode). On many MCUs the
     * reset default IS 00 for most pins, so this "works" by luck --
     * fragile the same way the LED-blink baseline was fragile. */

    while (1)
    {
        /* Polling read: IDR bit reflects the CURRENT electrical
         * level of the pin, sampled continuously by hardware --
         * unlike ODR, there's no read-modify-write hazard here
         * since we never write to IDR (it's read-only silicon). */
        if (GPIO_IDR & (1U << BUTTON_PIN))
        {
            GPIO_ODR |= (1U << LED_PIN);
        }
        else
        {
            GPIO_ODR &= ~(1U << LED_PIN);
        }
        /* MISSING: no debounce. A mechanical button bouncing during
         * a state transition will cause GPIO_ODR to be hammered with
         * spurious toggles for several milliseconds -- usually
         * invisible on an LED but catastrophic if this input drove
         * a counter, state machine, or safety-critical latch. */
    }
}
```

## 3. Hardened Version — Correct Input Configuration + Pull Resistor

```c
/* ============================================================
 * gpio_button_input.c
 * Bare-metal GPIO input polling with correct electrical
 * configuration. Target: STM32F4xx-style peripheral map
 * (illustrative -- confirm against your MCU's reference manual).
 * ============================================================ */

#include <stdint.h>

#define RCC_BASE            0x40023800U
#define RCC_AHB1ENR         (*(volatile uint32_t *)(RCC_BASE + 0x30))
#define RCC_AHB1ENR_GPIOAEN (1U << 0)

typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;     /* input data register -- READ ONLY */
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

#define GPIOA_BASE   0x40020000U
#define GPIOA        ((GPIO_TypeDef *)GPIOA_BASE)

#define BUTTON_PIN   0U
#define LED_PIN      5U

/* MODER field values (2 bits per pin) */
#define MODE_INPUT   0x0U
#define MODE_OUTPUT  0x1U

/* PUPDR field values (2 bits per pin) */
#define PUPD_NONE      0x0U
#define PUPD_PULLUP    0x1U
#define PUPD_PULLDOWN  0x2U

static void gpio_set_mode(GPIO_TypeDef *port, uint32_t pin, uint32_t mode)
{
    port->MODER &= ~(0x3U << (pin * 2U));
    port->MODER |=  (mode << (pin * 2U));
}

static void gpio_set_pupd(GPIO_TypeDef *port, uint32_t pin, uint32_t pupd)
{
    port->PUPDR &= ~(0x3U << (pin * 2U));
    port->PUPDR |=  (pupd << (pin * 2U));
}

static void delay(volatile uint32_t count)
{
    while (count--) { __asm volatile ("nop"); }
}

int main(void)
{
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

    /* --- Button input configuration --- */
    gpio_set_mode(GPIOA, BUTTON_PIN, MODE_INPUT);

    /* CRITICAL: pull resistor decision depends on wiring topology.
     *
     *   Active-LOW button (button connects pin to GND when pressed):
     *     -> use internal PULL-UP. Idle state reads 1, pressed reads 0.
     *     -> This is the MORE COMMON convention in production designs
     *        because it survives a short to ground more safely than
     *        a short to VDD, and matches many dev-board reference
     *        button circuits (e.g. STM32 Discovery/Nucleo B1 button
     *        is actually active-HIGH with external pull-down, but
     *        many custom PCBs use active-LOW + internal pull-up).
     *
     *   Active-HIGH button (button connects pin to VDD when pressed):
     *     -> use internal PULL-DOWN. Idle state reads 0, pressed reads 1.
     *
     * Choosing NONE and leaving the pin floating with no external
     * resistor is the #1 cause of "random ghost button presses" bug
     * reports -- an undriven CMOS input pin acts as a tiny antenna
     * and will oscillate on EMI/capacitive coupling. */
    gpio_set_pupd(GPIOA, BUTTON_PIN, PUPD_PULLUP);

    /* --- LED output configuration --- */
    gpio_set_mode(GPIOA, LED_PIN, MODE_OUTPUT);

    while (1)
    {
        /* Active-LOW read: pin reads 0 when PRESSED (pulled to GND),
         * 1 when RELEASED (pulled up internally). Invert the sense
         * here so the rest of the logic reads naturally as "pressed". */
        int pressed = ((GPIOA->IDR & (1U << BUTTON_PIN)) == 0U);

        if (pressed)
        {
            GPIOA->BSRR = (1U << LED_PIN);              /* atomic set   */
        }
        else
        {
            GPIOA->BSRR = (1U << (LED_PIN + 16U));       /* atomic clear */
        }

        delay(1000U);  /* crude polling interval, see Section 4 for
                           why this ALSO functions as a debounce gate */
    }
}
```

## 4. The Real Problem: Mechanical Bounce and Software Debouncing

```c
/* ============================================================
 * Why raw IDR polling is unreliable across a button TRANSITION,
 * and how to filter it deterministically.
 *
 * Oscilloscope reality of a "single" button press:
 *
 *   IDR bit:  1__0_1_0___0_0_1_0_______0__________1
 *             ^-- idle   ^-- mechanical bounce      ^-- idle
 *                        (typically 1-20ms of chatter)
 *
 * A naive polling loop running faster than the bounce period will
 * see MULTIPLE spurious 0->1->0 transitions for what the human
 * perceives as one clean press -- corrupting any edge-triggered
 * counter or state machine driven directly from IDR.
 * ============================================================ */

#include <stdint.h>
#include <stdbool.h>

#define DEBOUNCE_STABLE_MS   20U   /* must be N consecutive reads at
                                       the SAME level before we trust it;
                                       20ms is a conservative value that
                                       covers most tactile switches --
                                       datasheet bounce time should be
                                       consulted for production designs */

typedef struct {
    uint8_t  stable_state;   /* debounced, trusted logical state    */
    uint8_t  last_raw;       /* last raw sample, for edge detection */
    uint32_t stable_since_ms;
} button_debounce_t;

/* Call this once per polling tick (e.g. every 1ms from SysTick),
 * NOT in a tight spin loop -- deterministic sample rate is what
 * makes the ms-based counter meaningful. */
static bool button_poll(button_debounce_t *db, int raw_pressed,
                          uint32_t now_ms)
{
    if (raw_pressed != db->last_raw)
    {
        /* Raw level just changed -- restart the stability timer.
         * This is the same "reset-on-change" pattern used in
         * hardware Schmitt-trigger debounce circuits, just done
         * in software against a monotonic tick counter. */
        db->last_raw       = (uint8_t)raw_pressed;
        db->stable_since_ms = now_ms;
        return false;   /* no confirmed change yet */
    }

    if ((now_ms - db->stable_since_ms) >= DEBOUNCE_STABLE_MS &&
        db->stable_state != raw_pressed)
    {
        db->stable_state = (uint8_t)raw_pressed;
        return true;    /* confirmed, debounced transition occurred */
    }

    return false;
}

/* --- Usage inside a SysTick-driven main loop --- */
static volatile uint32_t g_tick_ms = 0;
void SysTick_Handler(void) { g_tick_ms++; }

int main(void)
{
    /* ... RCC + GPIO MODER/PUPDR init as in Section 3 ... */

    button_debounce_t btn = {0};

    while (1)
    {
        int raw_pressed = ((GPIOA->IDR & (1U << BUTTON_PIN)) == 0U);

        if (button_poll(&btn, raw_pressed, g_tick_ms))
        {
            /* This block executes EXACTLY ONCE per real physical
             * press/release, no matter how much electrical bounce
             * occurred -- suitable for driving a counter, toggling
             * a mode, or any edge-sensitive logic. */
            if (btn.stable_state)
            {
                GPIOA->BSRR = (1U << LED_PIN);
            }
            else
            {
                GPIOA->BSRR = (1U << (LED_PIN + 16U));
            }
        }
    }
}
```

## 5. Interrupt-Driven Alternative (EXTI) — Avoiding Polling Entirely

```c
/* ============================================================
 * Polling wastes CPU cycles and has latency bounded by loop
 * period. Production firmware typically uses EXTI (external
 * interrupt) lines so the button press wakes the CPU from sleep
 * (WFI) and is serviced with sub-microsecond latency -- same
 * power-efficiency philosophy as the SysTick+WFI delay pattern
 * from the LED-blink writeup.
 * ============================================================ */

#define EXTI_BASE      0x40013C00U
#define EXTI_IMR       (*(volatile uint32_t *)(EXTI_BASE + 0x00)) /* interrupt mask   */
#define EXTI_FTSR       (*(volatile uint32_t *)(EXTI_BASE + 0x0C)) /* falling trigger  */
#define EXTI_RTSR       (*(volatile uint32_t *)(EXTI_BASE + 0x08)) /* rising trigger   */
#define EXTI_PR        (*(volatile uint32_t *)(EXTI_BASE + 0x14)) /* pending register */

#define SYSCFG_BASE    0x40013800U
#define SYSCFG_EXTICR1 (*(volatile uint32_t *)(SYSCFG_BASE + 0x08))

/* Software debounce state, now touched from ISR context --
 * MUST be declared volatile since it's written asynchronously
 * relative to main()'s control flow. */
static volatile uint32_t g_last_press_tick = 0;
extern volatile uint32_t g_tick_ms;

void EXTI0_IRQHandler(void)
{
    if (EXTI_PR & (1U << BUTTON_PIN))
    {
        EXTI_PR = (1U << BUTTON_PIN);   /* W1C: write-1-to-clear pending bit --
                                            NOT a read-modify-write, writing 0
                                            to other bits is a no-op by design */

        /* Debounce-in-ISR pattern: ignore any edge that arrives
         * within DEBOUNCE_STABLE_MS of the last accepted edge.
         * This trades a tiny bit of missed-edge risk during the
         * bounce window for zero polling overhead. */
        if ((g_tick_ms - g_last_press_tick) >= 20U)
        {
            g_last_press_tick = g_tick_ms;

            /* Keep ISR work minimal -- toggle a flag/BSRR write only.
             * Any heavier logic belongs in main(), dispatched via a
             * flag check, to keep interrupt latency bounded (same
             * <100us interrupt-latency discipline as the Fenwick-tree
             * sensor-aggregation and HardFault-handler work). */
            GPIOA->ODR ^= (1U << LED_PIN);
        }
    }
}

static void button_exti_init(void)
{
    /* Route PA0 -> EXTI0 line via SYSCFG multiplexer */
    SYSCFG_EXTICR1 &= ~(0xFU << 0);   /* clear EXTI0 source select bits */
    SYSCFG_EXTICR1 |=  (0x0U << 0);   /* 0000 = Port A                  */

    EXTI_FTSR |= (1U << BUTTON_PIN);  /* trigger on falling edge
                                          (active-LOW press, pull-up idle-high) */
    EXTI_IMR  |= (1U << BUTTON_PIN);  /* unmask the interrupt line */

    /* NVIC enable omitted here -- core-specific (NVIC_EnableIRQ()
     * in CMSIS, or direct ISER register write on bare-metal) */
}
```

## 6. Polling vs. Interrupt-Driven: Trade-off Summary

| Aspect | Polling (Section 3/4) | Interrupt-Driven EXTI (Section 5) |
|---|---|---|
| CPU usage while idle | Continuous spin or fixed-rate sampling | Zero — core can `WFI` between events |
| Latency | Bounded by poll period (ms-scale) | Sub-µs, hardware-triggered |
| Debounce strategy | Time-window state machine, easy to reason about | Must debounce inside ISR or via timer-armed re-enable, trickier due to async context |
| Shared state hazards | None (single control flow) | `volatile` required on any ISR-shared variable; RMW on shared `ODR` needs `BSRR` (Section 4 of LED-blink writeup) |
| Best for | Simple, low-pin-count designs, teaching/bring-up | Power-sensitive designs, many concurrently-monitored inputs, real-time response requirements |

## 7. Key Takeaways

1. **`GPIO_IDR` is read-only and hardware-latched** — no read-modify-write hazard exists on the read side, unlike `ODR`.
2. **Floating inputs are a correctness bug, not a style choice.** An input pin with `PUPDR = NONE` and no external resistor will read noise; this is analogous to relying on undefined reset state in the `MODER` register from the LED-blink baseline — both are "works by accident" patterns that fail on different silicon revisions or board layouts.
3. **Mechanical bounce is a physical-layer reality that software must filter**, either via a time-windowed debounce state machine (Section 4) or a debounce-guard inside an EXTI ISR (Section 5). Skipping this is the input-side equivalent of skipping cache-maintenance operations on a DMA buffer — it "usually works" until it doesn't, and then it's a Heisenbug.
4. **Active-LOW vs. active-HIGH wiring changes both the pull-resistor choice and the sense of the comparison** — always verify against the schematic, not assumption.
5. **Polling is fine for bring-up and simple designs; interrupts are the production answer** when latency or power budget matters, mirroring the same "correctness first, then optimize for concurrency/power" progression used throughout the LED-blink, cache-coherency, and DVFS/RL-tuning work.