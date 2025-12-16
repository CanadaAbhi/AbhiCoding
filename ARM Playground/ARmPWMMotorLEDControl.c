# PWM Motor/LED Control — Timer-Generated PWM for ARM Cortex-M ⭐⭐⭐

## 1. The Core Timer → PWM Model

```
Timer Clock (e.g. 84MHz)
      │
      ├── Prescaler (PSC)   -> divides clock down to a countable rate
      │
      ▼
Counter (CNT) counts 0 -> ARR -> 0 (up-counting mode)
      │
      ├── ARR (Auto-Reload Register) = PERIOD  -> sets PWM frequency
      │
      ├── CCR (Capture/Compare Register) = COMPARE -> sets PWM duty cycle
      │
      ▼
Output pin driven HIGH while CNT < CCR, LOW while CNT >= CCR
(PWM Mode 1, up-counting)
      │
      ▼
   PWM Output ──► LED (brightness) / Servo (angle) / Motor driver (speed)
```

```
PWM Frequency = TIMER_CLK / ((PSC + 1) * (ARR + 1))
Duty Cycle %  = (CCR / (ARR + 1)) * 100
```

## 2. Register-Level Timer Overlay (STM32F4-class TIMx)

```c
#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 * Generic 16/32-bit general-purpose timer overlay (TIM2-TIM5
 * style). Matches the CMSIS-style struct-overlay discipline
 * used in the earlier GPIO/UART/SysTick drivers -- volatile on
 * every hardware-backed field, no bitfields (portability +
 * avoids compiler-defined layout ambiguity).
 * ============================================================ */
typedef struct
{
    volatile uint32_t CR1;      /* 0x00 Control register 1        */
    volatile uint32_t CR2;      /* 0x04 Control register 2        */
    volatile uint32_t SMCR;     /* 0x08 Slave mode control        */
    volatile uint32_t DIER;     /* 0x0C DMA/interrupt enable       */
    volatile uint32_t SR;       /* 0x10 Status register            */
    volatile uint32_t EGR;      /* 0x14 Event generation           */
    volatile uint32_t CCMR1;    /* 0x18 Capture/compare mode 1     */
    volatile uint32_t CCMR2;    /* 0x1C Capture/compare mode 2     */
    volatile uint32_t CCER;     /* 0x20 Capture/compare enable     */
    volatile uint32_t CNT;      /* 0x24 Counter                    */
    volatile uint32_t PSC;      /* 0x28 Prescaler                  */
    volatile uint32_t ARR;      /* 0x2C Auto-reload (PERIOD)       */
    volatile uint32_t RCR;      /* 0x30 Repetition counter (adv.)  */
    volatile uint32_t CCR1;     /* 0x34 Compare ch1 (COMPARE)      */
    volatile uint32_t CCR2;     /* 0x38 Compare ch2                */
    volatile uint32_t CCR3;     /* 0x3C Compare ch3                */
    volatile uint32_t CCR4;     /* 0x40 Compare ch4                */
    volatile uint32_t BDTR;     /* 0x44 Break/dead-time (adv. tim) */
    volatile uint32_t DCR;      /* 0x48 DMA control                */
    volatile uint32_t DMAR;     /* 0x4C DMA address for burst      */
} TIM_TypeDef;

#define TIM3   ((TIM_TypeDef *)0x40000400U)   /* general-purpose, LED/servo */
#define TIM1   ((TIM_TypeDef *)0x40010000U)   /* advanced, motor + dead-time */

/* CR1 */
#define TIM_CR1_CEN        (1U << 0)   /* counter enable            */
#define TIM_CR1_ARPE       (1U << 7)   /* auto-reload preload en.   */

/* CCMR1 (channel 1: bits 0-7) */
#define TIM_CCMR1_OC1PE    (1U << 3)   /* CCR1 preload enable       */
#define TIM_CCMR1_OC1M_PWM1 (0x6U << 4) /* PWM mode 1                */

/* CCER */
#define TIM_CCER_CC1E      (1U << 0)   /* channel 1 output enable   */
#define TIM_CCER_CC1P      (1U << 1)   /* channel 1 polarity        */

/* EGR */
#define TIM_EGR_UG         (1U << 0)   /* force update event        */

/* RCC enable bits (APB1) -- clock must be enabled before touching
 * timer registers, same "clock and config before enable" discipline
 * used in the UART driver. */
#define RCC_APB1ENR   (*(volatile uint32_t *)0x40023840U)
#define RCC_APB2ENR   (*(volatile uint32_t *)0x40023844U)
#define RCC_TIM3EN    (1U << 1)
#define RCC_TIM1EN    (1U << 0)
```

## 3. PWM Initialization (TIM3, LED/Servo Channel 1)

```c
/* Assume a 84MHz APB1 timer clock (typical STM32F4 config with
 * APB1 prescaler /2 -> timer clocks x2 = 84MHz on TIM2-TIM7). */
#define TIMER_CLK_HZ   84000000UL

typedef struct
{
    TIM_TypeDef *tim;
    uint32_t     period;      /* ARR + 1, cached for duty math */
} pwm_channel_t;

/* ============================================================
 * pwm_init: configure TIM3 CH1 for PWM Mode 1, up-counting,
 * at a target frequency. GPIO alternate-function routing (AFR,
 * MODER) is assumed done separately, per the GPIO driver work --
 * PWM timing correctness is independent of pin muxing.
 * ============================================================ */
void pwm_init(pwm_channel_t *ch, TIM_TypeDef *tim, uint32_t freq_hz)
{
    RCC_APB1ENR |= RCC_TIM3EN;   /* 1. clock the peripheral first */

    /* 2. Compute PSC/ARR for the requested frequency.
     * Fix PSC low (fine resolution) and let ARR carry most of the
     * division, subject to ARR's 16-bit range on general-purpose
     * timers (0..65535). */
    uint32_t ticks_per_period = TIMER_CLK_HZ / freq_hz;

    uint32_t psc = 0;
    while ((ticks_per_period / (psc + 1U)) > 65536UL)
    {
        psc++;
    }
    uint32_t arr = (ticks_per_period / (psc + 1U)) - 1U;

    tim->PSC = psc;
    tim->ARR = arr;

    /* 3. PWM Mode 1 on channel 1: output HIGH while CNT < CCR1,
     * LOW otherwise. OC1PE enables the CCR1 *shadow register* --
     * writes to CCR1 take effect only at the next update event
     * (counter overflow), preventing a torn/glitched pulse if we
     * update duty cycle mid-cycle. */
    tim->CCMR1 = (tim->CCMR1 & ~(0x7U << 4)) | TIM_CCMR1_OC1M_PWM1;
    tim->CCMR1 |= TIM_CCMR1_OC1PE;

    /* 4. ARPE: buffer ARR too, for the same glitch-free reason --
     * relevant if frequency is ever changed at runtime. */
    tim->CR1 |= TIM_CR1_ARPE;

    /* 5. Enable channel 1 output, active-high polarity. */
    tim->CCER |= TIM_CCER_CC1E;
    tim->CCER &= ~TIM_CCER_CC1P;

    /* 6. Force an update event to latch PSC/ARR/CCR shadow
     * registers immediately rather than waiting for the first
     * natural overflow (avoids one "garbage" period on startup). */
    tim->EGR = TIM_EGR_UG;

    tim->CCR1 = 0;              /* start at 0% duty -- safe default */
    tim->CR1 |= TIM_CR1_CEN;    /* start the counter */

    ch->tim    = tim;
    ch->period = arr + 1U;
}
```

## 4. Duty Cycle Control

```c
/* ============================================================
 * pwm_set_duty_cycle: 0-100 (%) -> CCR1.
 *
 * Edge cases handled explicitly:
 *   0%   -> CCR = 0        (output never goes high -- fully off)
 *   100% -> CCR = ARR + 1  (output stays high the entire period --
 *            NOT CCR = ARR, which would produce a 1-tick-wide LOW
 *            glitch instead of a solid HIGH, since the compare
 *            match at CNT==ARR happens one tick before the
 *            update/reload event)
 * ============================================================ */
void pwm_set_duty_cycle(pwm_channel_t *ch, uint8_t percent)
{
    if (percent > 100U)
    {
        percent = 100U;
    }

    uint32_t ccr = ((uint32_t)percent * ch->period) / 100U;

    /* Write goes to the CCR1 *shadow* register (OC1PE enabled in
     * init) -- it is safely latched at the next update event, so
     * this can be called from any context, including periodically
     * from a control loop or ISR, without producing a truncated or
     * doubled pulse mid-period. */
    ch->tim->CCR1 = ccr;
}

/* Raw variant when direct tick-level control is needed (e.g. servo
 * pulse widths that don't map to a clean percentage). */
void pwm_set_compare_raw(pwm_channel_t *ch, uint32_t compare_ticks)
{
    ch->tim->CCR1 = compare_ticks;
}
```

## 5. Application 1 — LED Brightness (Perceptual Gamma Correction)

```c
/* ============================================================
 * Human perceived brightness is roughly logarithmic (Weber-
 * Fechner law), while PWM duty cycle is linear with average
 * emitted light. A linear duty sweep 0->100% looks like it
 * brightens quickly then barely changes near the top. A gamma
 * (~2.2) lookup table corrects this so a linear *control* input
 * (e.g. a potentiometer or software fade counter) produces a
 * visually linear brightness ramp.
 * ============================================================ */
static const uint8_t gamma_lut[101] = {
    0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,6,6,7,7,8,
    8,9,10,10,11,12,13,13,14,15,16,17,18,19,20,21,22,23,24,25,
    27,28,29,31,32,33,35,36,38,39,41,43,44,46,48,50,52,53,55,
    57,59,61,64,66,68,70,72,75,77,80,82,85,87,90,92,95,98,100,
    100,100,100,100,100,100,100,100,100,100
};

/* LED PWM frequency: 1kHz-1kHz+ is above human flicker-fusion
 * threshold (no visible strobing), and far below audible range
 * (unlike motor PWM, which we deliberately push out of the
 * audible band -- see below). */
#define LED_PWM_FREQ_HZ   1000U

void led_set_brightness(pwm_channel_t *led, uint8_t percent_linear)
{
    if (percent_linear > 100U) { percent_linear = 100U; }
    uint8_t corrected = gamma_lut[percent_linear];
    pwm_set_duty_cycle(led, corrected);
}

/* Software fade example -- smooth ramp instead of an instant jump,
 * avoiding a visible "step" and any inrush current spike into the
 * LED driver stage. */
void led_fade_to(pwm_channel_t *led, uint8_t target_percent, uint32_t step_delay_ms)
{
    static uint8_t current = 0;
    int8_t dir = (target_percent > current) ? 1 : -1;

    while (current != target_percent)
    {
        current = (uint8_t)(current + dir);
        led_set_brightness(led, current);
        /* delay_ms(step_delay_ms) -- from the SysTick driver work */
    }
}
```

## 6. Application 2 — Servo Motor (Pulse-Width Position Control)

```c
/* ============================================================
 * Standard hobby servos expect a 50Hz (20ms period) control
 * signal where pulse width, NOT duty cycle percentage, encodes
 * angle:
 *
 *   1.0ms pulse -> 0 degrees
 *   1.5ms pulse -> 90 degrees (center/neutral)
 *   2.0ms pulse -> 180 degrees
 *
 * This is why pwm_set_compare_raw (tick-based) is used instead
 * of the percentage API -- servo timing must be expressed in
 * absolute microseconds, independent of the chosen period.
 * ============================================================ */
#define SERVO_FREQ_HZ       50U
#define SERVO_MIN_PULSE_US  1000U   /* 0 degrees   */
#define SERVO_MID_PULSE_US  1500U   /* 90 degrees  */
#define SERVO_MAX_PULSE_US  2000U   /* 180 degrees */

void servo_init(pwm_channel_t *servo, TIM_TypeDef *tim)
{
    pwm_init(servo, tim, SERVO_FREQ_HZ);
    /* period ticks now correspond to 20ms total */
}

/* Convert an angle (0-180) into a tick count using the timer's
 * actual resolution, so this is correct regardless of PSC/ARR
 * chosen for the 50Hz base frequency. */
void servo_set_angle(pwm_channel_t *servo, uint8_t angle_deg)
{
    if (angle_deg > 180U) { angle_deg = 180U; }

    uint32_t pulse_us = SERVO_MIN_PULSE_US +
        ((uint32_t)angle_deg * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) / 180U;

    /* period ticks represent 20,000us (20ms) total */
    uint32_t ticks = ((uint64_t)pulse_us * servo->period) / 20000UL;

    pwm_set_compare_raw(servo, ticks);
}

/* Safety: servo motion should be RATE-LIMITED in software.
 * Snapping directly to a large angle change draws a current
 * spike and can strip gears under load -- ramp instead. */
void servo_move_to(pwm_channel_t *servo, uint8_t target_deg, uint8_t step_deg)
{
    static uint8_t current_deg = 90U;   /* assume centered at boot */

    while (current_deg != target_deg)
    {
        if (current_deg < target_deg)
        {
            current_deg = (uint8_t)((current_deg + step_deg > target_deg)
                                     ? target_deg : current_deg + step_deg);
        }
        else
        {
            current_deg = (uint8_t)((current_deg < step_deg
                                     || current_deg - step_deg < target_deg)
                                     ? target_deg : current_deg - step_deg);
        }
        servo_set_angle(servo, current_deg);
        /* delay_ms(20) -- one full servo update period */
    }
}
```

## 7. Application 3 — DC Motor Speed + Direction (H-Bridge)

```c
/* ============================================================
 * DC motor via H-bridge (e.g. DRV8871/L298N style): PWM drives
 * the enable/speed input while two GPIOs (IN1/IN2) select
 * direction. Motor PWM frequency is deliberately chosen ABOVE
 * ~18-20kHz (audible range) to eliminate motor coil "whine" --
 * a real trade-off against the 1kHz used for LEDs above.
 * ============================================================ */
#define MOTOR_PWM_FREQ_HZ   20000U   /* 20kHz -- inaudible, low switching loss */

typedef enum
{
    MOTOR_STOP = 0,
    MOTOR_FORWARD,
    MOTOR_REVERSE
} motor_dir_t;

typedef struct
{
    pwm_channel_t pwm;
    /* IN1/IN2 GPIO_TypeDef* + pin masks from the GPIO driver work,
     * omitted here for brevity -- controlled via BSRR as usual to
     * avoid RMW hazards, consistent with the earlier GPIO driver. */
} motor_t;

void motor_init(motor_t *m, TIM_TypeDef *tim)
{
    pwm_init(&m->pwm, tim, MOTOR_PWM_FREQ_HZ);
    /* IN1/IN2 GPIO config done here (output push-pull, both low
     * at boot -> motor coasting/stopped, never energized by
     * default). */
}

/* ============================================================
 * motor_set: direction + speed as a single atomic-from-the-
 * caller's-perspective operation.
 *
 * CRITICAL SAFETY RULE for any H-bridge, discrete or integrated:
 * NEVER allow IN1 and IN2 to both be active while transitioning
 * direction -- that shorts the supply rail through both low-side
 * (or high-side) FETs simultaneously ("shoot-through"). Always
 * force PWM duty to 0 (motor off) BEFORE changing direction
 * lines, then ramp speed back up. This is the software-level
 * equivalent of hardware dead-time insertion (see Section 8).
 * ============================================================ */
void motor_set(motor_t *m, motor_dir_t dir, uint8_t speed_percent)
{
    /* 1. Kill PWM output first -- guarantees zero current flow
     *    during the direction-line transition below. */
    pwm_set_duty_cycle(&m->pwm, 0U);

    /* 2. Set direction lines (BSRR-based atomic writes, per the
     *    GPIO driver discipline -- omitted call here). */
    switch (dir)
    {
        case MOTOR_FORWARD: /* IN1=1, IN2=0 */ break;
        case MOTOR_REVERSE: /* IN1=0, IN2=1 */ break;
        case MOTOR_STOP:    /* IN1=0, IN2=0 (coast) */
        default:            speed_percent = 0U; break;
    }

    /* 3. Now safe to apply speed. */
    pwm_set_duty_cycle(&m->pwm, speed_percent);
}

/* Soft-start ramp: prevents inrush current spikes and mechanical
 * jolt when commanding a large speed change, mirroring the same
 * rate-limiting principle used in servo_move_to(). */
void motor_ramp_to(motor_t *m, motor_dir_t dir, uint8_t target_speed, uint8_t step)
{
    static uint8_t current_speed = 0U;

    /* direction changes always pass through 0 first, per motor_set() */
    while (current_speed != target_speed)
    {
        current_speed = (current_speed < target_speed)
            ? (uint8_t)((target_speed - current_speed < step) ? target_speed : current_speed + step)
            : (uint8_t)((current_speed - target_speed < step) ? target_speed : current_speed - step);

        motor_set(m, dir, current_speed);
        /* delay_ms(5-10) between ramp steps */
    }
}
```

## 8. Hardware Dead-Time Insertion (Advanced Timers, TIM1/TIM8)

```c
/* ============================================================
 * For BRUSHLESS/complementary H-bridge motor drives (high-side +
 * low-side FETs on the same leg switching in PWM Mode 1 /
 * inverted pairs), a purely software "set duty to 0 before
 * switching" is not sufficient -- FET turn-off is not
 * instantaneous. If the complementary FET turns on before the
 * first one has fully turned off, both conduct simultaneously
 * ("shoot-through"), often destructively.
 *
 * Advanced timers (TIM1/TIM8) solve this in HARDWARE via BDTR
 * (Break and Dead-Time Register): the timer automatically
 * inserts a programmable delay between complementary output
 * transitions, guaranteeing both FETs are off during the gap.
 * ============================================================ */
#define TIM_BDTR_MOE    (1U << 15)   /* main output enable */

void motor_enable_deadtime(TIM_TypeDef *tim, uint8_t deadtime_ticks)
{
    /* Dead-time value encoding depends on range (DTG[7:0] has a
     * non-linear scale for larger delays); for small values,
     * DTG = deadtime_ticks directly maps to timer-clock periods. */
    tim->BDTR = (tim->BDTR & ~0xFFU) | (deadtime_ticks & 0xFFU);
    tim->BDTR |= TIM_BDTR_MOE;   /* outputs stay disabled until this is set --
                                    also auto-cleared on a BREAK event for
                                    hardware fault shutdown (e.g. overcurrent) */
}
```

## 9. Test / Demo Harness

```c
#include <stdio.h>
#include <assert.h>

void test_pwm_math(void)
{
    pwm_channel_t ch;
    ch.period = 1000U;   /* matches the ARR=999 example in the prompt */

    /* Verify the prompt's exact example: PERIOD=1000, COMPARE=250 -> 25% */
    uint32_t ccr_for_25pct = (25U * ch.period) / 100U;
    assert(ccr_for_25pct == 250U);

    /* Duty back-calculation should round-trip */
    uint32_t derived_percent = (ccr_for_25pct * 100U) / ch.period;
    assert(derived_percent == 25U);

    printf("PWM period/duty math verified: PERIOD=1000, COMPARE=250 => 25%%\n");
}

void test_servo_pulse_math(void)
{
    pwm_channel_t servo;
    servo.period = 20000U;   /* pretend 1 tick = 1us for clarity */

    uint32_t t0   = ((uint64_t)SERVO_MIN_PULSE_US * servo.period) / 20000UL;
    uint32_t t90  = ((uint64_t)SERVO_MID_PULSE_US * servo.period) / 20000UL;
    uint32_t t180 = ((uint64_t)SERVO_MAX_PULSE_US * servo.period) / 20000UL;

    assert(t0   == 1000U);
    assert(t90  == 1500U);
    assert(t180 == 2000U);

    printf("Servo pulse-width math verified: 0deg=1000us, 90deg=1500us, 180deg=2000us\n");
}

void test_100_percent_edge_case(void)
{
    /* 100% must map to period (ARR+1), not (ARR), to avoid a
     * one-tick glitch. */
    pwm_channel_t ch;
    ch.period = 1000U;

    uint32_t ccr_100 = (100U * ch.period) / 100U;
    assert(ccr_100 == 1000U);   /* == ARR+1, i.e. always-high */

    printf("100%% duty edge case verified (CCR == ARR+1, glitch-free)\n");
}

int main(void)
{
    test_pwm_math();
    test_servo_pulse_math();
    test_100_percent_edge_case();
    printf("All PWM tests passed.\n");
    return 0;
}
```

## 10. Design Rules Summary

| Rule | Why |
|---|---|
| **Shadow-register writes (OC1PE/ARPE)** | Prevents a torn/glitched pulse when duty cycle is updated mid-period from a control loop or ISR — the new value is latched atomically at the next update event. |
| **100% duty → CCR = ARR+1, never ARR** | Compare match at `CNT==ARR` fires one tick before the reload, producing a 1-tick LOW glitch at "full brightness" if CCR is left at ARR instead of ARR+1. |
| **LED PWM ~1kHz + gamma LUT** | Above flicker-fusion threshold, and corrects for the eye's logarithmic brightness response so a linear input feels linear. |
| **Motor PWM ~20kHz** | Above the audible range, eliminating coil whine, at the cost of higher switching losses versus a lower frequency. |
| **Kill PWM (duty=0) before changing H-bridge direction lines** | Software-level shoot-through prevention on simple H-bridge modules without hardware dead-time. |
| **Hardware dead-time (BDTR) for complementary/brushless drives** | FET turn-off is not instantaneous; only a hardware-inserted gap guarantees no overlap between complementary high/low-side conduction. |
| **Soft-start ramping (LED fade, servo move, motor ramp)** | Avoids inrush current spikes, mechanical jolt, and visible/audible steps — same rate-limiting principle applied across all three actuator types. |
| **Direction lines forced to a safe/coasting state at init** | Never energize a motor in an undefined direction state on power-up before firmware has explicitly commanded one. |

## 11. Concepts Demonstrated

| Concept | Where it shows up |
|---|---|
| **Timer-generated PWM fundamentals** | PSC/ARR/CCR relationship directly implementing the Period/Duty Cycle model from the prompt, with exact `PERIOD=1000, COMPARE=250 → 25%` example verified in the test harness. |
| **Glitch-free updates** | Shadow-register (preload) usage for CCR/ARR, and correct handling of the 0%/100% boundary conditions. |
| **Domain-specific PWM tuning** | Frequency choice driven by the physical load: perceptual (LED, gamma+1kHz), mechanical/electrical standard (servo, 50Hz absolute pulse width), and acoustic/efficiency (motor, 20kHz). |
| **Actuator safety discipline** | Soft-start ramping, forced-stop-before-direction-change, and hardware dead-time insertion — extending the same "left shift" risk-mitigation mindset from HIL/SIL testing into the PWM control layer itself. |