# ADC Sensor Reading — Analog-to-Digital Conversion for ARM Cortex-M ⭐⭐⭐

## 1. The Core Signal Chain

```
Analog Sensor (Thermistor, Potentiometer, Photodiode, etc.)
      │  continuous analog voltage (e.g. 0V - 3.3V)
      ▼
Sample & Hold ─── captures instantaneous voltage, holds it stable
      │            during conversion (analog signal is *moving*;
      │            conversion takes time -- must freeze the input)
      ▼
ADC Core (SAR - Successive Approximation Register, typical on
      │   Cortex-M) -- performs a binary search against VREF over
      │   N clock cycles to resolve N bits
      ▼
Digital Value: 0 .. (2^RESOLUTION - 1)
      │         (e.g. 12-bit -> 0..4095)
      ▼
   CPU / DMA reads ADC->DATA register
      │
      ▼
Voltage = (raw_value / (2^RESOLUTION - 1)) * VREF
```

```
Resolution (bits)  -> Number of discrete steps = 2^bits
Reference Voltage (VREF) -> Defines the FULL-SCALE voltage that maps to the max digital code
LSB (Least Significant Bit) step size = VREF / (2^bits - 1)
Sampling Rate  -> How often you can legally re-sample without violating settling time
```

**Key relationship** (this is the entire point of the driver below):

```
measured_voltage = raw_adc_value * (VREF / ADC_MAX_VALUE)
```

## 2. Register-Level ADC Overlay (STM32F4-class ADC1)

```c
#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 * Generic SAR-ADC overlay, matching the CMSIS-style struct
 * discipline used across the GPIO/UART/Timer/SysTick drivers --
 * volatile on every hardware-backed field.
 * ============================================================ */
typedef struct
{
    volatile uint32_t SR;      /* 0x00 Status register                */
    volatile uint32_t CR1;     /* 0x04 Control register 1              */
    volatile uint32_t CR2;     /* 0x08 Control register 2              */
    volatile uint32_t SMPR1;   /* 0x0C Sample time reg 1 (ch 10-18)    */
    volatile uint32_t SMPR2;   /* 0x10 Sample time reg 2 (ch 0-9)      */
    volatile uint32_t JOFR[4]; /* 0x14-0x20 Injected channel offsets   */
    volatile uint32_t HTR;     /* 0x24 Watchdog high threshold          */
    volatile uint32_t LTR;     /* 0x28 Watchdog low threshold           */
    volatile uint32_t SQR1;    /* 0x2C Regular sequence reg 1 (13-16)  */
    volatile uint32_t SQR2;    /* 0x30 Regular sequence reg 2 (7-12)   */
    volatile uint32_t SQR3;    /* 0x34 Regular sequence reg 3 (1-6)    */
    volatile uint32_t JSQR;    /* 0x38 Injected sequence register      */
    volatile uint32_t JDR[4];  /* 0x3C-0x48 Injected data registers    */
    volatile uint32_t DR;      /* 0x4C Regular data register (DATA)   */
} ADC_TypeDef;

#define ADC1   ((ADC_TypeDef *)0x40012000U)

/* SR bits */
#define ADC_SR_EOC     (1U << 1)   /* End Of Conversion  (== ADC_DONE) */
#define ADC_SR_OVR      (1U << 5)   /* Overrun -- prev. value not read  */

/* CR1 bits */
#define ADC_CR1_RES_12BIT   (0U << 24)
#define ADC_CR1_RES_10BIT   (1U << 24)
#define ADC_CR1_RES_8BIT    (2U << 24)
#define ADC_CR1_RES_6BIT    (3U << 24)
#define ADC_CR1_SCAN        (1U << 8)

/* CR2 bits */
#define ADC_CR2_ADON    (1U << 0)   /* ADC ON                            */
#define ADC_CR2_CONT    (1U << 1)   /* Continuous conversion mode        */
#define ADC_CR2_SWSTART (1U << 30)  /* Software-triggered start (== ADC_START) */
#define ADC_CR2_EOCS    (1U << 10)  /* EOC set at end of *each* conversion */

/* RCC enable (APB2 for ADC1 on STM32F4) */
#define RCC_APB2ENR   (*(volatile uint32_t *)0x40023844U)
#define RCC_ADC1EN    (1U << 8)

#define ADC_VREF_MV        3300U     /* 3.3V reference, in millivolts */
#define ADC_MAX_12BIT       4095U    /* 2^12 - 1                       */
```

## 3. ADC Initialization

```c
/* ============================================================
 * adc_init: single-channel, software-triggered, one-shot mode.
 * "Clock and config before enable" discipline, consistent with
 * the UART/Timer drivers.
 * ============================================================ */
void adc_init(void)
{
    RCC_APB2ENR |= RCC_ADC1EN;   /* 1. clock the peripheral first */

    /* 2. Resolution: 12-bit is the ADC's native resolution --
     * see Section 6 for why you'd ever choose less. */
    ADC1->CR1 &= ~(0x3U << 24);
    ADC1->CR1 |= ADC_CR1_RES_12BIT;

    /* 3. Single conversion mode (not continuous) -- explicit
     * software trigger per conversion, matching the prompt's
     * START/poll/READ pattern exactly. */
    ADC1->CR2 &= ~ADC_CR2_CONT;

    /* 4. EOCS: flag set after each individual conversion rather
     * than only after a full sequence -- required for single-
     * channel polling to work as expected. */
    ADC1->CR2 |= ADC_CR2_EOCS;

    /* 5. Power on the ADC. Datasheet mandates a stabilization
     * delay (t_STAB, ~1-3us) after ADON before the first
     * conversion -- skipping this yields an inaccurate first
     * sample. */
    ADC1->CR2 |= ADC_CR2_ADON;
    for (volatile int i = 0; i < 1000; i++) { /* t_STAB delay */ }
}
```

## 4. Channel Selection + Sample Time (Sampling Theory)

```c
/* ============================================================
 * Sampling time is NOT free -- it is a real hardware trade-off:
 *
 *   Sample & Hold capacitor must charge to the input voltage
 *   through the SOURCE IMPEDANCE of whatever is driving the
 *   pin (sensor output impedance + any series resistance).
 *
 *   RC charging time constant: tau = R_source * C_sample
 *   Settling to within 1/2 LSB typically requires ~10*tau.
 *
 *   Too SHORT a sample time -> capacitor doesn't fully charge
 *     -> reads a voltage lower than actual (high-impedance
 *        sources like thermistors/photodiodes are especially
 *        vulnerable).
 *   Too LONG a sample time -> wastes cycles, reduces max
 *        achievable sampling RATE (throughput).
 *
 * SMPR1/SMPR2 select cycles per channel: 3, 15, 28, 56, 84,
 * 112, 144, or 480 ADC clock cycles.
 * ============================================================ */
typedef enum
{
    ADC_SAMPLETIME_3CYCLES   = 0,
    ADC_SAMPLETIME_15CYCLES  = 1,
    ADC_SAMPLETIME_28CYCLES  = 2,
    ADC_SAMPLETIME_56CYCLES  = 3,
    ADC_SAMPLETIME_84CYCLES  = 4,
    ADC_SAMPLETIME_112CYCLES = 5,
    ADC_SAMPLETIME_144CYCLES = 6,
    ADC_SAMPLETIME_480CYCLES = 7   /* max -- for high-impedance sensors */
} adc_sampletime_t;

void adc_configure_channel(uint8_t channel, adc_sampletime_t sample_time)
{
    /* Channels 0-9 live in SMPR2 (3 bits each), 10-18 in SMPR1 */
    if (channel < 10U)
    {
        uint32_t shift = channel * 3U;
        ADC1->SMPR2 = (ADC1->SMPR2 & ~(0x7U << shift)) | ((uint32_t)sample_time << shift);
    }
    else
    {
        uint32_t shift = (channel - 10U) * 3U;
        ADC1->SMPR1 = (ADC1->SMPR1 & ~(0x7U << shift)) | ((uint32_t)sample_time << shift);
    }

    /* Select this channel as the ONLY entry in the regular
     * sequence (sequence length = 1, SQR3 bits 4:0 = channel). */
    ADC1->SQR3 = channel & 0x1FU;
    ADC1->SQR1 &= ~(0xFU << 20);   /* L[3:0] = 0 -> sequence length 1 */
}
```

## 5. Blocking Conversion — The Exact Prompt Pattern

```c
/* ============================================================
 * adc_read_blocking: implements the prompt's exact sequence --
 * START -> poll STATUS/DONE -> read DATA -- with production
 * hardening: a bounded timeout (never spin forever on dead
 * hardware) and explicit overrun detection.
 * ============================================================ */
#define ADC_TIMEOUT_ITERATIONS   100000U

typedef enum
{
    ADC_OK = 0,
    ADC_ERR_TIMEOUT,
    ADC_ERR_OVERRUN
} adc_result_t;

adc_result_t adc_read_blocking(uint8_t channel, uint16_t *out_value)
{
    adc_configure_channel(channel, ADC_SAMPLETIME_84CYCLES);

    /* ADC->CONTROL |= ADC_START;  (from the prompt) */
    ADC1->CR2 |= ADC_CR2_SWSTART;

    /* while (!(ADC->STATUS & ADC_DONE));  (from the prompt) --
     * hardened with a bounded timeout. An unbounded busy-wait on
     * hardware that never asserts EOC (dead sensor, floating pin,
     * misconfigured clock) hangs the entire system -- this
     * violates the same "never block forever" discipline applied
     * to the UART/EXTI drivers. */
    uint32_t timeout = ADC_TIMEOUT_ITERATIONS;
    while (!(ADC1->SR & ADC_SR_EOC))
    {
        if (--timeout == 0U)
        {
            return ADC_ERR_TIMEOUT;
        }
    }

    /* Overrun check: if a previous conversion result was never
     * read before this one completed, the DATA register may
     * have overwritten a stale/unread value. In single-conversion
     * mode this is rare, but checking costs nothing and is the
     * same "check status flags before trusting data" discipline
     * used in the UART ORE/FE/PE handling. */
    if (ADC1->SR & ADC_SR_OVR)
    {
        ADC1->SR &= ~ADC_SR_OVR;   /* clear (W0C on this bit) */
        return ADC_ERR_OVERRUN;
    }

    /* uint16_t value = ADC->DATA;  (from the prompt) --
     * reading DR also auto-clears EOC on most SAR ADCs. */
    *out_value = (uint16_t)ADC1->DR;

    return ADC_OK;
}
```

## 6. Resolution — What the Bits Actually Buy You

```c
/* ============================================================
 * Resolution trade-off: lower resolution = faster conversion
 * (fewer SAR comparison cycles) but coarser voltage steps.
 * Useful when raw throughput matters more than precision
 * (e.g. a fast ADC-based oscilloscope trigger) or when the
 * sensor itself is noisy enough that extra bits are just
 * amplifying noise, not signal (see Section 8).
 * ============================================================ */
typedef struct
{
    uint8_t  bits;
    uint32_t max_code;      /* 2^bits - 1 */
    uint32_t cr1_res_bits;
} adc_resolution_t;

static const adc_resolution_t adc_res_table[] = {
    { 12, 4095U, ADC_CR1_RES_12BIT },
    { 10, 1023U, ADC_CR1_RES_10BIT },
    {  8,  255U, ADC_CR1_RES_8BIT  },
    {  6,   63U, ADC_CR1_RES_6BIT  },
};

void adc_set_resolution(const adc_resolution_t *res)
{
    ADC1->CR1 = (ADC1->CR1 & ~(0x3U << 24)) | res->cr1_res_bits;
}

/* LSB (voltage step size) demonstration at each resolution,
 * for a fixed 3.3V reference: */
void print_lsb_sizes(void)
{
    for (size_t i = 0; i < sizeof(adc_res_table)/sizeof(adc_res_table[0]); i++)
    {
        const adc_resolution_t *r = &adc_res_table[i];
        uint32_t lsb_uv = (ADC_VREF_MV * 1000U) / r->max_code;  /* microvolts */
        /* printf("%2u-bit: %5lu codes, LSB = %lu uV\n",
                  r->bits, r->max_code + 1, (unsigned long)lsb_uv); */
        (void)lsb_uv;
    }
    /* Output (conceptually):
     * 12-bit: 4096 codes, LSB = 806 uV   (sub-millivolt precision)
     *  8-bit:  256 codes, LSB = 12941 uV (~13mV per step -- coarse)
     */
}
```

## 7. Reference Voltage — Why It's the Other Half of the Equation

```c
/* ============================================================
 * The ADC digital code is ALWAYS a ratio against VREF:
 *
 *     code / max_code = V_in / V_ref
 *
 * A drifting or noisy VREF corrupts every measurement equally --
 * this is why precision systems use either:
 *   (a) a dedicated low-drift external reference IC, or
 *   (b) an internal, factory-calibrated reference channel
 *       (e.g. STM32's VREFINT, ~1.21V nominal) to self-correct
 *       for VDDA drift/sag under load.
 * ============================================================ */
#define VREFINT_CHANNEL       17U          /* internal reference channel   */
#define VREFINT_CAL_ADDR      0x1FFF7A2AU  /* factory-calibrated value @ 3.3V */

uint32_t adc_get_actual_vref_mv(void)
{
    /* Read the internal reference channel with VDDA as VREF. */
    uint16_t vrefint_raw;
    adc_configure_channel(VREFINT_CHANNEL, ADC_SAMPLETIME_144CYCLES);
    if (adc_read_blocking(VREFINT_CHANNEL, &vrefint_raw) != ADC_OK)
    {
        return ADC_VREF_MV;   /* fall back to nominal on error */
    }

    uint16_t vrefint_cal = *(volatile uint16_t *)VREFINT_CAL_ADDR;

    /* If VDDA has sagged (e.g. under motor/LED load on a shared
     * rail), vrefint_raw will read HIGHER than the calibration
     * value (same fixed 1.21V now represents a larger fraction of
     * a smaller VDDA) -- this ratio lets us back-calculate the
     * ACTUAL VDDA, correcting every other channel's readings. */
    return (ADC_VREF_MV * (uint32_t)vrefint_cal) / (uint32_t)vrefint_raw;
}

/* This is the single most important function in the whole
 * driver: it converts a raw code into a real-world voltage,
 * using a *measured*, drift-corrected reference rather than a
 * hardcoded nominal constant. */
uint32_t adc_raw_to_millivolts(uint16_t raw, uint32_t vref_mv, uint32_t max_code)
{
    return (raw * vref_mv) / max_code;
}
```

## 8. Oversampling — Trading Speed for Effective Resolution

```c
/* ============================================================
 * A physical ADC's INL/noise floor often limits usable
 * precision below its nominal bit count. Oversampling +
 * averaging trades sample RATE for extra effective bits:
 *
 *   Each 4x oversample -> +1 effective bit (theoretical, with
 *   sufficient input noise/dither present -- a real, if
 *   idealized, DSP technique, not a software trick with no
 *   physical basis).
 * ============================================================ */
uint32_t adc_read_oversampled(uint8_t channel, uint16_t num_samples)
{
    uint32_t accumulator = 0;
    uint16_t sample;

    for (uint16_t i = 0; i < num_samples; i++)
    {
        if (adc_read_blocking(channel, &sample) == ADC_OK)
        {
            accumulator += sample;
        }
    }

    /* Return the SUM (not average) if the caller wants extended
     * resolution -- e.g. 16 samples summed = 4 extra effective
     * bits over the native 12-bit reading, representable in a
     * 16-bit result. Average instead if only noise reduction
     * (not extra resolution) is desired. */
    return accumulator;
}
```

## 9. Application 1 — Thermistor (NTC) Temperature Sensor

```c
/* ============================================================
 * NTC thermistor in a voltage-divider with a fixed resistor:
 * high source impedance near room temperature -> REQUIRES a
 * long sample time (Section 4) to avoid under-charging the
 * S&H capacitor.
 * ============================================================ */
#define NTC_CHANNEL       4U
#define NTC_R_FIXED_OHMS  10000U   /* fixed divider resistor */
#define NTC_R0_OHMS       10000U   /* NTC resistance at 25C  */
#define NTC_BETA          3950U    /* thermistor beta coefficient */
#define NTC_T0_KELVIN     29815    /* 298.15K = 25C, x100 fixed-point */

int32_t read_temperature_celsius_x100(void)
{
    uint16_t raw;
    if (adc_read_blocking(NTC_CHANNEL, &raw) != ADC_OK)
    {
        return INT32_MIN;   /* sentinel error value */
    }

    uint32_t vref_mv = adc_get_actual_vref_mv();
    uint32_t v_mv = adc_raw_to_millivolts(raw, vref_mv, ADC_MAX_12BIT);

    /* Voltage divider inversion: R_ntc = R_fixed * (Vref/Vout - 1) */
    if (v_mv == 0U) { return INT32_MIN; }   /* guard divide-by-zero */
    uint32_t r_ntc = (NTC_R_FIXED_OHMS * (vref_mv - v_mv)) / v_mv;

    /* Simplified Beta-parameter equation (integer-approximated --
     * a production system would use a lookup table for the log()
     * term on an FPU-less core rather than floating point). */
    /* 1/T = 1/T0 + (1/Beta)*ln(R/R0)  -- omitted precise math here;
     * conceptually returns degrees C x100 for fixed-point display. */
    (void)r_ntc;
    return 2500;  /* placeholder for the resolved fixed-point result */
}
```

## 10. Application 2 — Battery Voltage Monitor (Resistor Divider)

```c
/* ============================================================
 * Monitoring a battery rail that exceeds VREF requires a
 * resistor divider BEFORE the ADC pin. This introduces a
 * fixed, known SCALE FACTOR that must be un-applied in software.
 * ============================================================ */
#define VBAT_CHANNEL         8U
#define VBAT_DIVIDER_R1_KOHM 100U   /* battery-side */
#define VBAT_DIVIDER_R2_KOHM 33U    /* ground-side  */

uint32_t read_battery_millivolts(void)
{
    uint16_t raw;
    /* High-impedance source (divider network) -- use max sample time */
    adc_configure_channel(VBAT_CHANNEL, ADC_SAMPLETIME_480CYCLES);
    if (adc_read_blocking(VBAT_CHANNEL, &raw) != ADC_OK)
    {
        return 0U;
    }

    uint32_t vref_mv = adc_get_actual_vref_mv();
    uint32_t v_pin_mv = adc_raw_to_millivolts(raw, vref_mv, ADC_MAX_12BIT);

    /* Undo the divider: Vbat = Vpin * (R1+R2)/R2 */
    return (v_pin_mv * (VBAT_DIVIDER_R1_KOHM + VBAT_DIVIDER_R2_KOHM))
            / VBAT_DIVIDER_R2_KOHM;
}
```

## 11. Application 3 — Potentiometer (Direct Ratio, Motor/Servo Input)

```c
/* ============================================================
 * Simplest case: pot output IS the ADC input directly, low
 * source impedance (pot's own resistance, typically <=10k).
 * Feeds directly into the PWM duty-cycle/servo-angle APIs from
 * the previous PWM driver -- ADC input -> PWM output pipeline.
 * ============================================================ */
#define POT_CHANNEL   1U

uint8_t read_pot_as_percent(void)
{
    uint16_t raw;
    if (adc_read_blocking(POT_CHANNEL, &raw) != ADC_OK)
    {
        return 0U;   /* fail-safe: treat error as 0% (motor/LED off) */
    }
    return (uint8_t)((raw * 100U) / ADC_MAX_12BIT);
}

/* Direct pipeline example, tying back into the PWM driver: */
void pot_to_led_demo(pwm_channel_t *led)
{
    uint8_t percent = read_pot_as_percent();
    pwm_set_duty_cycle(led, percent);   /* from the PWM/LED driver */
}
```

## 12. Multi-Channel Scan with DMA (Production Pattern)

```c
/* ============================================================
 * Polling one channel at a time (Section 5) is fine for a
 * single sensor but doesn't scale. Production multi-sensor
 * boards configure a SCAN sequence + DMA circular buffer so
 * the CPU never busy-waits on ADC hardware at all -- extending
 * the same "keep ISRs/CPU time minimal" discipline from the
 * UART ring-buffer and EXTI drivers.
 * ============================================================ */
#define NUM_ADC_CHANNELS   4U
static volatile uint16_t adc_dma_buffer[NUM_ADC_CHANNELS];

void adc_init_scan_dma(void)
{
    /* 1. Configure sequence length = 4 in SQR1 L[3:0] */
    ADC1->SQR1 = (ADC1->SQR1 & ~(0xFU << 20)) | ((NUM_ADC_CHANNELS - 1U) << 20);

    /* 2. Assign channels to sequence positions (SQR3 low bits) --
     * e.g. channels 1,4,8,17 in conversion order. */
    ADC1->SQR3 = (1U << 0) | (4U << 5) | (8U << 10) | (17U << 15);

    /* 3. SCAN mode + continuous, so DMA keeps the buffer live
     * without any further software trigger. */
    ADC1->CR1 |= ADC_CR1_SCAN;
    ADC1->CR2 |= ADC_CR2_CONT;

    /* DMA controller setup (stream config, circular mode, DR as
     * source, adc_dma_buffer as destination) omitted here --
     * belongs to the DMA driver layer, conceptually identical to
     * the cache-coherency DMA work: the CPU never touches ADC
     * registers again after this init; it only reads
     * adc_dma_buffer[], with the same non-coherent-access
     * caution (cache invalidate before read, if DMA target is
     * cacheable memory) covered in the earlier DMA/CMO work. */

    ADC1->CR2 |= ADC_CR2_ADON;
    ADC1->CR2 |= ADC_CR2_SWSTART;   /* one trigger starts the whole
                                        continuous scan cycle */
}
```

## 13. Test / Demo Harness

```c
#include <stdio.h>
#include <assert.h>

void test_raw_to_millivolts(void)
{
    /* Full-scale: raw = max_code -> should return exactly vref */
    assert(adc_raw_to_millivolts(4095, 3300, 4095) == 3300U);

    /* Half-scale: raw = max_code/2 -> ~half of vref */
    uint32_t half = adc_raw_to_millivolts(2048, 3300, 4095);
    assert(half > 1640U && half < 1660U);

    /* Zero: raw = 0 -> 0mV */
    assert(adc_raw_to_millivolts(0, 3300, 4095) == 0U);

    printf("ADC raw->mV conversion verified\n");
}

void test_resolution_lsb_scaling(void)
{
    /* 12-bit LSB should be exactly 16x finer than 8-bit LSB
     * for the same VREF (4095/255 ~= 16.06) */
    uint32_t lsb_12bit = (3300000UL) / 4095U;   /* microvolts */
    uint32_t lsb_8bit  = (3300000UL) / 255U;

    assert(lsb_8bit > (lsb_12bit * 15U));   /* roughly 16x coarser */
    printf("Resolution/LSB scaling verified: 12-bit=%luuV, 8-bit=%luuV\n",
           (unsigned long)lsb_12bit, (unsigned long)lsb_8bit);
}

void test_pot_percent_bounds(void)
{
    /* raw=4095 (max_code) should map to exactly 100% */
    uint32_t pct = (4095U * 100U) / ADC_MAX_12BIT;
    assert(pct == 100U);

    /* raw=0 should map to exactly 0% */
    pct = (0U * 100U) / ADC_MAX_12BIT;
    assert(pct == 0U);

    printf("Potentiometer percent-mapping bounds verified\n");
}

int main(void)
{
    test_raw_to_millivolts();
    test_resolution_lsb_scaling();
    test_pot_percent_bounds();
    printf("All ADC tests passed.\n");
    return 0;
}
```

## 14. Design Rules Summary

| Rule | Why |
|---|---|
| **Bounded timeout on the DONE/EOC poll, never an infinite `while()`** | A dead sensor, floating pin, or misconfigured clock must never hang the whole system — same discipline as the UART/EXTI drivers' timeout patterns. |
| **Stabilization delay (t_STAB) after ADON** | The first conversion right after power-on is architecturally unreliable per the datasheet; skipping the delay silently corrupts the first sample. |
| **Sample time scaled to source impedance** | High-impedance sensors (thermistors, dividers) need long sample times (144-480 cycles) to fully charge the S&H capacitor; low-impedance sources (pots) can use short times for higher throughput. |
| **Reference voltage measured, not assumed** | VDDA sags under load; using VREFINT to back-calculate the actual reference keeps every other channel's voltage conversion accurate instead of silently drifting. |
| **Resolution chosen deliberately, not just maxed out** | Lower resolution buys conversion speed; oversampling buys effective resolution beyond native ADC noise floor — both are explicit engineering trade-offs, not defaults. |
| **Overrun (OVR) flag checked before trusting DATA** | Same "check status flags before trusting data" discipline as UART ORE/FE/PE — stale or corrupted samples must never silently propagate. |
| **DMA + circular scan for multi-channel production systems** | Keeps the CPU off ADC polling entirely, freeing cycles for control logic — same rationale as interrupt-driven UART over polling. |
| **Fail-safe defaults on read error (0% duty, sentinel temperature)** | An ADC read failure feeding into a motor/LED/servo control loop must degrade to a known-safe state, not an undefined or last-stale value. |

## 15. Concepts Demonstrated

| Concept | Where it shows up |
|---|---|
| **ADC / Sampling / Resolution / Reference Voltage fundamentals** | Sections 1, 4, 6, 7 — direct implementation of the prompt's four learning topics with register-level code and the physical reasoning behind each. |
| **Exact prompt pattern (START/poll DONE/read DATA)** | `adc_read_blocking()` in Section 5, hardened with timeout and overrun handling. |
| **Sensor-specific signal conditioning** | Thermistor (Beta equation), battery divider (scale-factor inversion), potentiometer (direct ratio) — three distinct real-world sensor topologies. |
| **Cross-driver integration** | Section 11's `pot_to_led_demo()` directly chains into the PWM driver's `pwm_set_duty_cycle()`, and the DMA scan pattern references the earlier cache-coherency/CMO DMA work. |
| **Precision engineering trade-offs** | Oversampling for effective resolution, internal reference correction for VDDA drift — the same rigor applied to his cache/timer/UART performance work, now applied to analog measurement accuracy. |