# Bit Manipulation for ARM Embedded Interviews ⭐

## 1. Concept Map

```
Single bit ops        Multi-bit field ops       Bit-counting/scanning       ARM-specific
   SET/CLEAR/TOGGLE       MASK + SHIFT              popcount (Brian            CLZ/RBIT instr.
   /READ                  read-modify-write         Kernighan's trick)         Bit-banding
        |                       |                         |                        |
        +-----------------------+-------------------------+------------------------+
                                       |
                            Register-level correctness:
                       volatile, atomicity, RMW hazards,
                         signed vs unsigned shift UB
```

## 2. Hardened Macro Library (fixing hidden bugs in the baseline)

```c
/* ============================================================
 * bitops.h -- production-grade bit manipulation primitives
 *
 * The baseline macros work for the textbook case but have three
 * classic interview-trap bugs:
 *
 *   1. No type safety -- (1U << bit) on a bit >= 32 is UB for a
 *      32-bit reg, and silently wrong for a 64-bit reg (shifting
 *      a 32-bit literal into a 64-bit register truncates).
 *
 *   2. Macro hygiene -- unparenthesized arguments can break with
 *      expressions, e.g. SET_BIT(reg, x+1) with a badly written
 *      macro would expand incorrectly. The originals ARE mostly
 *      safe here, but multiple evaluation of `bit` if it were a
 *      function call with side effects (bit++) would fire twice.
 *
 *   3. No width parameterization -- can't reuse for uint8_t/
 *      uint16_t/uint64_t registers without rewriting.
 * ============================================================ */

#ifndef BITOPS_H
#define BITOPS_H

#include <stdint.h>

/* --- Width-generic single-bit macros ---
 * Casting the literal to the same type as `reg` prevents silent
 * truncation on 64-bit registers and matches MISRA-C's essential
 * type rules (a very common ARM embedded coding standard check). */

#define SET_BIT(reg, bit)     ((reg) |=  (typeof(reg))(1ULL << (bit)))
#define CLEAR_BIT(reg, bit)   ((reg) &= ~(typeof(reg))(1ULL << (bit)))
#define TOGGLE_BIT(reg, bit)  ((reg) ^=  (typeof(reg))(1ULL << (bit)))
#define READ_BIT(reg, bit)    (((reg) >> (bit)) & (typeof(reg))1U)

/* typeof() is a GCC/Clang extension -- extremely common in Linux
 * kernel and embedded codebases, but NOT standard C. If strict
 * ISO C is required (e.g. MISRA pedantic mode), provide explicit
 * width variants instead: */

#define SET_BIT32(reg, bit)    ((reg) |=  (uint32_t)(1U << (bit)))
#define CLEAR_BIT32(reg, bit)  ((reg) &= ~(uint32_t)(1U << (bit)))
#define TOGGLE_BIT32(reg, bit) ((reg) ^=  (uint32_t)(1U << (bit)))
#define READ_BIT32(reg, bit)   (((reg) >> (bit)) & 1U)

#define SET_BIT64(reg, bit)    ((reg) |=  (uint64_t)(1ULL << (bit)))
#define CLEAR_BIT64(reg, bit)  ((reg) &= ~(uint64_t)(1ULL << (bit)))

#endif /* BITOPS_H */
```

```c
/* ============================================================
 * Baseline program, now using the hardened macros, plus the
 * classic interview follow-up: "what if reg is a hardware
 * register, not a plain variable?"
 * ============================================================ */

#include <stdint.h>
#include "bitops.h"

int main(void)
{
    uint32_t reg = 0;

    SET_BIT32(reg, 3);      /* reg = 0x00000008 */
    CLEAR_BIT32(reg, 3);    /* reg = 0x00000000 */
    TOGGLE_BIT32(reg, 5);   /* reg = 0x00000020 */

    if (READ_BIT32(reg, 5))
    {
        /* Bit is set */
    }

    /* INTERVIEW TRAP: if `reg` were declared
     *   volatile uint32_t *GPIO_ODR = (volatile uint32_t*)0x40020014;
     * then SET_BIT32(*GPIO_ODR, 3) expands to a read-modify-write:
     *   *GPIO_ODR = *GPIO_ODR | (1U << 3);
     * This is NOT atomic. An interrupt firing between the read and
     * write can clobber another bit set by the ISR -- exactly the
     * lost-update race discussed in the GPIO output/BSRR writeup.
     * Correct answer: use a hardware atomic set/clear register
     * (BSRR-style) or disable interrupts around the RMW, never
     * plain |=/&= on a shared hardware register from both main
     * context and ISR context. */

    while (1) { /* ... */ }
}
```

## 3. Multi-Bit Field Operations (mask + shift, the real interview meat)

```c
/* ============================================================
 * Real registers rarely have single control bits -- they have
 * multi-bit FIELDS (e.g. STM32 GPIO MODER: 2 bits per pin,
 * ARM SCTLR: assorted 1-3 bit fields). This is the pattern
 * interviewers actually probe for.
 * ============================================================ */

#include <stdint.h>

/* Generic width-aware field write:
 *   width  = number of bits in the field
 *   pos    = starting bit position (LSB of the field)
 * Always clear-then-set -- never assume the field starts at 0. */
static inline void write_field(volatile uint32_t *reg,
                                uint32_t value,
                                uint32_t width,
                                uint32_t pos)
{
    uint32_t mask = ((1U << width) - 1U) << pos;

    /* Clear the field, then OR in the new value.
     * value is masked defensively in case caller passes an
     * out-of-range value (e.g. width=2 but value=5 -> only
     * low 2 bits are kept, matching hardware truncation behavior
     * rather than silently corrupting adjacent fields). */
    *reg = (*reg & ~mask) | ((value << pos) & mask);
}

static inline uint32_t read_field(volatile uint32_t *reg,
                                   uint32_t width,
                                   uint32_t pos)
{
    uint32_t mask = (1U << width) - 1U;
    return (*reg >> pos) & mask;
}

/* --- Worked example: STM32-style GPIO MODER, 2 bits per pin ---
 * This is the exact pattern from the GPIO output writeup,
 * generalized into a reusable primitive instead of a one-off
 * "clear then set" inline sequence. */
#define MODER_WIDTH   2U

static inline void gpio_set_mode(volatile uint32_t *moder,
                                  uint32_t pin, uint32_t mode)
{
    write_field(moder, mode, MODER_WIDTH, pin * MODER_WIDTH);
}
```

## 4. Bit-Counting and Scanning Tricks (asked in nearly every ARM interview)

```c
#include <stdint.h>

/* --- 1. Isolate the lowest set bit (LSB) ---
 * Classic two's-complement identity: x & (-x) leaves only the
 * lowest set bit. Used in bitmask iteration (e.g. walking a set
 * of pending interrupt flags one at a time). */
static inline uint32_t isolate_lsb(uint32_t x)
{
    return x & (uint32_t)(-(int32_t)x);
}

/* --- 2. Clear the lowest set bit ---
 * x & (x-1). Also the basis of Brian Kernighan's popcount, and
 * a fast is-power-of-2 test when combined with x != 0. */
static inline uint32_t clear_lsb(uint32_t x)
{
    return x & (x - 1U);
}

/* --- 3. Power-of-two check ---
 * A power of two has exactly one bit set. x & (x-1) clears that
 * one bit, so the result is 0 iff x was a power of two (and x != 0
 * excludes the degenerate case where x=0 would falsely pass). */
static inline int is_power_of_two(uint32_t x)
{
    return (x != 0U) && ((x & (x - 1U)) == 0U);
}

/* --- 4. Brian Kernighan popcount ---
 * O(number of set bits) rather than O(width) -- better than a
 * naive 32-iteration shift-and-test loop when the value is sparse
 * (e.g. counting active channels in a 32-sensor Fenwick-tree
 * bitmask from the sensor-aggregation work). */
static inline uint32_t popcount_kernighan(uint32_t x)
{
    uint32_t count = 0;
    while (x)
    {
        x &= (x - 1U);   /* clear lowest set bit each iteration */
        count++;
    }
    return count;
}

/* --- 5. Hardware popcount / CLZ ---
 * Production ARM code should prefer compiler builtins over manual
 * loops -- GCC/Clang lower these to a single CLZ (Count Leading
 * Zeros) instruction on ARMv7+/AArch64, or RBIT+CLZ for trailing
 * zeros. This is the answer interviewers are fishing for when they
 * ask "how would you do this in ONE instruction on ARM?" */
static inline uint32_t popcount_hw(uint32_t x)
{
    return (uint32_t)__builtin_popcount(x);
}

static inline int count_leading_zeros(uint32_t x)
{
    /* __builtin_clz(0) is UB -- always guard the zero case.
     * Maps directly to the ARM `CLZ` instruction. */
    return (x == 0U) ? 32 : __builtin_clz(x);
}

static inline int count_trailing_zeros(uint32_t x)
{
    /* Maps to `RBIT` (bit-reverse) followed by `CLZ` on ARM,
     * or a native CTZ where available. */
    return (x == 0U) ? 32 : __builtin_ctz(x);
}

/* --- 6. Find index of highest/lowest set bit ---
 * Directly useful for priority encoders -- e.g. picking the
 * highest-priority pending interrupt out of an NVIC-style bitmask. */
static inline int find_highest_set_bit(uint32_t x)
{
    return (x == 0U) ? -1 : (31 - __builtin_clz(x));
}

static inline int find_lowest_set_bit(uint32_t x)
{
    return (x == 0U) ? -1 : __builtin_ctz(x);
}
```

## 5. Bit Reversal, Byte Swap, and Rotate (endianness + ARM instruction mapping)

```c
#include <stdint.h>

/* --- Rotate left/right ---
 * The idiomatic form below is UB-free and what compilers
 * recognize as a rotate-instruction pattern (ARM `ROR`), unlike
 * naive (x << n) | (x >> (32-n)) which is UB when n == 0. */
static inline uint32_t rotl32(uint32_t x, uint32_t n)
{
    n &= 31U;   /* defend against n >= 32, which is UB for shifts */
    return (n == 0U) ? x : (x << n) | (x >> (32U - n));
}

static inline uint32_t rotr32(uint32_t x, uint32_t n)
{
    n &= 31U;
    return (n == 0U) ? x : (x >> n) | (x << (32U - n));
}

/* --- Byte swap (endianness conversion) ---
 * Relevant to the device-tree (FDT/DTB) parsing work: DTB is
 * always big-endian on the wire, and this is exactly the
 * primitive used to convert to the host's native (usually
 * little-endian ARM) representation. */
static inline uint32_t byteswap32(uint32_t x)
{
    return ((x >> 24) & 0x000000FFU) |
           ((x >>  8) & 0x0000FF00U) |
           ((x <<  8) & 0x00FF0000U) |
           ((x << 24) & 0xFF000000U);
}

/* Production code should prefer the compiler builtin, which
 * lowers to a single `REV` instruction on ARM: */
static inline uint32_t byteswap32_hw(uint32_t x)
{
    return __builtin_bswap32(x);
}

/* --- Full bit reversal ---
 * Maps to a single `RBIT` instruction on ARMv6T2+/AArch64.
 * The software fallback (divide-and-conquer swap) is a classic
 * whiteboard question: */
static inline uint32_t bit_reverse32(uint32_t x)
{
    x = ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
    x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
    x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
    x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}
```

## 6. ARM-Specific: Bit-Banding (Cortex-M) — Atomic Single-Bit Access

```c
/* ============================================================
 * Cortex-M3/M4 bit-banding: maps every bit in a 1MB "bit-band
 * region" to a full 32-bit word in a 32MB "alias region", so a
 * single-bit set/clear becomes a single atomic STR instruction
 * with NO read-modify-write at all -- solving the exact RMW
 * hazard flagged in Section 2 without needing BSRR-style
 * dedicated set/clear registers.
 *
 * Formula (from the ARMv7-M Architecture Reference Manual):
 *   alias_addr = alias_base + (byte_offset * 32) + (bit_number * 4)
 * ============================================================ */

#include <stdint.h>

#define SRAM_BASE        0x20000000U
#define SRAM_BB_BASE     0x22000000U   /* SRAM bit-band alias region */

#define PERIPH_BASE      0x40000000U
#define PERIPH_BB_BASE   0x42000000U   /* Peripheral bit-band alias  */

static inline volatile uint32_t *bitband_periph(uint32_t addr, uint32_t bit)
{
    uint32_t byte_offset = addr - PERIPH_BASE;
    uint32_t bb_addr = PERIPH_BB_BASE + (byte_offset * 32U) + (bit * 4U);
    return (volatile uint32_t *)bb_addr;
}

/* Usage: instead of a masked RMW on GPIO_ODR, write through the
 * bit-band alias -- this is a single atomic bus transaction,
 * immune to interrupt preemption between read and write. */
#define GPIO_ODR_ADDR   0x40020014U

static inline void led_set_atomic(uint32_t pin)
{
    *bitband_periph(GPIO_ODR_ADDR, pin) = 1U;
}

static inline void led_clear_atomic(uint32_t pin)
{
    *bitband_periph(GPIO_ODR_ADDR, pin) = 0U;
}

/* INTERVIEW NOTE: bit-banding is Cortex-M3/M4-specific and was
 * DROPPED in Cortex-M7/M33/M55 and all Cortex-A/AArch64 cores.
 * On those, the equivalent atomic single-bit idiom is either a
 * dedicated set/clear register (BSRR pattern) or LDREX/STREX
 * (Cortex-A, exclusive monitor) / native atomic instructions
 * (ARMv8.1 LSE: e.g. `STSET`, `STCLR`). Knowing this generational
 * distinction is a strong signal in an ARM-specific interview. */
```

## 7. LDREX/STREX — True Atomic Bit-Set on Cortex-A / AArch64

```c
/* ============================================================
 * For cores without bit-banding, atomic read-modify-write uses
 * the exclusive monitor: LDREX loads and "tags" the address,
 * STREX writes only if no other observer touched it since, and
 * reports success/failure. This is the same primitive underlying
 * atomic_fetch_or() and Linux kernel set_bit(), and directly
 * relevant to the lock-free algorithm / memory-ordering work.
 * ============================================================ */

static inline void atomic_set_bit_arm(volatile uint32_t *addr, uint32_t bit)
{
    uint32_t tmp, status;
    do {
        __asm volatile (
            "ldrex   %0, [%2]      \n"   /* tmp = *addr, mark exclusive */
            "orr     %0, %0, %3    \n"   /* tmp |= (1 << bit)           */
            "strex   %1, %0, [%2]  \n"   /* try to store; status=0 on OK*/
            : "=&r" (tmp), "=&r" (status)
            : "r" (addr), "r" (1U << bit)
            : "memory"
        );
    } while (status != 0U);   /* retry on contention (STREX failed) */
}

/* On AArch64 with ARMv8.1 LSE extensions, this collapses to a
 * single instruction: `stset` (atomic bit-set), eliminating the
 * retry loop entirely -- a good "what's newer/better" follow-up
 * answer if asked to optimize the LDREX/STREX version. */
```

## 8. Interview Quick-Reference Table

| Task | One-liner | ARM instruction it maps to |
|---|---|---|
| Set bit `n` | `x \|= (1U << n)` | `ORR` |
| Clear bit `n` | `x &= ~(1U << n)` | `BIC` |
| Toggle bit `n` | `x ^= (1U << n)` | `EOR` |
| Test bit `n` | `(x >> n) & 1` | `TST`/`LSR` |
| Clear lowest set bit | `x & (x-1)` | `AND` |
| Isolate lowest set bit | `x & (-x)` | `AND`/`RSB` |
| Is power of 2 | `x && !(x & (x-1))` | — |
| Count set bits | `__builtin_popcount(x)` | `VCNT` (NEON) / software |
| Leading zeros | `__builtin_clz(x)` | `CLZ` |
| Trailing zeros | `__builtin_ctz(x)` | `RBIT` + `CLZ` |
| Byte swap | `__builtin_bswap32(x)` | `REV` |
| Bit reverse | manual divide-and-conquer | `RBIT` |
| Rotate | `(x<<n)\|(x>>(32-n))` | `ROR` |
| Atomic single-bit set (M3/M4) | bit-band alias write | plain `STR` (aliased) |
| Atomic RMW (Cortex-A/A64) | `LDREX`/`STREX` loop | `LDREX`/`STREX`, or `STSET` on ARMv8.1+ |

## 9. Key Takeaways

1. **The four baseline macros are correct in intent but fragile in practice** — always parenthesize, always cast to the target register width, and never assume they're atomic on a real hardware register.
2. **Multi-bit fields (mask+shift+clear-then-set) are the pattern actually tested**, not isolated single-bit macros — MODER-style 2-bit fields are a recurring interview scenario.
3. **`x & (x-1)` and `x & (-x)` are the two identities to have memorized cold** — they underpin popcount, power-of-2 checks, and priority-encoder bit-walking.
4. **Prefer compiler builtins (`__builtin_clz/ctz/popcount/bswap`) over hand-rolled loops** in production code — they map to single ARM instructions (`CLZ`, `RBIT`, `REV`) and are both faster and less error-prone.
5. **Know the atomicity story for your specific core**: bit-banding (Cortex-M3/M4 only) vs. dedicated BSRR-style registers vs. `LDREX`/`STREX` (Cortex-A/AArch64) vs. ARMv8.1 LSE single-instruction atomics — this generational awareness is a strong differentiator in ARM-specific interviews.