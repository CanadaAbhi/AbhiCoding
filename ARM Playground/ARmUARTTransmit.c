# UART Transmit — Register-Level Polling Driver with Baud Rate Configuration

## 1. Concept Map

```
CPU writes byte
   |
UART->DATA (Transmit Data Register, TDR)
   |
Transmit Shift Register (hardware serializes MSB/LSB per config)
   |
TX pin -> asynchronous serial line (start bit, 8 data bits, [parity], stop bit(s))
   |
STATUS register flags (TXE/TC) report progress back to software
```

Two distinct "done" signals exist and conflating them is the most common UART bug:
- **TXE (Transmit Data Register Empty)**: the `DATA` register has been copied into the shift register — safe to write the *next* byte, but the *current* byte is still physically being shifted out on the wire.
- **TC (Transmission Complete)**: the shift register has fully drained and the stop bit has been sent — the line is truly idle. Required before disabling the UART or entering a sleep mode, otherwise the last byte gets truncated.

## 2. Register-Level Definitions

```c
/* ============================================================
 * STM32F4-class USART register map (illustrative addresses --
 * exact bit positions vary slightly by vendor, but the TXE/TC/
 * RXNE distinction and BRR-based baud generation are universal
 * across ARM Cortex-M UART/USART peripherals).
 * ============================================================ */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    volatile uint32_t SR;    /* 0x00: Status Register
                                 bit 5 RXNE - Receive data register not empty
                                 bit 6 TC   - Transmission Complete
                                 bit 7 TXE  - Transmit data register Empty
                                 bit 3 ORE  - Overrun error
                                 bit 1 FE   - Framing error
                                 bit 0 PE   - Parity error            */
    volatile uint32_t DR;    /* 0x04: Data Register -- write = TDR (load
                                 shift register), read = RDR (received byte).
                                 Same address, two different physical
                                 registers muxed by direction of access. */
    volatile uint32_t BRR;   /* 0x08: Baud Rate Register
                                 [15:4] DIV_Mantissa, [3:0] DIV_Fraction */
    volatile uint32_t CR1;   /* 0x0C: Control Register 1
                                 bit 13 UE  - USART Enable
                                 bit 3  TE  - Transmitter Enable
                                 bit 2  RE  - Receiver Enable
                                 bit 5  RXNEIE - RX interrupt enable
                                 bit 12 M   - Word length (0=8b,1=9b)
                                 bit 10 PCE - Parity control enable    */
    volatile uint32_t CR2;   /* 0x10: stop bits, etc. */
    volatile uint32_t CR3;   /* 0x14: flow control, DMA enable */
} USART_TypeDef;

#define USART2   ((USART_TypeDef *)0x40004400U)

#define USART_SR_PE     (1U << 0)
#define USART_SR_FE     (1U << 1)
#define USART_SR_ORE    (1U << 3)
#define USART_SR_RXNE   (1U << 5)
#define USART_SR_TC     (1U << 6)
#define USART_SR_TXE    (1U << 7)

#define USART_CR1_RE    (1U << 2)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_UE    (1U << 13)
```

## 3. Baud Rate Generation

```c
/* ============================================================
 * BRR encodes a fractional divider: actual_baud = f_CK / (16 * USARTDIV)
 * for the standard (non-oversampling-by-8) mode. USARTDIV is a
 * fixed-point value with a 4-bit fractional part, so:
 *
 *   USARTDIV = f_CK / (16 * desired_baud)
 *
 * Software computes this in 100ths to preserve the fraction
 * without floating point (float is often unavailable/expensive
 * on a bare-metal Cortex-M0/M3 build without an FPU).
 * ============================================================ */

static void usart_set_baud(USART_TypeDef *u, uint32_t pclk_hz, uint32_t baud)
{
    /* Fixed-point scale by 100 to retain 2 fractional digits
     * without floating point. */
    uint32_t usartdiv_x100 = (25U * pclk_hz) / (4U * baud);

    uint32_t mantissa = usartdiv_x100 / 100U;
    uint32_t fraction_x100 = usartdiv_x100 - (mantissa * 100U);

    /* Fraction field is 4 bits -> 16 steps -> round to nearest
     * of 16ths rather than truncating, to minimize baud error. */
    uint32_t fraction = ((fraction_x100 * 16U) + 50U) / 100U;

    if (fraction > 0x0FU)
    {
        /* Rounding overflowed into the mantissa */
        mantissa += 1U;
        fraction = 0U;
    }

    u->BRR = (uint32_t)((mantissa << 4U) | (fraction & 0x0FU));

    /* NOTE ON BAUD ERROR: with typical PCLK values (e.g. 16MHz,
     * 42MHz, 84MHz) at common baud rates (9600, 115200), residual
     * error after this rounding is usually < 1-2%, well within the
     * ~3% total budget (transmitter + receiver clock error +
     * sampling point) that async serial can tolerate before bit
     * errors appear near the end of each byte's stop bit. Always
     * verify tolerance for non-standard PCLK/baud combinations --
     * this is a real, measurable hardware constraint, not just a
     * theoretical rounding nuance. */
}
```

## 4. Initialization

```c
static void uart_init(USART_TypeDef *u, uint32_t pclk_hz, uint32_t baud)
{
    /* Peripheral clock (RCC) enable is assumed done by the caller/
     * board-init before this runs -- writing to USART registers on
     * an unclocked peripheral bus produces a bus fault, same
     * discipline as the RCC-before-GPIO note in the earlier
     * bare-metal GPIO work. */

    u->CR1 &= ~USART_CR1_UE;      /* disable while reconfiguring to
                                      avoid transmitting garbage
                                      mid-configuration */

    usart_set_baud(u, pclk_hz, baud);

    u->CR1 |= (USART_CR1_TE | USART_CR1_RE);  /* enable transmitter
                                                   and receiver */
    u->CR1 |= USART_CR1_UE;                    /* enable peripheral
                                                   LAST, after mode
                                                   bits and baud are
                                                   settled */
}
```

## 5. TX Path — Polling Driver (as specified)

```c
/* ============================================================
 * uart_send_char: block on TXE, then load DATA.
 * ============================================================ */
void uart_send_char(char c)
{
    while (!(USART2->SR & USART_SR_TXE))
    {
        /* Busy-wait: shift register is still draining the
         * PREVIOUS byte. This is a spin-loop, not a sleep --
         * acceptable for low-rate diagnostic/console UART, but
         * see Section 8 for why this is unacceptable in a path
         * that shares the CPU with real-time work. */
    }

    USART2->DR = (uint8_t)c;   /* writing DR clears TXE automatically
                                   (hardware side-effect of the write,
                                   not something software clears
                                   explicitly -- unlike EXTI->PR) */
}

/* ============================================================
 * uart_send_string: sequential polling send, byte by byte.
 * ============================================================ */
void uart_send_string(const char *str)
{
    while (*str)
    {
        uart_send_char(*str++);
    }
}

/* ============================================================
 * uart_flush: block until the FULL frame (including stop bit)
 * has left the shift register -- required before power-down,
 * UART disable, or switching TX pin to GPIO for another purpose.
 * TXE alone is insufficient here: TXE clears the instant DATA is
 * copied into the shift register, while the actual bits are
 * still being clocked out onto the wire.
 * ============================================================ */
void uart_flush(USART_TypeDef *u)
{
    while (!(u->SR & USART_SR_TC))
    {
        /* Wait for true idle-line completion */
    }
}
```

## 6. RX Path — Polling Receive with Error Flag Handling

```c
/* ============================================================
 * uart_recv_char: block on RXNE, then read DATA. Reading DR
 * clears RXNE as a hardware side-effect of the READ (mirror of
 * TXE clearing on WRITE) -- another register that behaves
 * unlike a plain memory location.
 * ============================================================ */
int uart_recv_char(USART_TypeDef *u, char *out)
{
    while (!(u->SR & USART_SR_RXNE))
    {
        /* Wait for a byte to arrive */
    }

    /* Check error flags BEFORE consuming DR: on this family,
     * SR must be read before DR to correctly clear ORE/FE/PE
     * per the datasheet's documented clear sequence (read SR,
     * then read DR). Skipping this ordering can leave a stale
     * error flag latched, silently blocking future RXNE events. */
    uint32_t status = u->SR;

    *out = (char)u->DR;

    if (status & USART_SR_ORE)
    {
        /* Overrun: a new byte arrived before the previous one
         * was read out of DR -- the new byte silently overwrote
         * it. This is the single biggest argument for moving RX
         * to interrupts (Section 8): a polling loop that is busy
         * elsewhere WILL eventually overrun on any nontrivial
         * incoming data rate. */
        return -1;
    }
    if (status & (USART_SR_FE | USART_SR_PE))
    {
        /* Framing/parity error: baud mismatch, noise on the line,
         * or a wiring/ground issue -- treat the byte as suspect. */
        return -2;
    }

    return 0;
}
```

## 7. Usage

```c
int main(void)
{
    /* Assume RCC has already enabled the USART2 and GPIO clocks,
     * and GPIO AF (alternate function) mode has been configured
     * for the TX/RX pins -- omitted here, same "clock/config
     * before enable" discipline as the GPIO/EXTI driver. */

    uart_init(USART2, 16000000U /* PCLK */, 115200U /* baud */);

    uart_send_string("Hello ARM!\r\n");
    uart_flush(USART2);   /* ensure the string is fully on the wire
                              before, e.g., proceeding to a low-power
                              sleep or a peripheral reconfiguration */

    while (1)
    {
        char c;
        if (uart_recv_char(USART2, &c) == 0)
        {
            uart_send_char(c);   /* simple echo */
        }
    }
}
```

## 8. Polling vs. Interrupt vs. DMA — Trade-offs

| Approach | TX behavior | RX behavior | Best for |
|---|---|---|---|
| **Polling** (this driver) | CPU spins on TXE per byte, blocking all other work | CPU spins on RXNE; **any** delay risks ORE overrun | Simple diagnostic/boot-time console, low data rate, no concurrent real-time work |
| **Interrupt-driven** | `TXE` interrupt refills DATA from a ring buffer; CPU free between bytes | `RXNE` interrupt drains DATA into a ring buffer the instant it arrives, eliminating overrun risk from CPU busy-elsewhere | General-purpose logging/console shared with real-time tasks — mirrors the EXTI/NVIC discipline already used for GPIO input |
| **DMA** | DMA engine streams a buffer to DATA with zero per-byte CPU involvement, IRQ only on buffer-complete | DMA streams DATA into a buffer continuously; CPU only notified on completion/idle-line timeout | High-throughput or CPU-budget-constrained links, matches the sensor-to-display and DMA/cache-coherency discipline elsewhere in this portfolio — but requires the same `DC CVAC`/`DC IVAC` cache maintenance ops for non-coherent DMA buffers |

## 9. Common Pitfalls Checklist

| Pitfall | Symptom | Fix |
|---|---|---|
| Confusing TXE with TC | Disabling UART or entering sleep truncates the last byte on the wire | Use `TC`, not `TXE`, before disable/power-down |
| Not checking `ORE` on RX | Silent data loss under any burst RX rate while CPU is busy elsewhere | Check `SR` before reading `DR`; move RX to interrupts if overrun-prone |
| Wrong SR/DR read order for error clear | `ORE`/`FE`/`PE` flags stay latched, RXNE stops updating, RX appears to "hang" | Read `SR` then `DR` in that order, per datasheet-documented clear sequence |
| Floating-point baud calculation on FPU-less core | Slow, or link-fails on some minimal bare-metal toolchains | Fixed-point (`x100`) integer arithmetic as shown in `usart_set_baud` |
| Forgetting `UE` (USART Enable) or enabling it before config settles | No output at all, or garbage first byte(s) | Configure baud/mode first, enable `UE` last |
| Busy-wait polling on a path shared with real-time deadlines | Violates `<100us` interrupt latency budgets elsewhere in the system; blocks higher-priority work | Escalate to interrupt-driven TX/RX with ring buffers, or DMA for high throughput |
| Writing `DATA` without checking `TXE` at all | Overwrites a byte still pending in the transmit register, corrupting the output stream | Always gate the write on `TXE`, as in `uart_send_char` |

## 10. Key Takeaways

1. **TXE and TC are not interchangeable** — TXE says "safe to load the next byte," TC says "the wire is truly idle." Using the wrong one for shutdown/sleep sequencing is the most common UART bug.
2. **DATA (`DR`) has hardware side-effects on read and write** — writing clears TXE, reading clears RXNE — the same pattern as EXTI's write-1-to-clear `PR`, where a peripheral register does not behave like plain memory.
3. **Baud rate is a fixed-point divider computation** (`BRR` mantissa + fraction), and residual rounding error is a real, bounded hardware constraint — not something to hand-wave with floating point on FPU-less cores.
4. **Polling is only safe when nothing else competes for the CPU** — RX polling in particular risks silent `ORE` overrun the instant the loop is busy elsewhere, which is the direct justification for escalating to interrupt-driven or DMA-based UART in any system with concurrent real-time obligations, following the same NVIC/ISR discipline established for GPIO/EXTI.
5. **Initialization order matters**: peripheral clock enable → configure baud/mode with `UE` disabled → enable `TE`/`RE` → enable `UE` last, mirroring the "clock and config before enable" discipline used throughout this bare-metal driver series.