Here's a production-grade **DMA-driven UART Transfer** implementation for ARM Cortex-M (STM32F4-class), extending the memory-mapped UART driver to offload byte-copying entirely to the DMA controller. This eliminates CPU involvement in the data-movement path (`Memory -> DMA -> UART`) and is the natural evolution beyond polling and interrupt-driven ring buffers.

## 1. DMA Controller Register Overlay (CMSIS-style)

```c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ---- DMA Controller (DMA1) base and stream layout ---- */
typedef struct {
    volatile uint32_t CR;      // Configuration register
    volatile uint32_t NDTR;    // Number of data register (16-bit, auto-decrements)
    volatile uint32_t PAR;     // Peripheral address register
    volatile uint32_t M0AR;    // Memory address 0 (primary buffer)
    volatile uint32_t M1AR;    // Memory address 1 (double-buffer mode)
    volatile uint32_t FCR;     // FIFO control register
} DMA_Stream_TypeDef;

typedef struct {
    volatile uint32_t LISR;    // Low interrupt status  (streams 0-3)
    volatile uint32_t HISR;    // High interrupt status (streams 4-7)
    volatile uint32_t LIFCR;   // Low interrupt flag clear
    volatile uint32_t HIFCR;   // High interrupt flag clear
} DMA_TypeDef;

#define DMA1_BASE        0x40026000UL
#define DMA1             ((DMA_TypeDef *)DMA1_BASE)
#define DMA1_Stream5     ((DMA_Stream_TypeDef *)(DMA1_BASE + 0x088))  // USART2_RX
#define DMA1_Stream6     ((DMA_Stream_TypeDef *)(DMA1_BASE + 0x0A0))  // USART2_TX

#define RCC_AHB1ENR      (*(volatile uint32_t *)0x40023830UL)
#define RCC_DMA1EN       (1U << 21)

/* DMA_SxCR bit fields */
#define DMA_SxCR_EN         (1U << 0)
#define DMA_SxCR_DMEIE      (1U << 1)
#define DMA_SxCR_TEIE       (1U << 2)
#define DMA_SxCR_HTIE       (1U << 3)
#define DMA_SxCR_TCIE       (1U << 4)
#define DMA_SxCR_DIR_M2P    (1U << 6)          // 01: memory-to-peripheral
#define DMA_SxCR_DIR_P2M    (0U << 6)          // 00: peripheral-to-memory
#define DMA_SxCR_CIRC       (1U << 8)          // circular mode
#define DMA_SxCR_MINC       (1U << 10)         // memory address increment
#define DMA_SxCR_PSIZE_8B   (0U << 11)
#define DMA_SxCR_MSIZE_8B   (0U << 13)
#define DMA_SxCR_PL_HIGH    (2U << 16)         // priority level
#define DMA_SxCR_CHSEL(ch)  ((uint32_t)(ch) << 25)

/* USART3 CR3 DMA enable bits (reuse driver from earlier UART work) */
#define USART_CR3_DMAT      (1U << 7)
#define USART_CR3_DMAR      (1U << 6)
```

## 2. TX Path: Memory → DMA → UART (Non-blocking)

```c
typedef void (*dma_tx_cb_t)(void);

static volatile bool        tx_busy = false;
static volatile dma_tx_cb_t tx_complete_cb = NULL;

/*
 * Configure and arm DMA1 Stream6 (Channel4) for USART2 TX.
 * Discipline mirrors prior work: clock-before-config, disable-before-reconfigure,
 * explicit field clearing, enable-last.
 */
bool uart_dma_send_async(const uint8_t *buffer, uint16_t length, dma_tx_cb_t cb)
{
    if (tx_busy || buffer == NULL || length == 0) {
        return false;   // Reject overlapping transfers - caller must wait/queue
    }

    RCC_AHB1ENR |= RCC_DMA1EN;                 // Clock DMA controller

    DMA1_Stream6->CR &= ~DMA_SxCR_EN;           // Disable stream before touching config
    while (DMA1_Stream6->CR & DMA_SxCR_EN) { }  // EN bit clears only when FIFO/AHB drain completes

    DMA1->HIFCR = 0x3FUL << 16;                 // Clear all pending flags for stream6 (W1C)

    DMA1_Stream6->PAR  = (uint32_t)&USART2_DATA_REG;  // Fixed peripheral address
    DMA1_Stream6->M0AR = (uint32_t)buffer;             // Source buffer in memory
    DMA1_Stream6->NDTR = length;                       // Transfer count (auto-decrements)

    DMA1_Stream6->CR = DMA_SxCR_CHSEL(4)     |   // USART2_TX = channel 4 on stream6
                        DMA_SxCR_DIR_M2P     |   // memory -> peripheral
                        DMA_SxCR_MINC        |   // increment memory ptr, NOT peripheral ptr
                        DMA_SxCR_PSIZE_8B    |
                        DMA_SxCR_MSIZE_8B    |
                        DMA_SxCR_PL_HIGH     |
                        DMA_SxCR_TCIE        |   // transfer-complete interrupt
                        DMA_SxCR_TEIE;           // transfer-error interrupt

    tx_complete_cb = cb;
    tx_busy = true;

    USART2_CR3 |= USART_CR3_DMAT;                // Tell UART peripheral to pull from DMA
    DMA1_Stream6->CR |= DMA_SxCR_EN;              // Arm - CPU is now free

    return true;
}

/* Blocking wrapper — WFI while DMA drains, preserving performance/watt discipline */
void uart_dma_send_sync(const uint8_t *buffer, uint16_t length)
{
    uart_dma_send_async(buffer, length, NULL);
    while (tx_busy) {
        __asm volatile ("wfi");   // CPU sleeps; DMA + UART continue autonomously
    }
}
```

## 3. DMA Completion / Error ISR

```c
/*
 * DMA1_Stream6_IRQHandler — lives in the vector table.
 * Stream6 flags live in HISR/HIFCR bits [21..16] per RM (TCIF6=21, HTIF6=20,
 * TEIF6=19, DMEIF6=18, FEIF6=16). Always read status, act, then W1C-clear —
 * mirrors the EXTI PR discipline from the GPIO interrupt framework.
 */
void DMA1_Stream6_IRQHandler(void)
{
    uint32_t status = DMA1->HISR;

    if (status & (1U << 21)) {                 // TCIF6 — transfer complete
        DMA1->HIFCR = (1U << 21);               // W1C
        tx_busy = false;
        USART2_CR3 &= ~USART_CR3_DMAT;
        if (tx_complete_cb) {
            tx_complete_cb();                   // Deferred work — keep ISR minimal
        }
    }

    if (status & (1U << 19)) {                  // TEIF6 — transfer error (bad address/AHB fault)
        DMA1->HIFCR = (1U << 19);
        tx_busy = false;
        DMA1_Stream6->CR &= ~DMA_SxCR_EN;
        // Escalate: log, retry, or fall back to interrupt-driven ring buffer path
    }

    if (status & (1U << 18)) {                  // DMEIF6 — direct-mode error (FIFO underrun)
        DMA1->HIFCR = (1U << 18);
        tx_busy = false;
    }
}
```

## 4. RX Path: UART → DMA → Memory (Circular, Streaming)

For continuous ingestion (e.g., sensor telemetry), circular mode with half/full-transfer interrupts avoids the SPSC ring-buffer's per-byte ISR overhead entirely.

```c
#define RX_DMA_BUF_LEN   256U
static uint8_t rx_dma_buf[RX_DMA_BUF_LEN] __attribute__((aligned(4)));
static volatile uint16_t rx_last_pos = 0;

typedef void (*dma_rx_chunk_cb_t)(const uint8_t *data, uint16_t len);
static dma_rx_chunk_cb_t rx_chunk_cb = NULL;

void uart_dma_rx_start(dma_rx_chunk_cb_t cb)
{
    RCC_AHB1ENR |= RCC_DMA1EN;

    DMA1_Stream5->CR &= ~DMA_SxCR_EN;
    while (DMA1_Stream5->CR & DMA_SxCR_EN) { }

    DMA1_Stream5->PAR  = (uint32_t)&USART2_DATA_REG;
    DMA1_Stream5->M0AR = (uint32_t)rx_dma_buf;
    DMA1_Stream5->NDTR = RX_DMA_BUF_LEN;

    rx_chunk_cb = cb;
    rx_last_pos = 0;

    DMA1_Stream5->CR = DMA_SxCR_CHSEL(4)     |
                        DMA_SxCR_DIR_P2M     |   // peripheral -> memory
                        DMA_SxCR_CIRC        |   // wraps automatically — never stalls
                        DMA_SxCR_MINC        |
                        DMA_SxCR_PSIZE_8B    |
                        DMA_SxCR_MSIZE_8B    |
                        DMA_SxCR_HTIE        |   // half-full: drain first half
                        DMA_SxCR_TCIE        |   // full: drain second half
                        DMA_SxCR_TEIE;

    USART2_CR3 |= USART_CR3_DMAR;
    DMA1_Stream5->CR |= DMA_SxCR_EN;
}

/*
 * Classic "double-buffer via single circular buffer" pattern: consumer only
 * ever reads the region the DMA engine has already vacated, computed from
 * NDTR (remaining count), never touching the region still being written.
 */
void DMA1_Stream5_IRQHandler(void)
{
    uint32_t status = DMA1->LISR;   // Stream5 flags live in LISR (bits [15:6])

    if (status & (1U << 11)) {                     // HTIF5 — half complete
        DMA1->LIFCR = (1U << 11);
        uint16_t half = RX_DMA_BUF_LEN / 2;
        if (rx_chunk_cb) rx_chunk_cb(&rx_dma_buf[rx_last_pos], half - rx_last_pos);
        rx_last_pos = half;
    }

    if (status & (1U << 5)) {                       // TCIF5 — full complete, wraps to 0
        DMA1->LIFCR = (1U << 5);
        if (rx_chunk_cb) rx_chunk_cb(&rx_dma_buf[rx_last_pos], RX_DMA_BUF_LEN - rx_last_pos);
        rx_last_pos = 0;
    }

    if (status & (1U << 3)) {                       // TEIF5
        DMA1->LIFCR = (1U << 3);
        DMA1_Stream5->CR &= ~DMA_SxCR_EN;
    }
}
```

## 5. Architectural Comparison — CPU Copy vs Interrupt Ring Buffer vs DMA

| Approach | Per-byte CPU cost | ISR frequency | Throughput ceiling | Best use case |
|---|---|---|---|---|
| Polling copy | Full busy-wait per byte | N/A (blocking) | Lowest — violates <100us latency budget | Bring-up/debug only |
| Interrupt SPSC ring buffer (prior work) | 1 ISR entry/exit per byte | N (one per byte) | Bounded by ISR overhead at high baud | Low-to-medium rate UART |
| **DMA memory-to-peripheral** | ~0 (one setup call) | 1 per buffer (or 2 in circular half/full mode) | Limited only by UART baud rate | Bulk transfers, streaming telemetry, log flushing |

DMA amortizes the fixed ISR entry/exit cost (register save/restore, typically tens of cycles on Cortex-M) across an entire buffer instead of every byte — directly extending the same "amortize the penalty" principle demonstrated in the multi-level cache AMAT analysis (4.658-cycle effective latency from a 200-cycle RAM penalty), just applied to interrupt overhead instead of memory hierarchy.

## 6. Discipline Notes (consistent with prior driver series)

- **Alignment**: `PSIZE`/`MSIZE` set to 8-bit here since UART DR is byte-wide; buffer must not require stricter alignment than `MSIZE` — for word-transfers (e.g., SPI flash bulk DMA) buffers need 4-byte alignment or the AHB access faults.
- **Clock-before-config-before-enable**: identical three-tier discipline used in GPIO/UART/I2C/SPI drivers — `RCC` enable, then `CR`/`PAR`/`M0AR`/`NDTR` programming, `EN` bit set last.
- **W1C interrupt flags**: `LIFCR`/`HIFCR` follow the same write-1-to-clear semantics as `EXTI_PR`, preventing storm/re-entry bugs.
- **Non-coherent DMA caution**: on Cortex-M there's no data cache so this is safe as-is; porting this pattern to a Cortex-A/TrustZone target requires the `DC CVAC`/`DC IVAC` cache maintenance primitives from the earlier DMA+cache-coherency simulator before/after arming the DMA engine, since the DMA controller is a non-coherent bus master relative to the CPU cache.
- **ISR minimalism**: both TX and RX handlers only clear flags and either flip a flag or invoke a short callback — heavy processing (e.g., protocol parsing) is deferred to the main loop/task context, preserving the <100us interrupt latency budget.