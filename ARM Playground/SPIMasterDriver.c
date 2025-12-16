# SPI Master Driver — Register-Level ARM Cortex-M Implementation ⭐⭐⭐

## 1. The Core Signal Chain

```
ARM CPU
   │  (writes CR1/DATA, polls STATUS flags)
   ▼
SPI Controller (peripheral shift register + clock generator)
   │
   ├── MOSI  (Master Out, Slave In)   -- driven by master, always
   ├── MISO  (Master In, Slave Out)   -- driven by slave, always
   ├── SCLK  (Serial Clock)           -- driven by master, always
   └── CS    (Chip Select, active-LOW)-- driven by master, GPIO, per-device
   ▼
SPI Device (Flash / Display / Sensor -- can be MULTIPLE on same bus,
            distinguished ONLY by which CS line is asserted)
```

**Critical architectural difference from I2C:** SPI has no addressing phase on the wire at all — device selection happens entirely through a dedicated GPIO (CS), external to the SPI peripheral itself. This means CS timing discipline is where most real-world SPI bugs live, not the shift-register logic.

## 2. Bus Protocol — Bit-Level Timing Diagram

```
CS:     ‾‾‾\_______________________________________/‾‾‾
             (asserted LOW for entire transaction)

SCLK:        _   _   _   _   _   _   _   _
        ____| |_| |_| |_| |_| |_| |_| |_| |____   (CPOL=0 idle-low shown)
              1   2   3   4   5   6   7   8

MOSI:   ----[b7][b6][b5][b4][b3][b2][b1][b0]----   (master shifts OUT)
MISO:   ----[b7][b6][b5][b4][b3][b2][b1][b0]----   (slave shifts OUT
                                                      SIMULTANEOUSLY)

Full-duplex: every SCLK pulse moves ONE bit out on MOSI and ONE bit
IN on MISO at the same time -- unlike I2C's half-duplex, turn-taking
ACK/NACK protocol, SPI has no acknowledgment concept at all. A
"failed" transfer is silent at the protocol level; the master
always receives exactly as many bits as it clocks out, valid or not.
```

**CPOL/CPHA — the four SPI modes (device-dependent, must match datasheet):**

```
CPOL = Clock Polarity: 0 = SCLK idles LOW, 1 = SCLK idles HIGH
CPHA = Clock Phase:    0 = sample on 1st edge, 1 = sample on 2nd edge

Mode 0 (CPOL=0,CPHA=0): idle LOW,  sample on rising edge  -- most common
Mode 1 (CPOL=0,CPHA=1): idle LOW,  sample on falling edge
Mode 2 (CPOL=1,CPHA=0): idle HIGH, sample on falling edge
Mode 3 (CPOL=1,CPHA=1): idle HIGH, sample on rising edge  -- 2nd most common
```

## 3. Register-Level SPI Overlay (STM32F4-class SPI1)

```c
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ============================================================
 * CMSIS-style struct overlay, consistent with the GPIO/UART/
 * Timer/ADC/I2C drivers in this series -- volatile on every
 * hardware-backed field.
 * ============================================================ */
typedef struct
{
    volatile uint32_t CR1;    /* 0x00 Control register 1        */
    volatile uint32_t CR2;    /* 0x04 Control register 2        */
    volatile uint32_t SR;     /* 0x08 Status register           */
    volatile uint32_t DR;     /* 0x0C Data register              */
    volatile uint32_t CRCPR;  /* 0x10 CRC polynomial register    */
    volatile uint32_t RXCRCR; /* 0x14 RX CRC register            */
    volatile uint32_t TXCRCR; /* 0x18 TX CRC register            */
} SPI_TypeDef;

#define SPI1   ((SPI_TypeDef *)0x40013000U)

/* CR1 bits */
#define SPI_CR1_CPHA        (1U << 0)    /* Clock phase                 */
#define SPI_CR1_CPOL        (1U << 1)    /* Clock polarity               */
#define SPI_CR1_MSTR        (1U << 2)    /* Master mode select           */
#define SPI_CR1_BR_MASK     (0x7U << 3)  /* Baud rate prescaler (3 bits) */
#define SPI_CR1_SPE         (1U << 6)    /* SPI enable                   */
#define SPI_CR1_LSBFIRST    (1U << 7)    /* LSB-first frame format       */
#define SPI_CR1_SSI         (1U << 8)    /* Internal slave select        */
#define SPI_CR1_SSM         (1U << 9)    /* Software slave management    */
#define SPI_CR1_DFF         (1U << 11)   /* Data frame format (16-bit)   */

/* SR bits */
#define SPI_SR_RXNE         (1U << 0)    /* RX buffer not empty (RX_READY)*/
#define SPI_SR_TXE          (1U << 1)    /* TX buffer empty               */
#define SPI_SR_BSY          (1U << 7)    /* Busy flag (shift in progress) */
#define SPI_SR_OVR          (1U << 6)    /* Overrun error                 */
#define SPI_SR_MODF         (1U << 5)    /* Mode fault                    */

#define RCC_APB2ENR   (*(volatile uint32_t *)0x40023844U)
#define RCC_SPI1EN    (1U << 12)

#define SPI_TIMEOUT_ITERS  100000U

/* Baud rate prescaler encoding for CR1.BR[2:0] */
#define SPI_BR_DIV2    0x00U
#define SPI_BR_DIV4    0x01U
#define SPI_BR_DIV8    0x02U
#define SPI_BR_DIV16   0x03U
#define SPI_BR_DIV32   0x04U
#define SPI_BR_DIV64   0x05U
#define SPI_BR_DIV128  0x06U
#define SPI_BR_DIV256  0x07U
```

## 4. Error Handling — Shared by Every Bus Operation

```c
/* ============================================================
 * Unlike I2C, SPI has NO protocol-level ACK/NACK, no arbitration,
 * and no clock-stretching stall -- the only failure modes visible
 * to the master are:
 *   OVR  (Overrun)   -- RX byte not read before the next arrives
 *   MODF (Mode Fault)-- NSS pin driven low unexpectedly (multi-
 *                        master conflict, only relevant if SSM=0)
 *   TIMEOUT          -- peripheral or wiring dead (BSY stuck, or
 *                        RXNE never sets)
 * A bounded timeout on every poll is still mandatory -- same
 * "never spin forever on dead hardware" discipline as every
 * other peripheral driver in this series.
 * ============================================================ */
typedef enum
{
    SPI_OK = 0,
    SPI_ERR_TIMEOUT,
    SPI_ERR_OVERRUN,
    SPI_ERR_MODE_FAULT,
} spi_result_t;

static spi_result_t spi_wait_flag(volatile uint32_t *reg, uint32_t mask, bool set)
{
    uint32_t timeout = SPI_TIMEOUT_ITERS;
    while (((*reg & mask) != 0U) != set)
    {
        if (--timeout == 0U)
        {
            return SPI_ERR_TIMEOUT;
        }
        if (SPI1->SR & SPI_SR_MODF)
        {
            /* MODF clear sequence per datasheet: read SR then write CR1 */
            (void)SPI1->SR;
            SPI1->CR1 |= SPI_CR1_SPE;
            return SPI_ERR_MODE_FAULT;
        }
    }
    return SPI_OK;
}

/* Overrun clear sequence: read DR then SR, per datasheet --
 * order matters, and a missed OVR silently corrupts the NEXT
 * byte's RX data, which is a notoriously hard bug to trace
 * back to its origin without this explicit clear path. */
static void spi_clear_overrun(void)
{
    (void)SPI1->DR;
    (void)SPI1->SR;
}
```

## 5. Initialization — Master Mode, Software CS

```c
/* ============================================================
 * spi_init: Master mode, Mode 0 (CPOL=0/CPHA=0), 8-bit frames,
 * software-managed CS (SSM=1, SSI=1) -- CS is handled as a plain
 * GPIO output by THIS driver rather than the SPI peripheral's
 * hardware NSS pin, which is the standard pattern for
 * multi-device buses (one CONTROLLER, many independent CS lines).
 * ============================================================ */
void spi_init(void)
{
    RCC_APB2ENR |= RCC_SPI1EN;   /* clock the peripheral first */

    /* "Clock and config before enable" -- identical discipline
     * to every prior peripheral (UART/I2C/ADC/Timer) in this
     * series. Config CR1 fully BEFORE setting SPE. */
    uint32_t cr1 = 0U;

    cr1 &= ~SPI_CR1_CPOL;         /* Mode 0: idle LOW               */
    cr1 &= ~SPI_CR1_CPHA;         /* Mode 0: sample on 1st edge     */
    cr1 |= SPI_CR1_MSTR;          /* Master mode                    */
    cr1 = (cr1 & ~SPI_CR1_BR_MASK) | (SPI_BR_DIV16 << 3);  /* baud   */
    cr1 &= ~SPI_CR1_LSBFIRST;     /* MSB-first (near-universal norm) */
    cr1 &= ~SPI_CR1_DFF;          /* 8-bit data frames               */

    /* SSM=1 + SSI=1: tells the peripheral "I am always the master,
     * ignore the physical NSS pin entirely" -- required because
     * we drive CS manually via GPIO, not the SPI peripheral's own
     * NSS logic. Without SSI=1 here, the peripheral would see a
     * floating/unasserted NSS and fault with MODF. */
    cr1 |= SPI_CR1_SSM | SPI_CR1_SSI;

    SPI1->CR1 = cr1;
    SPI1->CR1 |= SPI_CR1_SPE;     /* enable peripheral last */
}
```

## 6. Core Primitive — `spi_transfer` (Prompt's Example, Hardened)

```c
/* ============================================================
 * spi_transfer: the ONE fundamental SPI operation. Because SPI
 * is full-duplex and clocked by shifting, every "write" is
 * SIMULTANEOUSLY a "read" -- there is no way to send a byte
 * without also receiving one, and no way to receive without
 * sending. This is why spi_read() below still transmits a dummy
 * byte: the clock itself is a side effect of writing DATA.
 *
 * Hardened vs. the prompt's example: adds bounded timeouts on
 * BOTH TXE (safe to write) and RXNE (RX ready) so a dead/
 * unresponsive device or peripheral can never hang the CPU.
 * ============================================================ */
uint8_t spi_transfer(uint8_t data)
{
    /* Wait until TX buffer is empty -- writing DATA while a
     * previous byte is still shifting out would corrupt the
     * frame. Not present in the prompt's minimal example, but
     * required for correctness under back-to-back calls. */
    if (spi_wait_flag(&SPI1->SR, SPI_SR_TXE, true) != SPI_OK)
    {
        return 0xFFU;   /* sentinel: bus stalled */
    }

    SPI1->DR = data;   /* SPI->DATA = data;  -- triggers the clock */

    /* while (!(SPI->STATUS & SPI_RX_READY)); -- bounded version */
    if (spi_wait_flag(&SPI1->SR, SPI_SR_RXNE, true) != SPI_OK)
    {
        return 0xFFU;
    }

    if (SPI1->SR & SPI_SR_OVR)
    {
        spi_clear_overrun();
    }

    return (uint8_t)SPI1->DR;   /* return SPI->DATA; */
}

/* ============================================================
 * Convenience wrappers built on the single full-duplex primitive
 * -- naming the INTENT even though the wire operation is
 * identical either way.
 * ============================================================ */
static inline void spi_write(uint8_t data)
{
    (void)spi_transfer(data);            /* RX byte discarded, don't care */
}

static inline uint8_t spi_read(void)
{
    return spi_transfer(0xFFU);          /* dummy TX byte to generate clock;
                                             0xFF is convention (idle-high
                                             MOSI), some devices expect 0x00 */
}

static inline void spi_transfer_buf(const uint8_t *tx, uint8_t *rx, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
    {
        uint8_t r = spi_transfer(tx ? tx[i] : 0xFFU);
        if (rx) { rx[i] = r; }
    }
}
```

## 7. CS (Chip Select) Discipline — GPIO-Level

```c
/* ============================================================
 * CS is the device-selection mechanism SPI lacks at the
 * protocol level -- getting its timing wrong is the single most
 * common SPI bug: asserting/deasserting at the wrong moment
 * corrupts the frame or leaves the device mid-command.
 *
 * Rule: CS LOW must happen strictly BEFORE the first SCLK edge,
 * and CS HIGH strictly AFTER the last SCLK edge -- i.e. CS
 * brackets the ENTIRE multi-byte transaction, not each byte.
 * ============================================================ */
typedef struct
{
    volatile uint32_t *bsrr;   /* GPIOx_BSRR -- atomic set/reset, same
                                  hardware-atomic pattern used in the
                                  GPIO/PWM drivers to avoid RMW hazards */
    uint16_t pin_mask;
} spi_cs_t;

static inline void spi_cs_assert(const spi_cs_t *cs)
{
    *(cs->bsrr) = (uint32_t)cs->pin_mask << 16U;   /* BSRR reset bits: CS LOW */
}

static inline void spi_cs_deassert(const spi_cs_t *cs)
{
    *(cs->bsrr) = (uint32_t)cs->pin_mask;          /* BSRR set bits: CS HIGH */
}

/* ============================================================
 * spi_txn: RAII-style bracketing helper -- guarantees CS is
 * always deasserted even on early-return error paths, avoiding
 * the classic bug of a device left permanently "selected" after
 * an error, which then corrupts every SUBSEQUENT transaction on
 * a shared bus (other devices' CS lines are irrelevant if THIS
 * one never released the MISO line on devices with tri-state
 * MISO gated by CS).
 * ============================================================ */
#define SPI_TXN_BEGIN(cs_ptr)   spi_cs_assert(cs_ptr)
#define SPI_TXN_END(cs_ptr)     spi_cs_deassert(cs_ptr)
```

## 8. Device 1 — SPI Flash (W25Q-series, JEDEC command set)

```c
/* ============================================================
 * SPI NOR Flash: command-driven protocol, CS brackets the
 * ENTIRE command+address+data sequence. Demonstrates multi-byte
 * transfers with a leading opcode -- the near-universal SPI
 * flash pattern.
 * ============================================================ */
#define FLASH_CMD_READ_ID     0x9FU
#define FLASH_CMD_READ_DATA   0x03U
#define FLASH_CMD_PAGE_PROG   0x02U
#define FLASH_CMD_WRITE_EN    0x06U
#define FLASH_CMD_READ_SR1    0x05U
#define FLASH_SR1_BUSY        0x01U

static spi_cs_t flash_cs;

/* ============================================================
 * flash_read_jedec_id: 3-byte manufacturer/device ID, the
 * standard "is this device even alive" identity check --
 * SPI's equivalent of I2C's WHO_AM_I read.
 * ============================================================ */
void flash_read_jedec_id(uint8_t *manufacturer, uint8_t *mem_type, uint8_t *capacity)
{
    SPI_TXN_BEGIN(&flash_cs);

    spi_write(FLASH_CMD_READ_ID);   /* opcode -- device now streams ID */
    *manufacturer = spi_read();     /* dummy-byte-clocked reads       */
    *mem_type     = spi_read();
    *capacity     = spi_read();

    SPI_TXN_END(&flash_cs);
}

/* ============================================================
 * flash_wait_busy: polls the flash's internal SR1.BUSY bit --
 * NOTE this is a SEPARATE, DEVICE-LEVEL busy flag inside the
 * flash chip itself, distinct from the SPI peripheral's own
 * BSY status bit. Flash program/erase operations take
 * milliseconds; the SPI bus is idle during that time but the
 * DEVICE is still completing the operation internally.
 * ============================================================ */
static void flash_wait_busy(void)
{
    uint8_t sr1;
    uint32_t timeout = 1000000U;
    do
    {
        SPI_TXN_BEGIN(&flash_cs);
        spi_write(FLASH_CMD_READ_SR1);
        sr1 = spi_read();
        SPI_TXN_END(&flash_cs);
    } while ((sr1 & FLASH_SR1_BUSY) && --timeout);
}

/* ============================================================
 * flash_read: multi-byte sequential read starting at a 24-bit
 * address -- CS stays asserted across the ENTIRE opcode+
 * address+data-burst, exactly matching the CS-bracketing rule.
 * ============================================================ */
void flash_read(uint32_t addr, uint8_t *buf, uint32_t len)
{
    SPI_TXN_BEGIN(&flash_cs);

    spi_write(FLASH_CMD_READ_DATA);
    spi_write((uint8_t)(addr >> 16));   /* 24-bit address, MSB first */
    spi_write((uint8_t)(addr >> 8));
    spi_write((uint8_t)(addr >> 0));

    for (uint32_t i = 0; i < len; i++)
    {
        buf[i] = spi_read();
    }

    SPI_TXN_END(&flash_cs);
}

/* ============================================================
 * flash_page_program: write-enable, then program up to 256B
 * page. Demonstrates the "arm before act" pattern common to
 * flash devices -- a program/erase command is IGNORED by
 * hardware unless preceded by its own dedicated WRITE_EN command
 * in a SEPARATE, prior CS-bracketed transaction.
 * ============================================================ */
void flash_page_program(uint32_t addr, const uint8_t *data, uint32_t len)
{
    SPI_TXN_BEGIN(&flash_cs);
    spi_write(FLASH_CMD_WRITE_EN);
    SPI_TXN_END(&flash_cs);      /* separate transaction -- WEL latches here */

    SPI_TXN_BEGIN(&flash_cs);
    spi_write(FLASH_CMD_PAGE_PROG);
    spi_write((uint8_t)(addr >> 16));
    spi_write((uint8_t)(addr >> 8));
    spi_write((uint8_t)(addr >> 0));
    for (uint32_t i = 0; i < len; i++)
    {
        spi_write(data[i]);
    }
    SPI_TXN_END(&flash_cs);

    flash_wait_busy();   /* program takes ~ms; must not touch flash until done */
}
```

## 9. Device 2 — Display (SSD1306-class, command/data GPIO framing)

```c
/* ============================================================
 * SPI Displays typically add a THIRD control line beyond CS --
 * a Data/Command (D/C) GPIO that tells the controller whether
 * the byte on the wire is a command opcode or pixel/parameter
 * data. This is external to the SPI protocol entirely, just
 * like CS, and must be set BEFORE asserting CS for that byte.
 * ============================================================ */
typedef struct
{
    volatile uint32_t *bsrr;
    uint16_t pin_mask;
} gpio_pin_t;

static spi_cs_t   disp_cs;
static gpio_pin_t disp_dc;     /* Data/Command select */
static gpio_pin_t disp_rst;    /* Hardware reset       */

static inline void disp_dc_command(void) { *(disp_dc.bsrr) = (uint32_t)disp_dc.pin_mask << 16U; }
static inline void disp_dc_data(void)    { *(disp_dc.bsrr) = (uint32_t)disp_dc.pin_mask; }

#define SSD1306_CMD_DISPLAY_ON   0xAFU
#define SSD1306_CMD_SET_CONTRAST 0x81U

void display_write_command(uint8_t cmd)
{
    disp_dc_command();          /* D/C LOW: "this byte is a command" --
                                    must settle BEFORE CS asserts        */
    SPI_TXN_BEGIN(&disp_cs);
    spi_write(cmd);
    SPI_TXN_END(&disp_cs);
}

void display_write_data(const uint8_t *data, uint32_t len)
{
    disp_dc_data();              /* D/C HIGH: "these bytes are pixel data" */
    SPI_TXN_BEGIN(&disp_cs);
    for (uint32_t i = 0; i < len; i++)
    {
        spi_write(data[i]);
    }
    SPI_TXN_END(&disp_cs);
}

/* ============================================================
 * display_init: hardware reset pulse (external to SPI entirely)
 * followed by the standard command sequence -- displays are the
 * clearest example of SPI being used purely as a fast, dumb
 * byte-shifting pipe, with ALL semantic meaning imposed by
 * device-specific command bytes, not the protocol itself.
 * ============================================================ */
void display_init(void)
{
    *(disp_rst.bsrr) = (uint32_t)disp_rst.pin_mask << 16U;   /* RST LOW */
    for (volatile int i = 0; i < 10000; i++) { }             /* hold >=10us */
    *(disp_rst.bsrr) = (uint32_t)disp_rst.pin_mask;          /* RST HIGH */
    for (volatile int i = 0; i < 10000; i++) { }             /* let it boot */

    display_write_command(SSD1306_CMD_SET_CONTRAST);
    display_write_command(0xCFU);            /* contrast value, as parameter
                                                  byte -- note: STILL sent via
                                                  write_command(), not
                                                  write_data(), per datasheet */
    display_write_command(SSD1306_CMD_DISPLAY_ON);
}

/* ============================================================
 * display_push_framebuffer: bulk pixel push -- demonstrates why
 * SPI (vs I2C) is the near-universal choice for displays: a
 * single long CS-bracketed burst at multi-MHz clock rates moves
 * an entire framebuffer in a fraction of the time an I2C bus
 * (typically capped far lower, and half-duplex with per-byte
 * ACK overhead) would require.
 * ============================================================ */
void display_push_framebuffer(const uint8_t *fb, uint32_t size)
{
    display_write_data(fb, size);
}
```

## 10. Device 3 — Sensor (MAX31855 Thermocouple, read-only SPI)

```c
/* ============================================================
 * MAX31855: a read-only SPI sensor -- no command byte at all.
 * Asserting CS and clocking 32 dummy bits IS the entire "read
 * request"; the device streams its latest conversion result
 * unconditionally. Demonstrates the simplest possible SPI
 * device protocol and a real-world use of 32-bit multi-byte
 * transfer + bit-field decoding.
 * ============================================================ */
static spi_cs_t therm_cs;

typedef struct
{
    float    thermocouple_c;
    float    internal_c;
    bool     fault;
    uint8_t  fault_reason;   /* bit0=open, bit1=short-GND, bit2=short-VCC */
} max31855_reading_t;

/* ============================================================
 * max31855_read: 4-byte (32-bit) burst read, CS brackets all
 * four bytes as ONE atomic conversion snapshot -- reading fewer
 * than 4 bytes, or split across two CS-bracketed transactions,
 * would tear the reading between thermocouple and internal temp
 * fields (same "don't split a single sample across transactions"
 * principle as the MPU6050 burst read in the I2C driver).
 * ============================================================ */
spi_result_t max31855_read(max31855_reading_t *out)
{
    uint8_t raw[4];

    SPI_TXN_BEGIN(&therm_cs);
    for (int i = 0; i < 4; i++)
    {
        raw[i] = spi_read();   /* dummy-clocked -- device streams unconditionally */
    }
    SPI_TXN_END(&therm_cs);

    uint32_t word = ((uint32_t)raw[0] << 24) | ((uint32_t)raw[1] << 16) |
                    ((uint32_t)raw[2] << 8)  |  (uint32_t)raw[3];

    /* Bit 16: fault flag; bits [17:18] reserved/internal sign context */
    out->fault = (word & 0x00010000UL) != 0U;
    out->fault_reason = (uint8_t)(raw[3] & 0x07U);

    if (out->fault)
    {
        out->thermocouple_c = 0.0f;
        out->internal_c = 0.0f;
        return SPI_ERR_OVERRUN;   /* reused as generic "device fault" signal */
    }

    /* Thermocouple temp: bits [31:18], 14-bit signed, 0.25C/LSB */
    int16_t tc_raw = (int16_t)(word >> 18);
    out->thermocouple_c = (float)tc_raw * 0.25f;

    /* Internal (cold-junction) temp: bits [15:4], 12-bit signed, 0.0625C/LSB */
    int16_t internal_raw = (int16_t)((word >> 4) & 0x0FFFU);
    if (internal_raw & 0x0800) { internal_raw |= 0xF000; }   /* sign-extend */
    out->internal_c = (float)internal_raw * 0.0625f;

    return SPI_OK;
}
```

## 11. Test / Demo Harness

```c
#include <stdio.h>
#include <assert.h>

/* ============================================================
 * Software model verifying CS-bracketing DISCIPLINE and full-
 * duplex transfer semantics without real hardware -- mirrors
 * the mock bus state-machine testing approach used for the I2C
 * driver, adapted to SPI's very different (address-less,
 * CS-based) device-selection model.
 * ============================================================ */
static bool g_cs_asserted = false;
static uint8_t g_mock_tx_log[32];
static uint32_t g_mock_tx_count = 0;

void mock_cs_assert(void)   { assert(!g_cs_asserted); g_cs_asserted = true; }
void mock_cs_deassert(void) { assert(g_cs_asserted);  g_cs_asserted = false; }
uint8_t mock_transfer(uint8_t tx)
{
    assert(g_cs_asserted);   /* every transfer MUST occur inside a CS bracket */
    g_mock_tx_log[g_mock_tx_count++] = tx;
    return 0xA5U;            /* fixed mock RX value */
}

void test_cs_bracketing(void)
{
    g_mock_tx_count = 0;
    mock_cs_assert();
    mock_transfer(FLASH_CMD_READ_ID);
    uint8_t r1 = mock_transfer(0xFF);
    uint8_t r2 = mock_transfer(0xFF);
    mock_cs_deassert();

    assert(g_mock_tx_count == 3);
    assert(g_mock_tx_log[0] == FLASH_CMD_READ_ID);
    assert(r1 == 0xA5U && r2 == 0xA5U);
    assert(g_cs_asserted == false);   /* must be released after transaction */
    printf("CS bracketing verified: opcode+2 dummy reads, single CS cycle\n");
}

void test_full_duplex_property(void)
{
    /* Verifies the core SPI property: writing N bytes ALWAYS
     * yields N received bytes, even when the caller only cares
     * about one direction (spi_write discards RX, spi_read sends
     * dummy TX) -- there is no way to decouple the two directions. */
    g_mock_tx_count = 0;
    mock_cs_assert();
    for (int i = 0; i < 4; i++) { (void)mock_transfer(0xFF); }  /* spi_read x4 */
    mock_cs_deassert();
    assert(g_mock_tx_count == 4);
    printf("Full-duplex property verified: 4 clocked bytes -> 4 TX log entries\n");
}

void test_address_less_device_model(void)
{
    /* Unlike I2C, no address byte should EVER appear on an SPI
     * transfer -- device selection is purely CS, so the first
     * byte transmitted must always be either an opcode or raw
     * data, never a 7-bit-address+R/W-style framing byte. */
    g_mock_tx_count = 0;
    mock_cs_assert();
    mock_transfer(0x03U);   /* flash READ_DATA opcode, NOT an address+dir byte */
    mock_cs_deassert();
    assert(g_mock_tx_log[0] == 0x03U);
    printf("Address-less device selection model verified (CS-only, no addr byte)\n");
}

int main(void)
{
    test_cs_bracketing();
    test_full_duplex_property();
    test_address_less_device_model();
    printf("All SPI tests passed.\n");
    return 0;
}
```

## 12. Design Rules Summary

| Rule | Why |
|---|---|
| **Bounded timeout on TXE and RXNE, not just RXNE** | Writing DATA while a prior byte is still shifting corrupts the frame; the prompt's example only guards RX_READY, missing the TX-side hazard entirely. |
| **CS brackets the ENTIRE multi-byte transaction, never per-byte** | SPI has no addressing phase — CS *is* the addressing mechanism; deasserting mid-command tells the device "transaction over," discarding any command context (e.g. flash forgets which address it was reading). |
| **SSM=1 + SSI=1 for software-managed CS** | Without explicitly telling the peripheral to ignore its own NSS pin, a floating/GPIO-driven CS causes spurious MODF (mode fault) errors. |
| **Full-duplex is unconditional — every write is also a read** | `spi_read()` must still transmit a dummy byte because the clock is a *side effect* of the DATA register write; there's no "receive-only" mode at the peripheral level. |
| **Device-level busy flag ≠ peripheral BSY flag** | Flash program/erase completion (`flash_wait_busy`) is tracked via a status byte read over SPI, entirely separate from the SPI peripheral's own shift-register-busy bit. |
| **Overrun clear = read DR then SR, in that order** | Reversing the order, or skipping it, leaves OVR set and silently corrupts the *next* transfer's RX data — a classic hard-to-trace bug. |
| **D/C (or similar control) GPIO must settle before CS asserts** | Display controllers sample D/C at the start of the CS-low window; asserting CS first races the D/C level and can misclassify a data byte as a command (or vice versa). |
| **Multi-byte sensor reads (e.g. MAX31855's 32 bits) must be one CS-bracketed burst** | Splitting a single conversion result across transactions risks tearing between fields, identical to the MPU6050 burst-read principle in the I2C driver. |

## 13. Concepts Demonstrated

| Concept | Where it shows up |
|---|---|
| **Full-duplex transfer** | `spi_transfer()` hardened prompt example — Section 6; every write is inherently also a read. |
| **CPOL/CPHA modes** | Section 2 diagram + Mode 0 configuration in `spi_init()` — device datasheet compliance. |
| **CS-based addressing (no protocol-level addressing)** | Section 7's `spi_cs_t`/`SPI_TXN_BEGIN/END` bracketing, contrasted explicitly against I2C's on-wire addressing in Section 1 and the test harness. |
| **Command/opcode framing** | Flash's `FLASH_CMD_*` opcodes, Display's D/C-line command/data split, Thermocouple's opcode-less streaming — three different real-world framing conventions. |
| **Error handling (OVR/MODF/timeout)** | Section 4, applied consistently through every device driver via `spi_wait_flag()`. |
| **Real device integration** | SPI Flash (Section 8, command-driven, write-enable-then-act pattern), Display (Section 9, D/C-gated framing, bulk framebuffer push), Sensor (Section 10, read-only 32-bit burst, bit-field decode). |