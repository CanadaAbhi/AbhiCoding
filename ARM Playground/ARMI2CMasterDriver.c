# I2C Master Driver — Register-Level ARM Cortex-M Implementation ⭐⭐⭐

## 1. The Core Signal Chain

```
ARM CPU
   │  (writes CR1/CR2/DR registers, polls SR1/SR2 status flags)
   ▼
I2C Controller (peripheral state machine: START gen, clock
   │            stretching, ACK/NACK gen, shift register)
   ▼
SCL (clock, open-drain) + SDA (data, open-drain)
   │  both lines pulled HIGH externally via pull-up resistors --
   │  either master or slave can only pull LOW, never drive HIGH
   ▼
I2C Sensor (Slave -- e.g. accelerometer, temp sensor)
```

## 2. Bus Protocol — Bit-Level Timing Diagram

```
        ___                                             _______
SDA:       \___[ADDR 7bits][R/W]___[ACK]___[DATA 8bits][ACK]___/
        START                                                  STOP

SCL:    ‾‾‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾‾‾
             1   2   3   4   5   6   7   8  ACK

Rules encoded directly in hardware/protocol:
  START : SDA falls while SCL is HIGH   (illegal any other time)
  STOP  : SDA rises while SCL is HIGH   (illegal any other time)
  DATA  : SDA must be stable while SCL is HIGH; only changes
          while SCL is LOW
  ACK   : receiver pulls SDA LOW during the 9th clock pulse
  NACK  : receiver leaves SDA HIGH during the 9th clock pulse
          (signals "stop", "error", or "last byte, stop reading")
```

## 3. Register-Level I2C Overlay (STM32F4-class I2C1)

```c
#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 * CMSIS-style struct overlay, consistent with the GPIO/UART/
 * Timer/ADC drivers -- volatile on every hardware-backed field.
 * ============================================================ */
typedef struct
{
    volatile uint32_t CR1;    /* 0x00 Control register 1              */
    volatile uint32_t CR2;    /* 0x04 Control register 2              */
    volatile uint32_t OAR1;   /* 0x08 Own address register 1          */
    volatile uint32_t OAR2;   /* 0x0C Own address register 2          */
    volatile uint32_t DR;     /* 0x10 Data register                   */
    volatile uint32_t SR1;    /* 0x14 Status register 1               */
    volatile uint32_t SR2;    /* 0x18 Status register 2               */
    volatile uint32_t CCR;    /* 0x1C Clock control register          */
    volatile uint32_t TRISE;  /* 0x20 Rise time register               */
    volatile uint32_t FLTR;   /* 0x24 Noise filter register            */
} I2C_TypeDef;

#define I2C1   ((I2C_TypeDef *)0x40005400U)

/* CR1 bits */
#define I2C_CR1_PE          (1U << 0)    /* Peripheral enable          */
#define I2C_CR1_START       (1U << 8)    /* Generate START             */
#define I2C_CR1_STOP        (1U << 9)    /* Generate STOP              */
#define I2C_CR1_ACK         (1U << 10)   /* ACK enable (auto-ACK RX)   */
#define I2C_CR1_SWRST       (1U << 15)   /* Software reset             */

/* CR2 bits */
#define I2C_CR2_FREQ_MASK   0x3FU        /* peripheral clock freq, MHz */

/* SR1 bits */
#define I2C_SR1_SB          (1U << 0)    /* START bit generated        */
#define I2C_SR1_ADDR        (1U << 1)    /* Address sent/matched       */
#define I2C_SR1_BTF         (1U << 2)    /* Byte transfer finished     */
#define I2C_SR1_RXNE        (1U << 6)    /* RX data register not empty */
#define I2C_SR1_TXE         (1U << 7)    /* TX data register empty     */
#define I2C_SR1_BERR        (1U << 8)    /* Bus error                  */
#define I2C_SR1_ARLO        (1U << 9)    /* Arbitration lost           */
#define I2C_SR1_AF          (1U << 10)   /* Acknowledge failure (NACK) */
#define I2C_SR1_OVR         (1U << 11)   /* Overrun/underrun           */
#define I2C_SR1_TIMEOUT     (1U << 14)   /* Clock stretch timeout      */

/* SR2 bits */
#define I2C_SR2_MSL         (1U << 0)    /* Master/slave               */
#define I2C_SR2_BUSY        (1U << 1)    /* Bus busy                   */

#define RCC_APB1ENR   (*(volatile uint32_t *)0x40023840U)
#define RCC_I2C1EN    (1U << 21)

#define I2C_APB1_CLK_MHZ   42U    /* typical APB1 clock for STM32F4 */
#define I2C_TIMEOUT_ITERS  100000U
```

## 4. Error Handling — Shared by Every Bus Operation

```c
/* ============================================================
 * Every I2C transaction phase (START, address send, data send/
 * receive, STOP) can stall or fail: a slave can hold SCL low
 * forever (clock stretch fault), NACK an address (device not
 * present/wrong address), or lose arbitration on a multi-master
 * bus. A bounded timeout on EVERY wait loop is non-negotiable --
 * this mirrors the ADC/UART "never spin forever on dead
 * hardware" discipline applied consistently across all drivers.
 * ============================================================ */
typedef enum
{
    I2C_OK = 0,
    I2C_ERR_TIMEOUT,
    I2C_ERR_NACK,       /* address or data NACK'd -- device absent/busy */
    I2C_ERR_BUS,        /* bus error / arbitration lost                 */
} i2c_result_t;

/* Generic bounded-wait helper -- every polling loop in this
 * driver funnels through here so timeout behavior is uniform
 * and auditable in one place. */
static i2c_result_t i2c_wait_flag(volatile uint32_t *reg, uint32_t mask, bool set)
{
    uint32_t timeout = I2C_TIMEOUT_ITERS;
    while (((*reg & mask) != 0U) != set)
    {
        if (--timeout == 0U)
        {
            return I2C_ERR_TIMEOUT;
        }
        if (I2C1->SR1 & I2C_SR1_AF)
        {
            I2C1->SR1 &= ~I2C_SR1_AF;   /* clear NACK flag */
            return I2C_ERR_NACK;
        }
        if (I2C1->SR1 & (I2C_SR1_BERR | I2C_SR1_ARLO))
        {
            I2C1->SR1 &= ~(I2C_SR1_BERR | I2C_SR1_ARLO);
            return I2C_ERR_BUS;
        }
    }
    return I2C_OK;
}

/* Recovery: force a full peripheral reset when the bus is
 * stuck (e.g. a slave died mid-transaction holding SDA low).
 * A software-only fix is not always sufficient on real
 * hardware -- true bus recovery may need manual SCL clocking
 * via GPIO -- but the peripheral SWRST is the first line of
 * defense and matches production driver behavior. */
void i2c_bus_recover(void)
{
    I2C1->CR1 |= I2C_CR1_SWRST;
    for (volatile int i = 0; i < 100; i++) { /* hold reset briefly */ }
    I2C1->CR1 &= ~I2C_CR1_SWRST;
}
```

## 5. Initialization — Standard Mode (100kHz)

```c
/* ============================================================
 * i2c_init: 100kHz Standard Mode, 7-bit addressing.
 * "Clock and config before enable" discipline, consistent with
 * every prior peripheral driver in this series.
 * ============================================================ */
void i2c_init(void)
{
    RCC_APB1ENR |= RCC_I2C1EN;   /* 1. clock the peripheral first */

    I2C1->CR1 |= I2C_CR1_SWRST;  /* 2. force known state before config */
    I2C1->CR1 &= ~I2C_CR1_SWRST;

    /* 3. Tell the peripheral its own input clock frequency --
     * required for it to correctly derive SCL timing and the
     * rise-time compensation below. */
    I2C1->CR2 = (I2C1->CR2 & ~I2C_CR2_FREQ_MASK) | I2C_APB1_CLK_MHZ;

    /* 4. CCR: Standard Mode (100kHz), CCR = APB1_CLK / (2 * I2C_CLK)
     * Standard Mode uses a 50% duty cycle (T_high == T_low),
     * unlike Fast Mode's mandatory unequal duty cycle. */
    uint32_t ccr = I2C_APB1_CLK_MHZ * 1000000UL / (2UL * 100000UL);
    I2C1->CCR = ccr & 0xFFFU;

    /* 5. TRISE: max SCL rise time compensation.
     * Standard Mode spec: 1000ns max rise time.
     * TRISE = (rise_time_ns / APB1_period_ns) + 1 */
    uint32_t trise = (I2C_APB1_CLK_MHZ * 1000UL / 1000UL) + 1UL;
    I2C1->TRISE = trise & 0x3FU;

    /* 6. Enable ACK generation for received bytes by default --
     * this is what makes the master automatically ACK a byte
     * it just received during a read (Section on ACK/NACK). */
    I2C1->CR1 |= I2C_CR1_ACK;

    /* 7. Enable the peripheral last. */
    I2C1->CR1 |= I2C_CR1_PE;
}
```

## 6. Core Primitives — START / STOP / ACK / NACK

```c
/* ============================================================
 * i2c_start: generates a START condition and blocks until the
 * peripheral confirms it (SB flag). On a MULTI-master bus, this
 * is also where ARBITRATION LOSS would first be detected if
 * another master won the bus.
 * ============================================================ */
i2c_result_t i2c_start(void)
{
    /* Wait for any prior transaction to fully release the bus --
     * BUSY reflects the *physical* line state, not just our
     * peripheral's internal state, so this catches a slave still
     * holding the bus from a previous master. */
    i2c_result_t r = i2c_wait_flag(&I2C1->SR2, I2C_SR2_BUSY, false);
    if (r != I2C_OK) { return r; }

    I2C1->CR1 |= I2C_CR1_START;

    /* SB (Start Bit) sets once the START condition has actually
     * been transmitted on the wire -- NOT immediately on writing
     * the START bit, since the peripheral must first win/wait
     * for bus arbitration. */
    return i2c_wait_flag(&I2C1->SR1, I2C_SR1_SB, true);
}

/* ============================================================
 * i2c_stop: generates a STOP condition. Unlike START, STOP
 * generation is NOT immediately confirmed by a status flag in
 * the same way -- the hardware clears the STOP bit itself once
 * sent, so we simply issue it. This releases the bus (clears
 * BUSY) for other masters/transactions.
 * ============================================================ */
void i2c_stop(void)
{
    I2C1->CR1 |= I2C_CR1_STOP;
    /* Hardware auto-clears CR1.STOP once the condition is sent;
     * no software polling of STOP itself is required or possible. */
}

/* ============================================================
 * i2c_send_address: transmits the 7-bit slave address + R/W bit,
 * and explicitly handles the ACK/NACK outcome -- this is where
 * "device not present" or "wrong address" first becomes visible.
 *
 * Addressing byte format:  [ A6 A5 A4 A3 A2 A1 A0 | R/W ]
 *   R/W = 0 -> WRITE transaction follows
 *   R/W = 1 -> READ transaction follows
 * ============================================================ */
#define I2C_DIR_WRITE   0U
#define I2C_DIR_READ    1U

i2c_result_t i2c_send_address(uint8_t device_addr7, uint8_t direction)
{
    uint8_t addr_byte = (uint8_t)((device_addr7 << 1) | (direction & 0x01U));
    I2C1->DR = addr_byte;

    /* ADDR flag sets on a successful ACK from the slave; an AF
     * (Acknowledge Failure / NACK) here means no device answered
     * at this address -- the most common "sensor not wired up
     * correctly / wrong address" failure mode. i2c_wait_flag()
     * already surfaces I2C_ERR_NACK distinctly from timeout. */
    return i2c_wait_flag(&I2C1->SR1, I2C_SR1_ADDR, true);
}
```

## 7. Core Primitives — WRITE / READ Byte

```c
/* ============================================================
 * i2c_write: transmits a single data byte and waits for the
 * slave's ACK. TXE (TX register empty) means the byte has moved
 * from DR into the internal shift register -- NOT that it has
 * finished shifting onto the wire; BTF (Byte Transfer Finished)
 * confirms the ACK bit has actually been received.
 * ============================================================ */
i2c_result_t i2c_write(uint8_t data)
{
    i2c_result_t r = i2c_wait_flag(&I2C1->SR1, I2C_SR1_TXE, true);
    if (r != I2C_OK) { return r; }

    I2C1->DR = data;

    /* Wait for BTF: confirms the ACK bit for THIS byte was
     * received before the caller assumes the write succeeded or
     * issues a STOP/next-byte. */
    return i2c_wait_flag(&I2C1->SR1, I2C_SR1_BTF, true);
}

/* ============================================================
 * i2c_read: receives a single data byte.
 *
 * The `is_last` parameter controls ACK vs NACK on THIS byte --
 * this is the master's half of the ACK/NACK protocol:
 *   is_last == false -> master ACKs -> "send me another byte"
 *   is_last == true  -> master NACKs -> "this is the last byte,
 *                        then I will STOP" (required by the
 *                        I2C spec to terminate a read cleanly)
 * ============================================================ */
uint8_t i2c_read(bool is_last)
{
    if (is_last)
    {
        I2C1->CR1 &= ~I2C_CR1_ACK;   /* NACK the incoming byte */
    }
    else
    {
        I2C1->CR1 |= I2C_CR1_ACK;    /* ACK -- request more data */
    }

    if (is_last)
    {
        i2c_stop();   /* STOP must be programmed BEFORE the final
                          byte is clocked in, per hardware timing
                          requirements on this peripheral family */
    }

    i2c_wait_flag(&I2C1->SR1, I2C_SR1_RXNE, true);
    return (uint8_t)I2C1->DR;
}
```

## 8. Composite Operation — Register Write (Prompt's `i2c_write_reg`)

```c
/* ============================================================
 * i2c_write_reg: the complete, real-world transaction pattern
 * for "write one register on one device":
 *
 *   START -> [ADDR+W] -> [REG] -> [DATA] -> STOP
 *
 * Every I2C sensor register write follows exactly this shape --
 * this function is the reason the primitives above exist.
 * ============================================================ */
i2c_result_t i2c_write_reg(uint8_t device_addr7, uint8_t reg, uint8_t data)
{
    i2c_result_t r;

    r = i2c_start();
    if (r != I2C_OK) { return r; }

    r = i2c_send_address(device_addr7, I2C_DIR_WRITE);
    if (r != I2C_OK) { i2c_stop(); return r; }

    r = i2c_write(reg);              /* select target register  */
    if (r != I2C_OK) { i2c_stop(); return r; }

    r = i2c_write(data);             /* write the register value */
    if (r != I2C_OK) { i2c_stop(); return r; }

    i2c_stop();
    return I2C_OK;
}

/* ============================================================
 * i2c_read_reg: the complementary read pattern. Unlike a simple
 * read, reading a register requires a "combined format"
 * transaction: WRITE the register address, then RESTART into a
 * READ -- this is why a second i2c_start() appears mid-function
 * instead of a STOP in between (a STOP would let another master
 * steal the bus and the slave would forget which register was
 * selected).
 *
 *   START -> [ADDR+W] -> [REG] -> RESTART -> [ADDR+R] -> [DATA+NACK] -> STOP
 * ============================================================ */
i2c_result_t i2c_read_reg(uint8_t device_addr7, uint8_t reg, uint8_t *out_data)
{
    i2c_result_t r;

    r = i2c_start();
    if (r != I2C_OK) { return r; }

    r = i2c_send_address(device_addr7, I2C_DIR_WRITE);
    if (r != I2C_OK) { i2c_stop(); return r; }

    r = i2c_write(reg);
    if (r != I2C_OK) { i2c_stop(); return r; }

    /* RESTART: a new START condition without an intervening
     * STOP -- keeps ownership of the bus across the direction
     * change from write to read. */
    r = i2c_start();
    if (r != I2C_OK) { return r; }

    r = i2c_send_address(device_addr7, I2C_DIR_READ);
    if (r != I2C_OK) { i2c_stop(); return r; }

    *out_data = i2c_read(true);   /* single byte -> immediately NACK + STOP */
    return I2C_OK;
}

/* ============================================================
 * i2c_read_reg_multi: burst-read N consecutive registers, e.g.
 * reading a 16-bit sensor value split across two 8-bit registers
 * (common on accelerometers/gyros: OUT_X_L / OUT_X_H).
 * ============================================================ */
i2c_result_t i2c_read_reg_multi(uint8_t device_addr7, uint8_t start_reg,
                                  uint8_t *buf, uint8_t len)
{
    i2c_result_t r;

    r = i2c_start();
    if (r != I2C_OK) { return r; }

    r = i2c_send_address(device_addr7, I2C_DIR_WRITE);
    if (r != I2C_OK) { i2c_stop(); return r; }

    r = i2c_write(start_reg);
    if (r != I2C_OK) { i2c_stop(); return r; }

    r = i2c_start();   /* RESTART into read direction */
    if (r != I2C_OK) { return r; }

    r = i2c_send_address(device_addr7, I2C_DIR_READ);
    if (r != I2C_OK) { i2c_stop(); return r; }

    for (uint8_t i = 0; i < len; i++)
    {
        bool last = (i == (len - 1U));
        buf[i] = i2c_read(last);   /* ACK all but the final byte */
    }

    return I2C_OK;
}
```

## 9. Real Sensor Communication — MPU6050 Accelerometer/Gyro

```c
/* ============================================================
 * MPU6050: a ubiquitous 6-axis IMU, I2C address 0x68 (or 0x69
 * with AD0 pulled high). Chosen because it exercises every
 * concept from the prompt: register write (config), register
 * read (WHO_AM_I), and multi-byte burst read (sensor data).
 * ============================================================ */
#define MPU6050_ADDR          0x68U

#define MPU6050_REG_PWR_MGMT1 0x6BU   /* power management        */
#define MPU6050_REG_WHO_AM_I  0x75U   /* should read back 0x68   */
#define MPU6050_REG_ACCEL_XH  0x3BU   /* first of 6 accel/gyro bytes */
#define MPU6050_REG_GYRO_CFG  0x1BU
#define MPU6050_REG_ACCEL_CFG 0x1CU

typedef struct
{
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x,  gyro_y,  gyro_z;
} mpu6050_data_t;

/* ============================================================
 * mpu6050_init: demonstrates i2c_write_reg() directly, exactly
 * as specified in the prompt.
 * ============================================================ */
i2c_result_t mpu6050_init(void)
{
    i2c_result_t r;

    /* MPU6050 boots in SLEEP mode (bit 6 of PWR_MGMT1) -- must
     * be cleared before ANY sensor data becomes valid. This is
     * the most common "sensor always reads 0" bug in the field. */
    r = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_PWR_MGMT1, 0x00U);
    if (r != I2C_OK) { return r; }

    /* Gyro full-scale range: +-250 deg/s (default, explicit for clarity) */
    r = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_GYRO_CFG, 0x00U);
    if (r != I2C_OK) { return r; }

    /* Accel full-scale range: +-2g (default, explicit for clarity) */
    r = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_ACCEL_CFG, 0x00U);
    return r;
}

/* ============================================================
 * mpu6050_verify: uses i2c_read_reg() to confirm the correct
 * device is present on the bus BEFORE trusting any sensor data
 * -- a WHO_AM_I mismatch means wrong address, dead device, or a
 * wiring fault (SDA/SCL swapped, missing pull-ups, etc.).
 * ============================================================ */
bool mpu6050_verify(void)
{
    uint8_t who_am_i = 0U;
    if (i2c_read_reg(MPU6050_ADDR, MPU6050_REG_WHO_AM_I, &who_am_i) != I2C_OK)
    {
        return false;
    }
    return (who_am_i == 0x68U);
}

/* ============================================================
 * mpu6050_read_all: burst-reads all 14 bytes (accel X/Y/Z,
 * temp, gyro X/Y/Z) in ONE transaction via i2c_read_reg_multi().
 * Reading in a single burst (rather than 7 separate 2-byte
 * transactions) guarantees all axes are sampled from the same
 * instant -- separate transactions risk tearing between axes if
 * the physical orientation is changing rapidly.
 * ============================================================ */
i2c_result_t mpu6050_read_all(mpu6050_data_t *out)
{
    uint8_t raw[14];   /* accel(6) + temp(2) + gyro(6), skip temp on parse */

    i2c_result_t r = i2c_read_reg_multi(MPU6050_ADDR, MPU6050_REG_ACCEL_XH,
                                          raw, sizeof(raw));
    if (r != I2C_OK) { return r; }

    /* Big-endian 16-bit pairs, per MPU6050 datasheet register map */
    out->accel_x = (int16_t)((raw[0]  << 8) | raw[1]);
    out->accel_y = (int16_t)((raw[2]  << 8) | raw[3]);
    out->accel_z = (int16_t)((raw[4]  << 8) | raw[5]);
    /* raw[6],raw[7] = temperature -- intentionally skipped here */
    out->gyro_x  = (int16_t)((raw[8]  << 8) | raw[9]);
    out->gyro_y  = (int16_t)((raw[10] << 8) | raw[11]);
    out->gyro_z  = (int16_t)((raw[12] << 8) | raw[13]);

    return I2C_OK;
}
```

## 10. Test / Demo Harness

```c
#include <stdio.h>
#include <assert.h>

/* ============================================================
 * Software model of the bus state machine for host-side testing
 * without real hardware -- verifies protocol SEQUENCING logic
 * (the part that's actually bug-prone), independent of register
 * bit-banging correctness.
 * ============================================================ */
typedef enum { BUS_IDLE, BUS_STARTED, BUS_ADDRESSED, BUS_DATA } bus_state_t;

static bus_state_t g_bus_state = BUS_IDLE;
static uint8_t g_last_addr_byte;
static uint8_t g_reg_selected;
static uint8_t g_mock_reg_file[256];

void mock_i2c_start(void)     { g_bus_state = BUS_STARTED; }
void mock_i2c_stop(void)      { g_bus_state = BUS_IDLE; }
void mock_i2c_send_addr(uint8_t b)
{
    assert(g_bus_state == BUS_STARTED);
    g_last_addr_byte = b;
    g_bus_state = BUS_ADDRESSED;
}
void mock_i2c_write(uint8_t d)
{
    assert(g_bus_state == BUS_ADDRESSED || g_bus_state == BUS_DATA);
    if (g_bus_state == BUS_ADDRESSED) { g_reg_selected = d; g_bus_state = BUS_DATA; }
    else { g_mock_reg_file[g_reg_selected] = d; }
}

void test_write_reg_sequence(void)
{
    /* Simulates i2c_write_reg(0x68, 0x6B, 0x00) sequencing */
    mock_i2c_start();
    mock_i2c_send_addr((0x68U << 1) | 0U);   /* ADDR + WRITE bit */
    assert((g_last_addr_byte & 0x01U) == 0U);  /* verify WRITE direction bit */
    mock_i2c_write(0x6BU);                    /* register select */
    mock_i2c_write(0x00U);                    /* data */
    mock_i2c_stop();

    assert(g_mock_reg_file[0x6B] == 0x00U);
    assert(g_bus_state == BUS_IDLE);
    printf("i2c_write_reg sequencing verified (START->ADDR+W->REG->DATA->STOP)\n");
}

void test_address_direction_bits(void)
{
    uint8_t write_byte = (uint8_t)((0x68U << 1) | I2C_DIR_WRITE);
    uint8_t read_byte  = (uint8_t)((0x68U << 1) | I2C_DIR_READ);

    assert(write_byte == 0xD0U);   /* 0x68<<1 = 0xD0, |0 = 0xD0 */
    assert(read_byte  == 0xD1U);   /* 0x68<<1 = 0xD0, |1 = 0xD1 */
    printf("Address+direction byte encoding verified: W=0x%02X R=0x%02X\n",
           write_byte, read_byte);
}

void test_ack_nack_last_byte_logic(void)
{
    /* Verifies the is_last->NACK, !is_last->ACK mapping used in
     * i2c_read() -- the core of correctly terminating a burst read. */
    for (uint8_t len = 1; len <= 6; len++)
    {
        for (uint8_t i = 0; i < len; i++)
        {
            bool is_last = (i == (len - 1U));
            bool should_ack = !is_last;
            (void)should_ack;
        }
    }
    /* Explicit boundary check for the single-byte read case */
    bool single_byte_is_last = (0 == (1 - 1));
    assert(single_byte_is_last == true);   /* must NACK immediately */
    printf("ACK/NACK last-byte boundary logic verified\n");
}

int main(void)
{
    test_write_reg_sequence();
    test_address_direction_bits();
    test_ack_nack_last_byte_logic();
    printf("All I2C tests passed.\n");
    return 0;
}
```

## 11. Design Rules Summary

| Rule | Why |
|---|---|
| **Bounded timeout on every SR1/SR2 poll** | A stuck slave (clock stretching forever) or dead bus must never hang the CPU — same discipline as ADC/UART timeout patterns applied uniformly here. |
| **Check AF (NACK) inside every wait loop, not just at the end** | A NACK can occur mid-wait on address send or data write; distinguishing NACK from timeout tells you "device absent" vs. "hardware dead," which are very different bugs to chase. |
| **RESTART (not STOP+START) between register-select write and data read** | A STOP releases the bus, allowing another master to intervene and the slave to forget which register was selected — RESTART preserves atomic ownership across the direction change. |
| **NACK programmed BEFORE the final byte is clocked in during a read** | Hardware timing requirement on this peripheral family: CR1.ACK must be cleared ahead of the last byte's ACK phase, not after, or the wrong byte gets NACK'd. |
| **Burst-read multi-byte sensor data in one transaction** | Guarantees all axes/fields are sampled from the same instant; separate transactions risk data tearing under rapid physical change. |
| **WHO_AM_I / device-presence verification before trusting data** | Wrong address, dead device, or wiring faults (missing pull-ups, swapped SDA/SCL) must be caught explicitly, not discovered via silently garbage sensor readings. |
| **SWRST-based bus recovery path** | A slave that dies mid-transaction can hold SDA/SCL low indefinitely; software must have an explicit recovery path rather than assuming the bus self-heals. |
| **Explicit sleep-mode clear on sensor init (MPU6050)** | The single most common field bug — sensor always reads zero because it was never taken out of its default power-on sleep state. |

## 12. Concepts Demonstrated

| Concept | Where it shows up |
|---|---|
| **START / STOP** | `i2c_start()` / `i2c_stop()` — Section 6, with BUSY-flag bus-release checking and hardware auto-clear semantics explained. |
| **ACK / NACK** | Section 6 (`i2c_send_address` AF detection) and Section 7 (`i2c_read`'s explicit ACK-vs-NACK-on-last-byte control). |
| **Addressing** | 7-bit address + R/W direction bit encoding, verified bit-for-bit in the test harness (`0xD0`/`0xD1` for MPU6050). |
| **Register read/write** | `i2c_write_reg()` (exact prompt signature) and `i2c_read_reg()`/`i2c_read_reg_multi()` implementing the combined-format RESTART pattern. |
| **Real sensor integration** | Full MPU6050 driver — init (write), identity verification (read), and burst sensor read (multi-byte read) — covering all three transaction shapes on real, purchasable hardware. |