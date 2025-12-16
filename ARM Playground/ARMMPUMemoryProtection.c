Here's a production-grade **MPU (Memory Protection Unit) configuration** for ARM Cortex-M (ARMv7-M architecture — e.g., Cortex-M4/M7), building on the fault-handling and register-introspection work from earlier. This enforces the three-region policy you specified and includes a full illegal-access test with fault decoding.

## 1. MPU Register Overlay (ARMv7-M System Control Space)

```c
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* ---- MPU registers, fixed SCS location per ARMv7-M ARM ---- */
typedef struct {
    volatile uint32_t TYPE;    // 0xE000ED90 - region count, separate I/D support
    volatile uint32_t CTRL;    // 0xE000ED94 - enable, default map, priv-fault behavior
    volatile uint32_t RNR;     // 0xE000ED98 - region number select
    volatile uint32_t RBAR;    // 0xE000ED9C - base address + region select alias
    volatile uint32_t RASR;    // 0xE000EDA0 - size, attributes, enable
} MPU_TypeDef;

#define MPU_BASE   0xE000ED90UL
#define MPU        ((MPU_TypeDef *)MPU_BASE)

/* System Control Block - needed to enable MemManage fault + read fault status */
typedef struct {
    volatile uint32_t CPUID;
    volatile uint32_t ICSR;
    volatile uint32_t VTOR;
    volatile uint32_t AIRCR;
    volatile uint32_t SCR;
    volatile uint32_t CCR;
    volatile uint8_t  SHP[12];
    volatile uint32_t SHCSR;   // 0xE000ED24 - enables MemManage/Bus/Usage faults
    volatile uint32_t CFSR;    // 0xE000ED28 - MMFSR|BFSR|UFSR combined
    volatile uint32_t HFSR;    // 0xE000ED2C - HardFault status
    volatile uint32_t DFSR;
    volatile uint32_t MMFAR;   // 0xE000ED34 - faulting address (MemManage)
    volatile uint32_t BFAR;    // 0xE000ED38 - faulting address (BusFault)
} SCB_TypeDef;

#define SCB_BASE   0xE000ED00UL
#define SCB        ((SCB_TypeDef *)SCB_BASE)

/* ---- MPU_CTRL bits ---- */
#define MPU_CTRL_ENABLE       (1U << 0)   // Enable MPU
#define MPU_CTRL_HFNMIENA     (1U << 1)   // MPU active during HardFault/NMI
#define MPU_CTRL_PRIVDEFENA   (1U << 2)   // Privileged code uses default map for unmapped regions

/* ---- MPU_RASR field helpers ---- */
#define MPU_RASR_ENABLE       (1U << 0)
#define MPU_RASR_SIZE(n)      ((uint32_t)(n) << 1)     // region size = 2^(n+1) bytes
#define MPU_RASR_SRD(x)       ((uint32_t)(x) << 8)      // subregion disable
#define MPU_RASR_B            (1U << 16)                // bufferable
#define MPU_RASR_C            (1U << 17)                // cacheable
#define MPU_RASR_S            (1U << 18)                // shareable
#define MPU_RASR_TEX(x)       ((uint32_t)(x) << 19)     // type extension field
#define MPU_RASR_AP(x)        ((uint32_t)(x) << 24)     // access permission
#define MPU_RASR_XN           (1U << 28)                // execute-never

/* ---- AP encodings (Table B3-15, ARMv7-M ARM) ---- */
#define AP_NO_ACCESS          0x0
#define AP_PRIV_RW_UNPRIV_NO  0x1
#define AP_PRIV_RW_UNPRIV_RO  0x2
#define AP_PRIV_RW_UNPRIV_RW  0x3   // full read/write, both privilege levels
#define AP_PRIV_RO_UNPRIV_NO  0x5
#define AP_PRIV_RO_UNPRIV_RO  0x6

/* ---- SHCSR bit for enabling MemManage fault as a distinct exception ---- */
#define SHCSR_MEMFAULTENA     (1U << 16)
#define SHCSR_BUSFAULTENA     (1U << 17)
#define SHCSR_USGFAULTENA     (1U << 18)

/* ---- CFSR sub-fields (MMFSR occupies bits [7:0]) ---- */
#define CFSR_MMFSR_IACCVIOL   (1U << 0)   // instruction fetch from protected region
#define CFSR_MMFSR_DACCVIOL   (1U << 1)   // data access violation
#define CFSR_MMFSR_MUNSTKERR  (1U << 3)   // fault unstacking on exception return
#define CFSR_MMFSR_MSTKERR    (1U << 4)   // fault stacking on exception entry
#define CFSR_MMFSR_MMARVALID  (1U << 7)   // MMFAR holds a valid faulting address
```

## 2. Region Size Encoding Helper

MPU region sizes must be power-of-two per the ARMv7-M protocol (`REGION_SIZE = 2^(SIZE_FIELD+1)`), and base addresses must be naturally aligned to that size.

```c
/*
 * Converts a byte size to the MPU RASR SIZE field.
 * Example: 1MB (0x100000) -> log2(0x100000) - 1 = 19
 * Enforces the ARMv7-M rule that region size is always a power of two >= 32 bytes.
 */
static uint32_t mpu_size_field(uint32_t bytes)
{
    uint32_t log2_size = 0;
    uint32_t n = bytes;

    while (n > 1) {
        n >>= 1;
        log2_size++;
    }
    return log2_size - 1;   // RASR SIZE field is (log2(bytes) - 1)
}
```

## 3. Region Configuration — Code / RAM / Peripheral

```c
#define REGION_CODE        0
#define REGION_RAM         1
#define REGION_PERIPH      2

/*
 * Programs one MPU region. Discipline mirrors GPIO/UART drivers:
 * select region (RNR), then write RBAR/RASR, enable last via RASR.ENABLE.
 * MPU must be globally disabled while regions are (re)programmed to avoid
 * a transient window where an old, conflicting region definition is live.
 */
static void mpu_configure_region(uint8_t region_num, uint32_t base_addr,
                                  uint32_t size_bytes, uint32_t ap,
                                  uint32_t tex_scb, bool xn)
{
    MPU->RNR  = region_num;
    MPU->RBAR = base_addr & 0xFFFFFFE0UL;   // base must be size-aligned; low 5 bits reserved

    uint32_t rasr = MPU_RASR_SIZE(mpu_size_field(size_bytes))
                  | MPU_RASR_AP(ap)
                  | tex_scb
                  | MPU_RASR_ENABLE;

    if (xn) {
        rasr |= MPU_RASR_XN;
    }

    MPU->RASR = rasr;
}

/*
 * Full three-region policy:
 *
 *  Region 0: 0x00000000-0x000FFFFF (1MB) Code    - Read+Execute,  Normal memory, WT cacheable
 *  Region 1: 0x20000000-0x2000FFFF (64KB) RAM    - Read+Write,    Normal memory, WB cacheable, no-execute
 *  Region 2: 0x40000000-0x400FFFFF (1MB) Periph  - Read+Write,    Device memory (strongly ordered), no-execute
 *
 * Any address NOT covered by an enabled region and NOT covered by the
 * default background map (disabled here for unprivileged code) faults.
 */
void mpu_init(void)
{
    MPU->CTRL = 0;   // Disable MPU entirely while (re)configuring regions

    /* Region 0: Flash/Code -- Normal memory, cacheable, execute allowed (XN=0) */
    mpu_configure_region(
        REGION_CODE,
        0x00000000UL,
        0x00100000UL,                          // 1MB
        AP_PRIV_RO_UNPRIV_RO,                  // Read-only for both privilege levels
        MPU_RASR_TEX(0) | MPU_RASR_C,           // Normal memory, write-through cacheable
        false                                    // XN = 0 -> execution permitted
    );

    /* Region 1: SRAM -- Normal memory, read/write, execute-never */
    mpu_configure_region(
        REGION_RAM,
        0x20000000UL,
        0x00010000UL,                          // 64KB
        AP_PRIV_RW_UNPRIV_RW,                  // Full read/write both levels
        MPU_RASR_TEX(0) | MPU_RASR_C | MPU_RASR_B, // Normal memory, write-back cacheable+bufferable
        true                                     // XN = 1 -> code cannot execute from RAM (W^X policy)
    );

    /* Region 2: Peripherals -- Device memory: no caching, no speculative access, no reordering */
    mpu_configure_region(
        REGION_PERIPH,
        0x40000000UL,
        0x00100000UL,                          // 1MB peripheral block
        AP_PRIV_RW_UNPRIV_RW,
        MPU_RASR_TEX(2) | MPU_RASR_S,           // TEX=2,C=0,B=0 -> Device memory, shareable
        true                                     // XN = 1 -> peripherals are never executable
    );

    /* Enable MemManage as its own fault, distinct from generic HardFault escalation */
    SCB->SHCSR |= SHCSR_MEMFAULTENA;

    /*
     * PRIVDEFENA=0 deliberately: with it clear, any address NOT covered by
     * an explicit region above is inaccessible even to privileged code.
     * This is the strict policy -- background map is intentionally denied
     * so unmapped/undocumented peripheral holes fault loudly instead of
     * silently succeeding via the default memory map.
     */
    MPU->CTRL = MPU_CTRL_ENABLE | MPU_CTRL_HFNMIENA;

    __asm volatile ("dsb");   // Ensure region writes complete before ISB
    __asm volatile ("isb");   // Flush pipeline so new memory map applies to next fetch
}
```

## 4. MemManage Fault Handler — Decode and Report

```c
/*
 * Fault frame layout matches the earlier HardFault_Handler work: naked entry
 * captures the auto-stacked frame, this function does the analysis.
 * MemManage faults carry richer diagnostic info than generic HardFault:
 * CFSR.MMFSR tells you WHY, MMFAR tells you WHERE (if MMARVALID is set).
 */
typedef struct {
    uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
} fault_stack_frame_t;

void MemManage_Handler_C(fault_stack_frame_t *frame)
{
    uint32_t cfsr = SCB->CFSR;
    uint8_t  mmfsr = (uint8_t)(cfsr & 0xFF);

    printf("\n=== MemManage Fault ===\n");
    printf("Faulting PC : 0x%08lX\n", (unsigned long)frame->pc);
    printf("LR (EXC_RET): 0x%08lX\n", (unsigned long)frame->lr);
    printf("MMFSR       : 0x%02X\n", mmfsr);

    if (mmfsr & CFSR_MMFSR_IACCVIOL) {
        printf("  -> Instruction access violation: attempted EXECUTE from a\n");
        printf("     protected/XN region (e.g., RAM with XN=1).\n");
    }
    if (mmfsr & CFSR_MMFSR_DACCVIOL) {
        printf("  -> Data access violation: read/write blocked by region AP bits\n");
        printf("     (e.g., write to a read-only Code region).\n");
    }
    if (mmfsr & CFSR_MMFSR_MSTKERR) {
        printf("  -> Fault occurred while STACKING for exception entry.\n");
    }
    if (mmfsr & CFSR_MMFSR_MUNSTKERR) {
        printf("  -> Fault occurred while UNSTACKING on exception return.\n");
    }
    if (mmfsr & CFSR_MMFSR_MMARVALID) {
        printf("  -> Faulting address (MMFAR): 0x%08lX\n", (unsigned long)SCB->MMFAR);
    } else {
        printf("  -> MMFAR not valid for this fault type.\n");
    }

    /* Clear MMFSR by writing 1s to the fault status bits (W1C, like EXTI_PR) */
    SCB->CFSR = mmfsr;

    /*
     * Production policy: never blindly resume into the faulting instruction.
     * Options: terminate offending task (RTOS), reset subsystem, or halt
     * for debug. Here we halt to keep the fault state visible under Trace32.
     */
    while (1) {
        __asm volatile ("wfi");
    }
}

/*
 * Naked entry -- identical discipline to HardFault_Handler: capture the
 * live stack pointer (MSP or PSP, decoded from EXC_RETURN bit 2) and hand
 * off to the C analysis routine without corrupting the auto-stacked frame.
 */
__attribute__((naked)) void MemManage_Handler(void)
{
    __asm volatile (
        "tst lr, #4              \n"   // bit 2 of EXC_RETURN: 0=MSP, 1=PSP
        "ite eq                  \n"
        "mrseq r0, msp           \n"
        "mrsne r0, psp           \n"
        "b MemManage_Handler_C   \n"
    );
}
```

## 5. Test Harness — Deliberate Illegal Accesses

```c
/* Test 1: Data access violation -- write into the Code region (read-only) */
void test_illegal_write_to_code(void)
{
    volatile uint32_t *code_ptr = (volatile uint32_t *)0x00001000UL; // inside 1MB code region
    printf("Attempting write to Code region (should fault: read-only)...\n");
    *code_ptr = 0xDEADBEEF;   // Traps into MemManage_Handler via DACCVIOL
    printf("This line should NEVER print.\n");
}

/* Test 2: Instruction access violation -- execute from RAM region (XN=1) */
typedef void (*func_ptr_t)(void);

void test_illegal_execute_from_ram(void)
{
    uint8_t ram_stub[8] __attribute__((aligned(4))) = {
        0x70, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   // "BX LR" opcode + padding
    };
    func_ptr_t fn = (func_ptr_t)((uint32_t)ram_stub | 1); // +1 for Thumb bit

    printf("Attempting to execute code fetched from RAM (should fault: XN)...\n");
    fn();   // Traps into MemManage_Handler via IACCVIOL
    printf("This line should NEVER print.\n");
}

/* Test 3: Access to a completely unmapped address (no covering region, PRIVDEFENA=0) */
void test_illegal_unmapped_access(void)
{
    volatile uint32_t *unmapped = (volatile uint32_t *)0x90000000UL;
    printf("Attempting read from unmapped address (should fault: no region covers it)...\n");
    uint32_t val = *unmapped;
    (void)val;
    printf("This line should NEVER print.\n");
}

int main(void)
{
    mpu_init();
    printf("MPU configured: Code(RO+X) / RAM(RW+XN) / Peripheral(RW+XN, Device mem)\n");

    test_illegal_write_to_code();
    // test_illegal_execute_from_ram();
    // test_illegal_unmapped_access();

    while (1) { }
}
```

## 6. Region Summary Table

| Region | Address Range | Size | AP | Memory Type | XN | Rationale |
|---|---|---|---|---|---|---|
| 0: Code | 0x00000000–0x000FFFFF | 1MB | RO/RO | Normal, WT cacheable | 0 | Execution allowed; write-protected against runtime corruption |
| 1: RAM | 0x20000000–0x2000FFFF | 64KB | RW/RW | Normal, WB cacheable+bufferable | 1 | W^X enforcement — data-only, blocks code-injection execution |
| 2: Peripheral | 0x40000000–0x400FFFFF | 1MB | RW/RW | Device (TEX=2, no cache/reorder) | 1 | Prevents speculative/reordered access to registers with side effects |
| (background) | everything else | — | — | — | — | PRIVDEFENA=0 → faults loudly instead of silently succeeding |

## 7. Key Concepts Demonstrated

- **Memory attributes vs access permissions**: `AP`/`XN` control *who can do what* (privilege + operation), while `TEX/C/B/S` control *how the bus treats the transaction* (cacheable, bufferable, ordering guarantees). Conflating these is a common bug — e.g., marking peripherals as Normal+cacheable causes stale-read bugs identical in spirit to the DMA/cache-coherency Heisenbugs from the earlier CMO work.
- **W^X (write XOR execute) enforcement**: Region 1 (RAM) explicitly sets `XN=1`, closing the classic code-injection vector where an attacker writes shellcode to a data buffer and redirects `PC` into it — directly relevant to the TrustZone/TZASC secure-heap work, where MPU acts as the EL0/unprivileged analog to TZASC's Secure/Non-secure DRAM partitioning.
- **PRIVDEFENA=0 as a strict-fail posture**: without it, any address outside your three explicit regions silently falls back to the ARMv7-M default memory map — undermining the entire point of defining regions. Clearing it means *every* access must be explicitly authorized, matching the deny-by-default posture used in the TZASC region descriptors.
- **Fault status is authoritative, not the instruction that trapped**: `CFSR.MMFSR` distinguishes instruction-fetch violations (`IACCVIOL`) from data violations (`DACCVIOL`) from stacking errors — critical for correctly triaging whether the bug is a wild pointer, a stack overflow corrupting an adjacent region, or a genuine code-injection attempt.
- **Privileged vs unprivileged split**: the `AP` encoding table lets you grant, e.g., privileged RW + unprivileged RO on the same physical region — the mechanism an RTOS uses to let kernel code freely modify state that user tasks can only read, extending the same privilege-separation model as `EL0` vs `EL1` PSTATE access rules from the AArch64 introspection work.