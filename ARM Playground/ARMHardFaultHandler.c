Here's a complete **HardFault Handler** implementation for ARM Cortex-M, progressing from the minimal stub you specified to the full stacked-register extraction and fault decoding — extending the naked-handler pattern from my earlier bare-metal fault-handling work.

## 1. Minimal Handler (Starting Point)

```c
void HardFault_Handler(void)
{
    while (1)
    {
        /* Debug fault - attach debugger, inspect call stack manually */
    }
}
```

This works but tells you nothing on its own — you're stuck manually walking the stack in a debugger with no automated decode of *why* the fault happened. The advanced version below fixes that.

## 2. Triggering Fault — NULL Pointer Dereference

```c
void trigger_null_pointer_fault(void)
{
    int *ptr = NULL;
    printf("About to dereference NULL...\n");
    *ptr = 10;                          // Write to address 0x00000000
    printf("This line should NEVER execute.\n");
}
```

On most Cortex-M parts, address `0x00000000` maps into the vector table / Flash region. Depending on MPU configuration (see prior MPU work) and whether that address is executable-but-not-writable, this typically raises either:
- A **precise BusFault** (if escalated) or
- A **HardFault** directly, if `BusFault`/`MemManage` are disabled in `SHCSR` (their default reset state) — in which case all faults escalate to HardFault per the ARMv7-M "fault escalation" rule.

## 3. Register Overlay (SCB Fault Status)

```c
#include <stdint.h>
#include <stdio.h>

typedef struct {
    volatile uint32_t CPUID;
    volatile uint32_t ICSR;
    volatile uint32_t VTOR;
    volatile uint32_t AIRCR;
    volatile uint32_t SCR;
    volatile uint32_t CCR;
    volatile uint8_t  SHP[12];
    volatile uint32_t SHCSR;   // 0xE000ED24
    volatile uint32_t CFSR;    // 0xE000ED28 - MMFSR[7:0] | BFSR[15:8] | UFSR[31:16]
    volatile uint32_t HFSR;    // 0xE000ED2C
    volatile uint32_t DFSR;
    volatile uint32_t MMFAR;   // 0xE000ED34
    volatile uint32_t BFAR;    // 0xE000ED38
} SCB_TypeDef;

#define SCB   ((SCB_TypeDef *)0xE000ED00UL)

/* HFSR bits */
#define HFSR_VECTTBL    (1U << 1)    // fault reading the vector table itself
#define HFSR_FORCED     (1U << 30)   // escalated from MemManage/Bus/Usage fault
#define HFSR_DEBUGEVT   (1U << 31)

/* CFSR.BFSR (bits [15:8]) */
#define BFSR_IBUSERR    (1U << 8)
#define BFSR_PRECISERR  (1U << 9)    // precise bus fault - faulting address known
#define BFSR_IMPRECISERR (1U << 10)  // imprecise - address NOT reliable
#define BFSR_UNSTKERR   (1U << 11)
#define BFSR_STKERR     (1U << 12)
#define BFSR_BFARVALID  (1U << 15)

/* CFSR.MMFSR (bits [7:0]) */
#define MMFSR_IACCVIOL  (1U << 0)
#define MMFSR_DACCVIOL  (1U << 1)
#define MMFSR_MUNSTKERR (1U << 3)
#define MMFSR_MSTKERR   (1U << 4)
#define MMFSR_MMARVALID (1U << 7)

/* CFSR.UFSR (bits [31:16]) */
#define UFSR_UNDEFINSTR (1U << 16)
#define UFSR_INVSTATE   (1U << 17)
#define UFSR_INVPC      (1U << 18)
#define UFSR_NOCP       (1U << 19)
#define UFSR_UNALIGNED  (1U << 24)
#define UFSR_DIVBYZERO  (1U << 25)
```

## 4. Advanced Handler — Extract Stacked Registers

```c
/*
 * Hardware auto-stacks these 8 words on exception entry, in this exact order,
 * regardless of which stack (MSP/PSP) was active. This is architectural,
 * not implementation-defined -- same layout used in my earlier ARMv7-M
 * core-register introspection work.
 */
typedef struct {
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;      // return address into the faulting caller (not the fault itself)
    uint32_t pc;       // address of the faulting instruction
    uint32_t xpsr;     // condition flags + exception number at fault time
} fault_stack_frame_t;

/*
 * C-level analysis routine. Receives the stack pointer that was ACTIVE
 * at the moment of the fault (MSP or PSP), decoded by the naked entry
 * stub below from EXC_RETURN bit 2.
 */
void HardFault_Handler_C(fault_stack_frame_t *frame)
{
    uint32_t cfsr = SCB->CFSR;
    uint32_t hfsr = SCB->HFSR;
    uint8_t  mmfsr = (uint8_t)(cfsr & 0xFF);
    uint8_t  bfsr  = (uint8_t)((cfsr >> 8) & 0xFF);
    uint16_t ufsr  = (uint16_t)((cfsr >> 16) & 0xFFFF);

    printf("\n========== HardFault ==========\n");
    printf("R0  = 0x%08lX   R1  = 0x%08lX\n", (unsigned long)frame->r0, (unsigned long)frame->r1);
    printf("R2  = 0x%08lX   R3  = 0x%08lX\n", (unsigned long)frame->r2, (unsigned long)frame->r3);
    printf("R12 = 0x%08lX   LR  = 0x%08lX\n", (unsigned long)frame->r12, (unsigned long)frame->lr);
    printf("PC  = 0x%08lX  <- faulting instruction address\n", (unsigned long)frame->pc);
    printf("xPSR= 0x%08lX\n", (unsigned long)frame->xpsr);

    printf("\n-- Fault Status --\n");
    printf("HFSR = 0x%08lX\n", (unsigned long)hfsr);
    if (hfsr & HFSR_FORCED) {
        printf("  FORCED: escalated from a configurable fault (Bus/Mem/Usage)\n");
        printf("  because that fault's handler was disabled or lower priority.\n");
    }
    if (hfsr & HFSR_VECTTBL) {
        printf("  VECTTBL: fault while reading the vector table itself.\n");
    }

    printf("CFSR = 0x%08lX  (UFSR=0x%04X BFSR=0x%02X MMFSR=0x%02X)\n",
           (unsigned long)cfsr, ufsr, bfsr, mmfsr);

    if (bfsr & BFSR_PRECISERR) {
        printf("  BFSR.PRECISERR: precise BusFault.\n");
        if (bfsr & BFSR_BFARVALID) {
            printf("    Faulting address (BFAR): 0x%08lX\n", (unsigned long)SCB->BFAR);
        }
    }
    if (bfsr & BFSR_IMPRECISERR) {
        printf("  BFSR.IMPRECISERR: imprecise BusFault - BFAR is UNRELIABLE.\n");
        printf("    (write buffering delayed the fault past the triggering instr.)\n");
    }
    if (mmfsr & MMFSR_DACCVIOL) {
        printf("  MMFSR.DACCVIOL: MPU-blocked data access.\n");
        if (mmfsr & MMFSR_MMARVALID) {
            printf("    Faulting address (MMFAR): 0x%08lX\n", (unsigned long)SCB->MMFAR);
        }
    }
    if (ufsr & UFSR_UNDEFINSTR) {
        printf("  UFSR.UNDEFINSTR: undefined instruction executed.\n");
    }
    if (ufsr & UFSR_INVPC) {
        printf("  UFSR.INVPC: invalid PC load, e.g. bad exception return.\n");
    }
    if (ufsr & UFSR_UNALIGNED) {
        printf("  UFSR.UNALIGNED: unaligned access trapped (if UNALIGN_TRP set in CCR).\n");
    }
    if (ufsr & UFSR_DIVBYZERO) {
        printf("  UFSR.DIVBYZERO: integer divide-by-zero trapped (if DIV_0_TRP set).\n");
    }

    /*
     * Decode this specific example: PC will point at the STR instruction
     * that wrote to *ptr, R0 (or whichever register held ptr, per calling
     * convention/compiler) will read back as 0x00000000, and BFAR/MMFAR
     * (whichever is valid) will show 0x00000000 as the faulting address --
     * confirming this was the NULL write, not something else.
     */
    printf("\n-- Diagnosis --\n");
    printf("If faulting address == 0x00000000, this confirms a NULL pointer write.\n");

    /* Clear fault status registers (W1C) so a future fault isn't masked by stale bits */
    SCB->CFSR = cfsr;

    /*
     * Production policy: do NOT return from HardFault via a corrupted frame.
     * Halt here for the debugger (Trace32/JTAG) to inspect full state, or
     * in a shipped product: log to non-volatile fault record, then reset
     * via SCB->AIRCR (SYSRESETREQ) for graceful recovery.
     */
    while (1) {
        __asm volatile ("nop");   // debugger breakpoint target
    }
}

/*
 * Naked entry point -- this MUST be the actual vector table entry.
 * It performs zero register-clobbering setup before reading the stack
 * pointer, because the compiler-generated prologue of a normal C function
 * would push registers and corrupt our view of the fault frame.
 *
 * EXC_RETURN bit 2 (value 0x4) tells us which stack was active:
 *   0 -> handler was using MSP (fault occurred in another handler / early boot)
 *   1 -> handler was using PSP (fault occurred in normal thread-mode code)
 */
__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile (
        "tst lr, #4                \n"
        "ite eq                    \n"
        "mrseq r0, msp             \n"
        "mrsne r0, psp             \n"
        "b HardFault_Handler_C     \n"
    );
}
```

## 5. Enabling Precise Fault Reporting (Setup Required Before Trigger)

```c
#define SHCSR_MEMFAULTENA  (1U << 16)
#define SHCSR_BUSFAULTENA  (1U << 17)
#define SHCSR_USGFAULTENA  (1U << 18)
#define CCR_DIV_0_TRP      (1U << 4)
#define CCR_UNALIGN_TRP    (1U << 3)

void fault_handling_init(void)
{
    /*
     * Without enabling BusFault/MemManage/UsageFault individually, ALL
     * faults escalate directly to HardFault with HFSR.FORCED set, and you
     * lose the granular MMFSR/BFSR/UFSR detail -- you'd only know "a fault
     * happened," not which kind or why. Enable them so specific handlers
     * fire (or at minimum, so CFSR sub-fields populate before escalation).
     */
    SCB->SHCSR |= SHCSR_MEMFAULTENA | SHCSR_BUSFAULTENA | SHCSR_USGFAULTENA;

    /* Optional: trap divide-by-zero and unaligned access as UsageFaults
     * instead of silently returning garbage or a wrong result. */
    SCB->CCR |= CCR_DIV_0_TRP | CCR_UNALIGN_TRP;
}
```

## 6. Full Test Harness

```c
int main(void)
{
    fault_handling_init();

    printf("System up. Triggering intentional NULL pointer write...\n");
    trigger_null_pointer_fault();

    /* unreachable */
    while (1) { }
}
```

## Expected Output on Fault

```
About to dereference NULL...

========== HardFault ==========
R0  = 0x0000000A   R1  = 0x20001F00   ...
PC  = 0x08000142  <- faulting instruction address
xPSR= 0x61000000

-- Fault Status --
HFSR = 0x40000000
  FORCED: escalated from a configurable fault (Bus/Mem/Usage)
CFSR = 0x00008200  (UFSR=0x0000 BFSR=0x82 MMFSR=0x00)
  BFSR.PRECISERR: precise BusFault.
    Faulting address (BFAR): 0x00000000

-- Diagnosis --
If faulting address == 0x00000000, this confirms a NULL pointer write.
```

*(Exact fault type — MemFault vs BusFault — depends on target: on a part with MPU denying access to `0x0`, it's `MMFSR.DACCVIOL`; on a part with no MPU region covering `0x0` at all, it's typically a `BFSR.PRECISERR` bus fault on the AHB access itself. Both escalate to `HardFault_Handler` if their individual fault handlers aren't separately vectored, or route to the dedicated `BusFault_Handler`/`MemManage_Handler` if `SHCSR` enables are set — which is why step 5's setup matters.)*

## Key Concepts Demonstrated

- **Fault escalation hierarchy**: MemManage → BusFault → UsageFault → HardFault, controlled by `SHCSR` enable bits; disabling the specific handlers forces everything through `HardFault_Handler` with `HFSR.FORCED` set, at the cost of granularity.
- **Naked handler discipline**: identical to the ARMv7-M core-register introspection and MPU fault work — a normal C function prologue would stack registers *before* you can read the hardware-stacked frame, corrupting your diagnostic view. `__attribute__((naked))` is mandatory here.
- **Precise vs imprecise faults**: `BFSR.PRECISERR` means `BFAR` reliably points at the faulting address; `IMPRECISERR` means store-buffer reordering broke that guarantee — a critical distinction when triaging memory corruption vs. simple NULL derefs.
- **EXC_RETURN bit 2 (MSP/PSP selection)**: reused directly from the earlier AArch64/ARMv7-M exception-frame analysis — necessary because the fault could occur in either handler-mode (MSP) or thread-mode (PSP) context, and reading the wrong stack pointer yields garbage register values.
- **W1C clearing of CFSR**: prevents stale fault bits from masking or confusing diagnosis of the *next* fault, following the same write-1-to-clear discipline used throughout the EXTI/DMA/MPU work.