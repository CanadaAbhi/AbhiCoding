# Reading ARM Core Registers (Cortex-M / ARMv7-M baseline, ARMv8-A notes included)

## 1. Register Map

```
ARM Core Registers (ARMv7-M / Cortex-M, e.g. Cortex-M0/M3/M4/M7)
 |
 +-- R0-R12   General purpose (R0-R3 = scratch/args, R4-R11 = callee-saved,
 |                              R12/IP = intra-procedure-call scratch)
 |
 +-- R13 (SP) Stack Pointer -- banked: MSP (Main) / PSP (Process)
 |
 +-- R14 (LR) Link Register -- return address / EXC_RETURN on exception entry
 |
 +-- R15 (PC) Program Counter -- always reads as (current instr addr + 4) in ARM state
 |
 +-- xPSR     Combined Program Status Register
       |
       +-- APSR  bits[31:27] N,Z,C,V,Q  (ALU flags)
       +-- IPSR  bits[8:0]   current exception number (0 = Thread mode)
       +-- EPSR  bit[24]     T-bit (Thumb state), bits[15:10,26:25] IT[7:0]
```

## 2. Full Implementation — Bare-Metal Cortex-M

```c
/* ============================================================
 * read_cpu_registers.c
 * Bare-metal ARMv7-M (Cortex-M0/M3/M4/M7) register introspection.
 *
 * Build (example, Cortex-M4):
 *   arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O0 -g \
 *       read_cpu_registers.c -o regs.elf
 *
 * NOTE: -O0 is deliberate. At higher optimization levels the
 * compiler is free to reuse r0-r12 for its own purposes between
 * the register-variable binding and the point you read it, which
 * would report stale/garbage values. This file is a *debug/
 * educational* tool, not something to inline into optimized
 * production paths -- production code instead captures registers
 * automatically via exception stacking (Section 4) or a debugger
 * (Lauterbach Trace32, J-Link) which reads the physical register
 * file directly without disturbing program state at all.
 * ============================================================ */

#include <stdint.h>
#include <stdio.h>

/* ---------------------------------------------------------------
 * SP (R13) -- MSP: Main Stack Pointer
 * On Cortex-M, "SP" is an alias that resolves to MSP or PSP
 * depending on CONTROL.SPSEL. MRS lets us name the exact bank.
 * --------------------------------------------------------------- */
static inline uint32_t get_msp(void)
{
    uint32_t value;
    __asm volatile ("MRS %0, MSP" : "=r"(value));
    return value;
}

static inline uint32_t get_psp(void)
{
    uint32_t value;
    __asm volatile ("MRS %0, PSP" : "=r"(value));
    return value;
}

/* Generic SP read -- whichever stack pointer is currently active
 * (MSP in Handler mode or Thread mode w/ SPSEL=0, PSP otherwise). */
static inline uint32_t get_sp(void)
{
    uint32_t value;
    __asm volatile ("MOV %0, SP" : "=r"(value));
    return value;
}

/* ---------------------------------------------------------------
 * LR (R14) -- Link Register.
 * In normal code this holds the return address of the calling
 * function. Inside an exception handler it instead holds
 * EXC_RETURN (a magic value like 0xFFFFFFF9) that tells the
 * processor how to unstack and which stack/mode to return to.
 * --------------------------------------------------------------- */
static inline uint32_t get_lr(void)
{
    uint32_t value;
    __asm volatile ("MOV %0, LR" : "=r"(value));
    return value;
}

/* ---------------------------------------------------------------
 * PC (R15) -- Program Counter.
 * There is no MRS for PC on ARMv7-M; the architecturally correct
 * way to sample it is the ADR pseudo-instruction, which computes
 * a PC-relative address at the current instruction -- giving you
 * PC at the point of the ADR, not some stale/pipelined value.
 * --------------------------------------------------------------- */
static inline uint32_t get_pc(void)
{
    uint32_t value;
    __asm volatile ("ADR %0, ." : "=r"(value));
    return value;
}

/* ---------------------------------------------------------------
 * xPSR -- combined Application/Interrupt/Execution PSR.
 * Cortex-M-specific: a single MRS instruction reads all three
 * logical views packed into one 32-bit register.
 * --------------------------------------------------------------- */
static inline uint32_t get_xpsr(void)
{
    uint32_t value;
    __asm volatile ("MRS %0, xPSR" : "=r"(value));
    return value;
}

/* ---------------------------------------------------------------
 * R0-R12 -- General purpose registers.
 *
 * These have no MRS/dedicated read instruction; the *only* way to
 * observe "the current value of physical register Rn" is to bind
 * a C variable to that exact register using GCC's explicit
 * register variable extension, then read it before the compiler
 * has a chance to reuse the register for something else.
 *
 * `register uint32_t r0 asm("r0");` tells GCC: "this C variable
 * IS r0, do not spill it, do not reassign it." Reading it
 * immediately in the same statement/expression captures the true
 * physical register content at that program point.
 * --------------------------------------------------------------- */
static void dump_gpr_snapshot(void)
{
    register uint32_t r0  asm("r0");
    register uint32_t r1  asm("r1");
    register uint32_t r2  asm("r2");
    register uint32_t r3  asm("r3");
    register uint32_t r4  asm("r4");
    register uint32_t r5  asm("r5");
    register uint32_t r6  asm("r6");
    register uint32_t r7  asm("r7");
    register uint32_t r8  asm("r8");
    register uint32_t r9  asm("r9");
    register uint32_t r10 asm("r10");
    register uint32_t r11 asm("r11");
    register uint32_t r12 asm("r12");

    /* An empty asm volatile with the registers listed as inputs
     * forces GCC to materialize their CURRENT contents into these
     * bindings right here, rather than optimizing the reads away. */
    __asm volatile ("" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
                          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
                          "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
                          "=r"(r12));

    printf("  R0  = 0x%08lX   R1  = 0x%08lX   R2  = 0x%08lX   R3  = 0x%08lX\n",
           (unsigned long)r0, (unsigned long)r1, (unsigned long)r2, (unsigned long)r3);
    printf("  R4  = 0x%08lX   R5  = 0x%08lX   R6  = 0x%08lX   R7  = 0x%08lX\n",
           (unsigned long)r4, (unsigned long)r5, (unsigned long)r6, (unsigned long)r7);
    printf("  R8  = 0x%08lX   R9  = 0x%08lX   R10 = 0x%08lX   R11 = 0x%08lX\n",
           (unsigned long)r8, (unsigned long)r9, (unsigned long)r10, (unsigned long)r11);
    printf("  R12 = 0x%08lX (IP -- intra-procedure-call scratch)\n",
           (unsigned long)r12);
}

/* ---------------------------------------------------------------
 * xPSR bit-field decoder.
 *
 *  31 30 29 28 27 26 25 24 23..20 19..16 15..10 9  8..0
 *  N  Z  C  V  Q  ---T---  ---reserved---  ICI/IT    ICI/IT  T  EXCEPTION#
 *
 * Simplified practical decode (sufficient for debug output):
 *   N (bit31), Z (bit30), C (bit29), V (bit28), Q (bit27) -- APSR flags
 *   T (bit24)                                              -- Thumb state (EPSR)
 *   IPSR[8:0] (bits 8:0)                                   -- exception number
 * --------------------------------------------------------------- */
static void decode_xpsr(uint32_t xpsr)
{
    printf("  xPSR = 0x%08lX\n", (unsigned long)xpsr);
    printf("    N (Negative) = %u\n", (xpsr >> 31) & 1U);
    printf("    Z (Zero)     = %u\n", (xpsr >> 30) & 1U);
    printf("    C (Carry)    = %u\n", (xpsr >> 29) & 1U);
    printf("    V (Overflow) = %u\n", (xpsr >> 28) & 1U);
    printf("    Q (Sticky Sat)= %u\n", (xpsr >> 27) & 1U);
    printf("    T (Thumb)    = %u  (must be 1 on Cortex-M -- ARM state unsupported)\n",
           (xpsr >> 24) & 1U);

    uint32_t exception_num = xpsr & 0x1FFU;
    printf("    IPSR (exception#) = %u  -> %s\n",
           exception_num,
           exception_num == 0   ? "Thread mode (no exception active)" :
           exception_num == 2   ? "NMI" :
           exception_num == 3   ? "HardFault" :
           exception_num == 11  ? "SVCall" :
           exception_num == 14  ? "PendSV" :
           exception_num == 15  ? "SysTick" :
           exception_num >= 16  ? "External IRQ (IRQn = exception# - 16)" :
                                   "Reserved/other system exception");
}

int main(void)
{
    printf("=============================================\n");
    printf("       ARM Cortex-M Core Register Dump        \n");
    printf("=============================================\n\n");

    printf("-- General Purpose Registers (R0-R12) --\n");
    dump_gpr_snapshot();

    printf("\n-- Stack Pointers --\n");
    printf("  SP  (active) = 0x%08lX\n", (unsigned long)get_sp());
    printf("  MSP           = 0x%08lX  (Main Stack Pointer)\n", (unsigned long)get_msp());
    printf("  PSP           = 0x%08lX  (Process Stack Pointer)\n", (unsigned long)get_psp());

    printf("\n-- Link Register --\n");
    printf("  LR  = 0x%08lX\n", (unsigned long)get_lr());

    printf("\n-- Program Counter --\n");
    printf("  PC  = 0x%08lX\n", (unsigned long)get_pc());

    printf("\n-- Program Status Register --\n");
    decode_xpsr(get_xpsr());

    while (1) { /* halt -- typical bare-metal end state */ }
    return 0;
}
```

## 3. Sample Output (Cortex-M4, semihosted printf, at Thread mode entry)

```
=============================================
       ARM Cortex-M Core Register Dump
=============================================

-- General Purpose Registers (R0-R12) --
  R0  = 0x00000000   R1  = 0x00000001   R2  = 0x20001FF0   R3  = 0x00000000
  R4  = 0x00000000   R5  = 0x00000000   R6  = 0x00000000   R7  = 0x20001FE8
  R8  = 0x00000000   R9  = 0x08000000   R10 = 0x00000000   R11 = 0x00000000
  R12 = 0x00000003 (IP -- intra-procedure-call scratch)

-- Stack Pointers --
  SP  (active) = 0x20001FE0
  MSP           = 0x20001FE0  (Main Stack Pointer)
  PSP           = 0x00000000  (Process Stack Pointer)

-- Link Register --
  LR  = 0x08000245

-- Program Counter --
  PC  = 0x0800024C

-- Program Status Register --
  xPSR = 0x01000000
    N (Negative) = 0
    Z (Zero)     = 0
    C (Carry)    = 0
    V (Overflow) = 0
    Q (Sticky Sat)= 0
    T (Thumb)    = 1  (must be 1 on Cortex-M -- ARM state unsupported)
    IPSR (exception#) = 0  -> Thread mode (no exception active)
```

## 4. Production Pattern: Capturing Registers in a HardFault Handler

The technique in Section 2 is fine for debug logging, but it disturbs the very state you're inspecting (function-call overhead can shift LR/SP slightly). The **real** production pattern used in fault handlers exploits the fact that Cortex-M **automatically stacks** R0-R3, R12, LR, PC, and xPSR on exception entry:

```c
/* ============================================================
 * Real fault-handler register capture -- no register-variable
 * tricks needed, because the hardware already pushed the exact
 * pre-fault register state onto the stack for us.
 *
 * On exception entry, Cortex-M automatically pushes, low->high
 * address:
 *   [sp+0]  R0
 *   [sp+4]  R1
 *   [sp+8]  R2
 *   [sp+12] R3
 *   [sp+16] R12
 *   [sp+20] LR   (pre-fault LR)
 *   [sp+24] PC   (faulting instruction's return address)
 *   [sp+28] xPSR (pre-fault xPSR)
 * ============================================================ */

typedef struct {
    uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
} HardFaultStackFrame;

/* Naked: no compiler-generated prologue/epilogue, so LR still
 * holds EXC_RETURN when we inspect it, and we control exactly
 * which stack pointer (MSP vs PSP) to pass along. */
__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile (
        "TST LR, #4              \n"  /* bit 2 of EXC_RETURN: 0=MSP was used, 1=PSP */
        "ITE EQ                  \n"
        "MRSEQ R0, MSP           \n"  /* faulting frame is on MSP */
        "MRSNE R0, PSP           \n"  /* faulting frame is on PSP */
        "B HardFault_Handler_C   \n"  /* tail-call into C with frame ptr in R0 */
    );
}

void HardFault_Handler_C(HardFaultStackFrame *frame)
{
    printf("*** HARD FAULT ***\n");
    printf("  R0   = 0x%08lX\n", (unsigned long)frame->r0);
    printf("  R1   = 0x%08lX\n", (unsigned long)frame->r1);
    printf("  R2   = 0x%08lX\n", (unsigned long)frame->r2);
    printf("  R3   = 0x%08lX\n", (unsigned long)frame->r3);
    printf("  R12  = 0x%08lX\n", (unsigned long)frame->r12);
    printf("  LR   = 0x%08lX  (pre-fault return address)\n", (unsigned long)frame->lr);
    printf("  PC   = 0x%08lX  (FAULTING INSTRUCTION)\n", (unsigned long)frame->pc);
    printf("  xPSR = 0x%08lX\n", (unsigned long)frame->xpsr);

    /* R4-R11 are NOT auto-stacked -- if you need them too, you
     * must push them explicitly in the naked handler's asm before
     * branching, e.g. "PUSH {R4-R11}" then pass a second pointer. */

    while (1) { /* halt for debugger attach (Lauterbach Trace32 / J-Link) */ }
}
```

This is the exact mechanism debuggers and RTOS fault decoders (FreeRTOS `configASSERT`, Zephyr `k_fatal_halt`) rely on — **PC tells you the faulting instruction**, **LR's bit pattern (EXC_RETURN) tells you which stack/mode to unwind**, and R4-R11 require an explicit extra push since the hardware doesn't stack them automatically (a deliberate ARMv7-M design trade-off to minimize interrupt latency — fewer registers stacked = faster exception entry, consistent with the <100µs interrupt latency budgets in the Fenwick-tree sensor work).

## 5. Notes for ARMv8-A / AArch64 (Cortex-A, application-class)

```c
/* AArch64 user-space (Linux) equivalents -- register set and
 * access rules differ significantly from Cortex-M: */

static inline uint64_t get_sp_a64(void)
{
    uint64_t value;
    __asm volatile ("MOV %0, SP" : "=r"(value));   /* SP still readable via MOV */
    return value;
}

static inline uint64_t get_pc_a64(void)
{
    uint64_t value;
    __asm volatile ("ADR %0, ." : "=r"(value));    /* no MRS for PC here either */
    return value;
}

static inline uint64_t get_nzcv(void)
{
    uint64_t value;
    __asm volatile ("MRS %0, NZCV" : "=r"(value)); /* condition flags only --
                                                        full PSTATE (EL, IRQ mask,
                                                        etc.) is NOT readable from
                                                        EL0 userspace at all; that
                                                        requires EL1 (kernel) or a
                                                        debugger over JTAG/Trace32 */
    return value;
}

/* X0-X30 general purpose registers: same register-variable
 * technique applies, just with "x0".."x30" instead of "r0".."r12",
 * and LR is X30 rather than a dedicated R14. There is no combined
 * xPSR on AArch64 -- it's split into PSTATE fields (NZCV, DAIF,
 * CurrentEL, SPSel) each requiring separate MRS instructions, and
 * most are EL1+-only for security reasons (EL0 cannot read its own
 * exception level or interrupt mask state, by design). */
```

## 6. Design Summary

| Register | Access method (ARMv7-M) | Access method (AArch64 EL0) |
|---|---|---|
| R0-R12 / X0-X30 | GCC register-variable binding, `-O0` only | Same technique, `xN` names |
| SP (R13) | `MOV Rd, SP` | `MOV Xd, SP` |
| LR (R14) / X30 | `MOV Rd, LR` | `X30` is just a GPR — read directly |
| PC (R15) | `ADR Rd, .` (no MRS exists) | `ADR Xd, .` (no MRS exists) |
| xPSR / PSTATE | `MRS Rd, xPSR` (single combined read) | Split: `MRS Xd, NZCV` only at EL0; DAIF/CurrentEL need EL1+ |
| Best production practice | Hardware-auto-stacked exception frame (R0-R3,R12,LR,PC,xPSR) in fault handler | Kernel-mode `pt_regs` struct captured on exception entry, or ptrace/Trace32 for external inspection |

This mirrors the same "software vs. hardware truth source" theme as the earlier cache/CMO simulators: register-variable tricks are useful for **teaching and light debug instrumentation**, but the authoritative, non-perturbing way to read core state in production is either **hardware-driven stacking on exception entry** or **external debug access via JTAG/Trace32**, which reads the physical register file without executing any instructions on the target at all.