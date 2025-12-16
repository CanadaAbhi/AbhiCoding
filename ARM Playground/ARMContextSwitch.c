Here's a **simplified but architecturally faithful ARM Cortex-M context switch**, built on PendSV/SVC/SysTick — the exact mechanism FreeRTOS's Cortex-M port uses under the hood, stripped down to its essential moving parts.

## Conceptual Flow

```
Task A running (PSP points into Task A's stack)
        |
   SysTick fires --> scheduler picks Task B --> pends PendSV
        |
   PendSV_Handler (lowest priority, runs after everything else drains)
        |
   Save Task A context:  stmdb psp!, {r4-r11}   (r0-r3/r12/lr/pc/xpsr already
                                                   hw-stacked on exception entry)
        |
   current_task->sp = psp   (bookmark where Task A left off)
        |
   Restore Task B context: ldmia (next_task->sp)!, {r4-r11}
        |
   msr psp, r0   +   bx 0xFFFFFFFD (EXC_RETURN: Thread mode, use PSP)
        |
Task B resumes exactly where it last yielded
```

The key insight: **you only manually save/restore R4–R11**. R0–R3, R12, LR, PC, xPSR are automatically pushed/popped by hardware on every exception entry/exit — this is the same auto-stacked frame from my HardFault work, just repurposed here as the vehicle for switching stacks instead of just diagnosing faults.

## 1. Task Control Block

```c
#include <stdint.h>

#define NUM_TASKS         2
#define STACK_SIZE_WORDS  64   /* 256 bytes per task -- deliberately small for demo */

typedef struct {
    uint32_t *sp;                    /* MUST be first member -- asm indexes offset 0 */
    uint32_t  stack[STACK_SIZE_WORDS];
    void    (*entry)(void);
    uint8_t   state;                 /* READY / RUNNING, unused in this minimal demo */
} TCB_t;

TCB_t  tcb_pool[NUM_TASKS];
TCB_t *current_task = NULL;   /* task that just ran (or NULL on first switch) */
TCB_t *next_task     = NULL;  /* task the scheduler picked */
```

`sp` being the first struct member isn't cosmetic — the assembly does `ldr r0, [r1]` on a raw `TCB_t*`, which only works because `sp` sits at offset 0.

## 2. Fabricating an Initial Stack Frame (First-Run Trick)

A task that has *never run* has no real saved context yet. So we hand-craft a fake exception frame that looks exactly like one PendSV would have produced, tricking the restore path into "resuming" a task that's actually starting for the first time.

```c
void task_exit_trap(void)
{
    /* A well-behaved task loops forever and never returns. This is the
     * safety net if one does -- mirrors the LR-as-trap pattern from my
     * HardFault handler's fault-frame analysis. */
    while (1) { __asm volatile ("nop"); }
}

void task_stack_init(TCB_t *task, void (*entry)(void))
{
    /* Full-descending stack: sp starts one-past-the-end, grows downward. */
    uint32_t *sp = &task->stack[STACK_SIZE_WORDS];
    sp -= 16;   /* reserve: 8 words (r4-r11) + 8 words (hw exception frame) */

    /* ---- Software-saved registers (restored by PendSV's ldmia) ---- */
    sp[0] = 0x04040404;  /* R4  */
    sp[1] = 0x05050505;  /* R5  */
    sp[2] = 0x06060606;  /* R6  */
    sp[3] = 0x07070707;  /* R7  */
    sp[4] = 0x08080808;  /* R8  */
    sp[5] = 0x09090909;  /* R9  */
    sp[6] = 0x10101010;  /* R10 */
    sp[7] = 0x11111111;  /* R11 */

    /* ---- Hardware-auto-stacked frame -- "returned into" on first run ---- */
    sp[8]  = 0x00000000;               /* R0  */
    sp[9]  = 0x01010101;               /* R1  */
    sp[10] = 0x02020202;               /* R2  */
    sp[11] = 0x03030303;               /* R3  */
    sp[12] = 0x12121212;               /* R12 */
    sp[13] = (uint32_t)task_exit_trap; /* LR  -- guard if task function returns */
    sp[14] = (uint32_t)entry;          /* PC  -- task's first instruction */
    sp[15] = 0x01000000;               /* xPSR -- T-bit (bit24) MUST be set;
                                           Cortex-M is Thumb-only, omitting this
                                           causes an immediate UsageFault. */

    task->sp    = sp;
    task->entry = entry;
}
```

The register values `0x04040404` etc. are deliberately obvious junk — real code doesn't care what garbage sits in R4–R11 on first entry, but making them visually distinct is invaluable when you're staring at a debugger's memory view verifying the frame layout is correct.

## 3. PendSV Handler — the Actual Context Switch

```c
__attribute__((naked)) void PendSV_Handler(void)
{
    __asm volatile (
        "mrs r0, psp                \n"  /* r0 = outgoing task's stack pointer */
        "isb                        \n"

        "ldr r1, =current_task      \n"
        "ldr r1, [r1]               \n"
        "cbz r1, 1f                 \n"  /* skip save if current_task == NULL
                                             (very first switch, nothing to save) */

        "stmdb r0!, {r4-r11}        \n"  /* push callee-saved regs onto outgoing
                                             task's own stack (r0-r3/r12/lr/pc/xpsr
                                             are already hw-stacked above this) */
        "str r0, [r1]               \n"  /* current_task->sp = updated psp */

        "1:                         \n"
        "ldr r2, =next_task         \n"
        "ldr r2, [r2]               \n"
        "ldr r0, [r2]               \n"  /* r0 = next_task->sp */
        "ldmia r0!, {r4-r11}        \n"  /* restore incoming task's callee-saved regs */
        "msr psp, r0                \n"
        "isb                        \n"

        "ldr r1, =current_task      \n"
        "str r2, [r1]               \n"  /* current_task = next_task */

        "ldr r0, =0xFFFFFFFD        \n"  /* EXC_RETURN: Thread mode, PSP, no FPU */
        "bx r0                      \n"  /* hardware pops r0-r3/r12/lr/pc/xpsr
                                             from PSP -- lands us inside Task B */
        ::: "r0", "r1", "r2"
    );
}
```

**Why `naked`?** A normal C function prologue pushes its own registers onto whatever stack is currently active *before* your code runs — corrupting the exact frame you're trying to manipulate. Same discipline as the `HardFault_Handler` naked-entry pattern.

**Why only R4–R11?** Per AAPCS, R0–R3 and R12 are caller-saved (any function can clobber them, so the hardware exception entry already preserves them for you), while R4–R11 are callee-saved — the compiler assumes they survive across calls, so *we* must explicitly preserve them since this naked function bypasses normal calling convention.

## 4. Triggering the Switch — SysTick Defers to PendSV

```c
#define SCB_ICSR        (*(volatile uint32_t *)0xE000ED04)
#define ICSR_PENDSVSET  (1U << 28)

#define SCB_SHPR3       (*(volatile uint32_t *)0xE000ED20)

void pendsv_set_lowest_priority(void)
{
    /* PendSV priority = bits [23:16] of SHPR3. Setting it to the lowest
     * possible priority (0xFF) guarantees PendSV fires ONLY after every
     * other pending interrupt has drained -- so a context switch never
     * preempts a higher-priority ISR mid-flight, preserving the
     * <100us interrupt latency discipline from my SysTick/UART work. */
    SCB_SHPR3 |= (0xFFUL << 16);
}

volatile uint8_t task_index = 0;

void scheduler_select_next(void)
{
    task_index = (task_index + 1) % NUM_TASKS;
    next_task  = &tcb_pool[task_index];
}

void SysTick_Handler(void)
{
    scheduler_select_next();

    /* Don't switch context inline here -- just request it. PendSV runs
     * at the tail of the exception queue, ensuring deterministic,
     * non-reentrant switch timing. */
    SCB_ICSR |= ICSR_PENDSVSET;
}
```

## 5. Launching the First Task — SVC Bootstraps PSP

Starting the *very first* task is a special case: there's no "outgoing" task to save, and we're not currently inside an exception that naturally exits via `bx EXC_RETURN`. The standard trick (identical to FreeRTOS's `vPortStartFirstTask`) is to trigger an `SVC` instruction, then hijack its own exception return to jump into Task A.

```c
__attribute__((naked)) void SVC_Handler(void)
{
    __asm volatile (
        "ldr r3, =current_task     \n"
        "ldr r0, [r3]               \n"  /* r0 = current_task (TCB*) */
        "ldr r0, [r0]                \n" /* r0 = current_task->sp   */
        "ldmia r0!, {r4-r11}         \n" /* load the fake initial r4-r11 */
        "msr psp, r0                 \n" /* psp now points at the hw-frame start */
        "isb                         \n"

        "mov r0, #2                  \n"
        "msr control, r0             \n" /* CONTROL.SPSEL=1: thread mode uses PSP
                                             from now on; stay privileged (bit0=0) */
        "isb                         \n"

        "orr lr, lr, #0x0D           \n" /* force EXC_RETURN bits to ...1101:
                                             return to Thread mode using PSP */
        "bx lr                       \n" /* hardware pops R0-R3/R12/LR/PC/xPSR
                                             from PSP -- we land inside Task A's
                                             entry function for the first time */
        ::: "r0", "r3"
    );
}

void start_scheduler(void)
{
    current_task = &tcb_pool[0];
    next_task    = &tcb_pool[0];
    __asm volatile ("svc #0");
    /* unreachable */
}
```

## 6. Full Wiring in `main()`

```c
void task_a_entry(void)
{
    while (1) {
        /* toggle LED A, or whatever Task A does */
        for (volatile int i = 0; i < 100000; i++) { }
    }
}

void task_b_entry(void)
{
    while (1) {
        /* toggle LED B */
        for (volatile int i = 0; i < 100000; i++) { }
    }
}

int main(void)
{
    task_stack_init(&tcb_pool[0], task_a_entry);
    task_stack_init(&tcb_pool[1], task_b_entry);

    pendsv_set_lowest_priority();
    /* systick_init(...) from my earlier SysTick framework goes here,
     * e.g. a 10ms tick for round-robin time-slicing */

    start_scheduler();   /* never returns */

    while (1) { }
}
```

## Priority Ordering (Critical, Easy to Get Wrong)

| Exception | Priority | Why |
|---|---|---|
| SVC | Higher (e.g. 0) | One-shot bootstrap; must not be delayed |
| SysTick | Higher/equal | Time-critical tick source for scheduling decisions |
| PendSV | **Lowest (0xFF)** | Must run *last*, after all other ISRs finish, so a context switch never happens in the middle of servicing a higher-priority interrupt |

Getting PendSV's priority wrong is the single most common bug when building this from scratch — if it's not strictly lowest, you can switch stacks while a nested ISR still expects the old task's stack to be there, corrupting state in ways that look exactly like the Heisenbugs from my DMA/cache-coherency work: intermittent, hard to reproduce, and only visible under specific interrupt timing.

## What This Deliberately Omits (vs. Real FreeRTOS)

- **No FPU lazy-stacking** (`CONTROL.FPCA`, `EXC_RETURN` bit 4) — real ports conditionally save `S0–S15`/`FPSCR` only if the task actually touched the FPU.
- **No priority-based scheduling** — this is pure round-robin; FreeRTOS uses a ready-list bitmap and `CLZ`-based highest-priority lookup (same `CLZ` instruction from my bit-manipulation library).
- **No stack overflow guard** — a production version would place a canary word at `stack[0]` and check it in `SysTick_Handler`, consistent with the bounded-stack discipline in my tree-traversal and FDT-parser libraries.
- **No critical section / `BASEPRI` masking** around `current_task`/`next_task` updates — safe here only because PendSV is the sole writer and runs non-reentrantly at lowest priority.

This gives you the real skeleton FreeRTOS's `port.c` builds on — once this clicks, reading `vPortSVCHandler`, `xPortPendSVHandler`, and `vTaskSwitchContext` in the actual FreeRTOS source will feel like recognizing a pattern rather than decoding unfamiliar assembly.