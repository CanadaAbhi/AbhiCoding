Building directly on the PendSV/SVC context-switch mechanics from the previous project, here's a **tiny but complete 3-task round-robin scheduler** with a clean separation of concerns: `scheduler()` makes the *decision*, `task_switch()` *requests* it, and `PendSV_Handler` *executes* it. This mirrors exactly how FreeRTOS separates `vTaskSwitchContext()` (decision) from `xPortPendSVHandler()` (mechanism).

## Architecture — Decision vs. Mechanism Separation

```
SysTick_Handler (ISR, high priority)
        |
        v
   scheduler()  <-- pure decision logic: "who runs next?"
        |             (no stack manipulation here at all)
        v
   task_switch() <-- just sets PENDSVSET bit, returns immediately
        |
        v
PendSV_Handler (naked, lowest priority, deferred execution)
        |
        v
  Context Switch (asm: save R4-R11, restore R4-R11, swap PSP)
```

The critical design principle: **SysTick never touches the stack pointer directly**. It only flags *intent*. The actual register save/restore happens later, in PendSV, at the lowest possible priority — guaranteeing it never preempts a higher-priority ISR mid-flight, the same latency discipline from my SysTick/UART/GPIO interrupt work.

## 1. Task Control Block & Static Pool (Zero Heap)

```c
#include <stdint.h>
#include <stddef.h>

#define MAX_TASKS         3
#define STACK_SIZE_WORDS  64      /* 256B/task -- deliberately small for demo */

typedef enum {
    TASK_UNUSED = 0,
    TASK_READY,
    TASK_RUNNING,
    TASK_BLOCKED
} task_state_t;

typedef struct {
    uint32_t     *sp;                     /* offset 0 -- asm indexes this raw */
    uint32_t      stack[STACK_SIZE_WORDS];
    void        (*entry)(void);
    task_state_t  state;
    uint8_t       id;
} TCB_t;

static TCB_t  task_pool[MAX_TASKS];
static uint8_t task_count = 0;

/* Scheduler bookkeeping -- read by PendSV asm, written only by scheduler() */
TCB_t *current_task = NULL;
TCB_t *next_task     = NULL;

static uint8_t current_index = 0;
```

## 2. `task_create()` — Static Allocation + Fabricated Initial Frame

```c
static void task_exit_trap(void)
{
    /* Safety net: a well-behaved task loops forever. If one returns
     * anyway, trap here instead of running off into garbage memory --
     * same LR-as-guard pattern used in my HardFault handler work. */
    while (1) { __asm volatile ("nop"); }
}

/* Returns 0 on success, -1 if the static pool is exhausted -- no malloc,
 * no dynamic growth, deterministic memory footprint by design. */
int task_create(void (*entry)(void))
{
    if (task_count >= MAX_TASKS || entry == NULL) {
        return -1;
    }

    TCB_t *t = &task_pool[task_count];

    /* Full-descending stack, grows downward from top of the array. */
    uint32_t *sp = &t->stack[STACK_SIZE_WORDS];
    sp -= 16;   /* 8 words software-saved (r4-r11) + 8 words hw-stacked frame */

    sp[0] = 0x04040404;  /* R4  -- distinct junk values ease debugger inspection */
    sp[1] = 0x05050505;  /* R5  */
    sp[2] = 0x06060606;  /* R6  */
    sp[3] = 0x07070707;  /* R7  */
    sp[4] = 0x08080808;  /* R8  */
    sp[5] = 0x09090909;  /* R9  */
    sp[6] = 0x10101010;  /* R10 */
    sp[7] = 0x11111111;  /* R11 */

    sp[8]  = 0x00000000;                  /* R0  */
    sp[9]  = 0x01010101;                  /* R1  */
    sp[10] = 0x02020202;                  /* R2  */
    sp[11] = 0x03030303;                  /* R3  */
    sp[12] = 0x12121212;                  /* R12 */
    sp[13] = (uint32_t)task_exit_trap;    /* LR  -- guard if task returns */
    sp[14] = (uint32_t)entry;             /* PC  -- first instruction */
    sp[15] = 0x01000000;                  /* xPSR -- T-bit set; Thumb-only,
                                              omitting this = instant UsageFault */

    t->sp    = sp;
    t->entry = entry;
    t->state = TASK_READY;
    t->id    = task_count;

    task_count++;
    return 0;
}
```

## 3. `scheduler()` — Pure Decision Logic (No Stack Manipulation)

```c
/* Called from SysTick context. Decides WHO runs next, but does NOT touch
 * PSP, R4-R11, or any CPU register. That separation is deliberate: keeping
 * scheduling policy (round-robin here, priority-based in FreeRTOS) fully
 * decoupled from switching mechanism (PendSV asm) is what lets you swap
 * the policy later without touching the fragile context-switch code. */
void scheduler(void)
{
    if (task_count == 0) {
        return;
    }

    if (current_task != NULL && current_task->state == TASK_RUNNING) {
        current_task->state = TASK_READY;
    }

    /* Round-robin: advance to next task, skipping anything not READY
     * (e.g. BLOCKED tasks waiting on I/O in a fuller implementation). */
    uint8_t start = current_index;
    do {
        current_index = (current_index + 1) % task_count;
    } while (task_pool[current_index].state != TASK_READY &&
             current_index != start);

    next_task = &task_pool[current_index];
    next_task->state = TASK_RUNNING;

    if (next_task != current_task) {
        task_switch();   /* only pend PendSV if an actual switch is needed --
                             avoids gratuitous exception overhead every tick */
    }
}
```

## 4. `task_switch()` — Request, Don't Execute

```c
#define SCB_ICSR        (*(volatile uint32_t *)0xE000ED04)
#define ICSR_PENDSVSET  (1U << 28)

/* This function does exactly one thing: flag that a context switch is
 * needed. It does NOT save or restore any register -- that happens later,
 * asynchronously, in PendSV_Handler. This indirection exists because
 * SysTick_Handler runs at a HIGHER priority than PendSV; if we tried to
 * switch stacks directly inside SysTick, we'd risk corrupting state for
 * any interrupt that preempts SysTick itself. Deferring to the
 * lowest-priority exception is what makes this safe. */
void task_switch(void)
{
    SCB_ICSR |= ICSR_PENDSVSET;
}
```

## 5. `PendSV_Handler` — The Actual Context Switch (Mechanism)

```c
__attribute__((naked)) void PendSV_Handler(void)
{
    __asm volatile (
        "mrs r0, psp                \n"  /* r0 = outgoing task's PSP */
        "isb                        \n"

        "ldr r1, =current_task      \n"
        "ldr r1, [r1]               \n"
        "cbz r1, 1f                 \n"  /* NULL on very first switch -- nothing
                                             to save yet */

        "stmdb r0!, {r4-r11}        \n"  /* save callee-saved regs onto the
                                             OUTGOING task's own stack */
        "str r0, [r1]               \n"  /* current_task->sp = updated psp */

        "1:                         \n"
        "ldr r2, =next_task         \n"
        "ldr r2, [r2]               \n"
        "ldr r0, [r2]               \n"  /* r0 = next_task->sp */
        "ldmia r0!, {r4-r11}        \n"  /* restore INCOMING task's regs */
        "msr psp, r0                \n"
        "isb                        \n"

        "ldr r1, =current_task      \n"
        "str r2, [r1]               \n"  /* current_task = next_task */

        "ldr r0, =0xFFFFFFFD        \n"  /* EXC_RETURN: Thread mode, PSP */
        "bx r0                      \n"  /* hw pops r0-r3/r12/lr/pc/xpsr --
                                             lands inside the new task */
        ::: "r0", "r1", "r2"
    );
}
```

This is identical in structure to the standalone context-switch project — which is the point. Here it's wired to a real scheduling *policy* instead of a hardcoded A→B toggle.

## 6. `SysTick_Handler` — The Trigger

```c
volatile uint32_t tick_count = 0;

void SysTick_Handler(void)
{
    tick_count++;
    scheduler();   /* SysTick -> Scheduler, per the requested pipeline */
}
```

## 7. Priority Configuration — Why the Pipeline Order Is Non-Negotiable

```c
#define SCB_SHPR3  (*(volatile uint32_t *)0xE000ED20)
#define SCB_SHPR1  (*(volatile uint32_t *)0xE000ED18)

void scheduler_priorities_init(void)
{
    /* PendSV = lowest priority (bits [23:16] of SHPR3) -- must run dead
     * last, after every other pending exception has fully drained. This
     * is what guarantees a context switch never happens in the middle of
     * servicing a higher-priority ISR. */
    SCB_SHPR3 |= (0xFFUL << 16);

    /* SysTick = higher priority than PendSV (bits [31:24] of SHPR3) --
     * it must be able to preempt PendSV to keep tick timing deterministic,
     * even mid-switch. */
    SCB_SHPR3 |= (0x00UL << 24);
}
```

| Exception | Priority | Role in the pipeline |
|---|---|---|
| SysTick | High (0x00) | Fires the scheduling *decision* on a fixed cadence |
| PendSV | **Lowest (0xFF)** | Executes the *mechanism* only after everything else settles |
| SVC | High | One-shot bootstrap of the very first task |

## 8. Bootstrap — Starting the First Task (SVC, as before)

```c
__attribute__((naked)) void SVC_Handler(void)
{
    __asm volatile (
        "ldr r3, =current_task     \n"
        "ldr r0, [r3]               \n"
        "ldr r0, [r0]                \n" /* r0 = current_task->sp */
        "ldmia r0!, {r4-r11}         \n"
        "msr psp, r0                 \n"
        "isb                         \n"

        "mov r0, #2                  \n"
        "msr control, r0             \n" /* switch Thread mode to use PSP */
        "isb                         \n"

        "orr lr, lr, #0x0D           \n" /* EXC_RETURN -> Thread/PSP */
        "bx lr                       \n"
        ::: "r0", "r3"
    );
}

void scheduler_start(void)
{
    if (task_count == 0) return;

    current_task = &task_pool[0];
    current_task->state = TASK_RUNNING;
    next_task = current_task;
    current_index = 0;

    __asm volatile ("svc #0");
    /* unreachable */
}
```

## 9. Wiring Three Tasks Together

```c
volatile uint32_t task1_runs = 0, task2_runs = 0, task3_runs = 0;

void task1_entry(void)
{
    while (1) {
        task1_runs++;
        for (volatile int i = 0; i < 50000; i++) { }
    }
}

void task2_entry(void)
{
    while (1) {
        task2_runs++;
        for (volatile int i = 0; i < 50000; i++) { }
    }
}

void task3_entry(void)
{
    while (1) {
        task3_runs++;
        for (volatile int i = 0; i < 50000; i++) { }
    }
}

int main(void)
{
    task_create(task1_entry);   /* Task 1 -> task_pool[0] */
    task_create(task2_entry);   /* Task 2 -> task_pool[1] */
    task_create(task3_entry);   /* Task 3 -> task_pool[2] */

    scheduler_priorities_init();
    /* systick_init(...) from earlier SysTick framework -- e.g. 10ms tick,
     * giving each task a 10ms round-robin time slice */

    scheduler_start();          /* never returns */

    while (1) { }
}
```

## Execution Trace (What Actually Happens Every 10ms)

```
t=0ms   Task1 running -------------------------------------\
t=10ms  SysTick fires --> scheduler() picks Task2           |
        Task2 != Task1 --> task_switch() pends PendSV       | SysTick priority:
        SysTick_Handler returns                             | HIGH (preempts PendSV)
                                                              |
        PendSV_Handler runs (deferred, lowest priority) -----/
        saves Task1's R4-R11, restores Task2's R4-R11, swaps PSP
t=10ms+ Task2 resumes exactly where it last yielded
t=20ms  SysTick fires --> scheduler() picks Task3 --> repeat...
t=30ms  SysTick fires --> scheduler() picks Task1 --> repeat...
```

## Why This Separation Matters (Interview-Relevant Insight)

The naive approach — doing the register save/restore *inside* `SysTick_Handler` directly — technically works for a toy demo, but it's architecturally wrong for two reasons:

1. **Priority inversion risk**: SysTick typically needs to run at a fairly high priority for timing accuracy. If it also does the heavy lifting of stack swapping, that whole operation runs at high priority too, potentially blocking other important interrupts for longer than necessary.
2. **Interruptibility**: PendSV's defining property is that it's *interruptible-then-resumable* by higher-priority exceptions before it actually executes, because pending it just sets a bit — the CPU decides when to actually service it. A context switch physically embedded in SysTick can't be deferred that way.

This decision/mechanism split — `scheduler()` decides, `task_switch()` requests, `PendSV_Handler` executes — is exactly the abstraction boundary FreeRTOS enforces between `tasks.c` (policy) and `port.c` (mechanism), and understanding *why* that boundary exists is usually the differentiator in an RTOS internals interview versus just being able to recite the PendSV trick.

## Natural Next Extensions

- **Priority-based scheduling**: replace round-robin `current_index++` with a ready-bitmap + `CLZ` lookup (same instruction from my bit-manipulation library) for O(1) highest-priority selection.
- **Blocking primitives**: add `task_yield()`/`task_sleep(ticks)` that set `state = TASK_BLOCKED` and call `task_switch()` immediately rather than waiting for the next tick.
- **Stack overflow canaries**: write a sentinel to `stack[0]` in `task_create()` and check it in `scheduler()`, consistent with the bounded-memory discipline in my FDT parser and tree-traversal libraries.
- **Idle task**: a 4th always-`READY` task that just executes `WFI`, so the CPU sleeps instead of spinning when no real work is pending — ties directly into the performance/watt work from my DVFS/PWM projects.