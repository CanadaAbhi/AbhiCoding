.section .text.monitor
.global monitor_smc_entry

monitor_smc_entry:
    push {r4-r12, lr}        // save Non-secure context

    cmp   r0, #0x1
    beq   handle_service_1

    mov   r0, #0xFFFFFFFF    // unknown SMC function ID
    b     monitor_exit

handle_service_1:
    bl    secure_service_1   // dispatch to actual secure handler
    b     monitor_exit

monitor_exit:
    pop   {r4-r12, lr}
    movs  pc, lr             // return to NS caller, CPSR restored from SPSR
