#include <stdint.h>

uint32_t call_secure_service(uint32_t func_id, uint32_t arg0,
                              uint32_t arg1, uint32_t arg2)
{
    register uint32_t r0 asm("r0") = func_id;
    register uint32_t r1 asm("r1") = arg0;
    register uint32_t r2 asm("r2") = arg1;
    register uint32_t r3 asm("r3") = arg2;

    asm volatile(
        "smc #0\n"
        : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)
        :
        : "memory"
    );

    return r0;   // r0 carries the return value from secure world
}
