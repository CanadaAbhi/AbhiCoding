#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t reverseBits(uint32_t n) {
    uint32_t result = 0;
    for (int i = 0; i < 32; i++) {
        result <<= 1;
        result |= (n & 1);
        n >>= 1;
    }
    return result;
}

int main(void) {
    uint32_t x;
    printf("Enter a 32‑bit unsigned integer: ");
    if (scanf("%u", &x) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    uint32_t rev = reverseBits(x);
    printf("Reversed bits: %u (0x%X)\n", rev, rev);
    return EXIT_SUCCESS;
}
