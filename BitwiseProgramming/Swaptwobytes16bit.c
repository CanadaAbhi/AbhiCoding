#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint16_t swapBytes16(uint16_t x) {
    return (uint16_t)(((x & 0x00FFU) << 8) | ((x & 0xFF00U) >> 8));
}

int main(void) {
    uint16_t x;
    printf("Enter a 16‑bit unsigned integer (0‑65535): ");
    unsigned int t;
    if (scanf("%u", &t) != 1 || t > 0xFFFFU) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    x = (uint16_t)t;
    uint16_t swapped = swapBytes16(x);
    printf("Original: 0x%04X  Swapped bytes: 0x%04X\n", x, swapped);
    return EXIT_SUCCESS;
}
