#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t rotateLeft(uint32_t n, unsigned int d) {
    return (n << d) | (n >> (32 - d));
}

uint32_t rotateRight(uint32_t n, unsigned int d) {
    return (n >> d) | (n << (32 - d));
}

int main(void) {
    uint32_t x;
    unsigned int d;
    printf("Enter an unsigned integer and rotate distance: ");
    if (scanf("%u %u", &x, &d) != 2) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    uint32_t l = rotateLeft(x, d);
    uint32_t r = rotateRight(x, d);
    printf("Rotate left by %u:  %u (0x%X)\n", d, l, l);
    printf("Rotate right by %u: %u (0x%X)\n", d, r, r);
    return EXIT_SUCCESS;
}
