#include <stdio.h>
#include <stdlib.h>

unsigned int setBit(unsigned int n, unsigned int pos) {
    return n | (1U << pos);
}

unsigned int clearBit(unsigned int n, unsigned int pos) {
    return n & ~(1U << pos);
}

unsigned int toggleBit(unsigned int n, unsigned int pos) {
    return n ^ (1U << pos);
}

unsigned int testBit(unsigned int n, unsigned int pos) { 
    return n & (1U << pos); 
}
int main(void) {
    unsigned int n;
    unsigned int pos;
    printf("Enter an unsigned integer: ");
    if (scanf("%u", &n) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    printf("Enter bit position to manipulate (0 = LSB): ");
    if (scanf("%u", &pos) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    printf("Original:      %u (0x%X)\n", n, n);
    printf("Set bit %u:    %u (0x%X)\n", pos, setBit(n, pos), setBit(n, pos));
    printf("Clear bit %u:  %u (0x%X)\n", pos, clearBit(n, pos), clearBit(n, pos));
    printf("Toggle bit %u: %u (0x%X)\n", pos, toggleBit(n, pos), toggleBit(n, pos));
    printf("Test bit %u: %u (0x%X)\n", pos, testBit(n, pos), testBit(n, pos));
    return EXIT_SUCCESS;
}
