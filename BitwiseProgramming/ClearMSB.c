#include <stdio.h>
#include <stdlib.h>

unsigned int clearRightmostSetBit(unsigned int n) {
    return n & (n - 1);
}

int main(void) {
    unsigned int x;
    printf("Enter a non‑negative integer: ");
    if (scanf("%u", &x) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    unsigned int result = clearRightmostSetBit(x);
    printf("Original:      %u (0x%X)\n", x, x);
    printf("After clearing rightmost set bit: %u (0x%X)\n", result, result);
    return EXIT_SUCCESS;
}
