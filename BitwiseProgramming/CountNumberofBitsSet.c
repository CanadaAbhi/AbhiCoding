#include <stdio.h>
#include <stdlib.h>

unsigned int countSetBits(unsigned int n) {
    unsigned int count = 0;
    while (n) {
        n &= (n - 1);  // drop the lowest set bit
        ++count;
    }
    return count;
}

int main(void) {
    unsigned int x;
    printf("Enter a non‑negative integer: ");
    if (scanf("%u", &x) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    printf("Number of set bits in %u is %u.\n", x, countSetBits(x));
    return EXIT_SUCCESS;
}
