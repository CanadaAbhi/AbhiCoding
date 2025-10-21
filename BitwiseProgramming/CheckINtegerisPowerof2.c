#include <stdio.h>
#include <stdlib.h>

int isPowerOfTwo(unsigned int n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}

int main(void) {
    unsigned int x;
    printf("Enter a positive integer: ");
    if (scanf("%u", &x) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    if (isPowerOfTwo(x)) {
        printf("%u is a power of 2.\n", x);
    } else {
        printf("%u is NOT a power of 2.\n", x);
    }
    return EXIT_SUCCESS;
}
