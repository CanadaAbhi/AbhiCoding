#include <stdio.h>
#include <stdlib.h>

void swapBits(int *a, int *b) {
    if (a == b) return;  // if they point to same location, do nothing
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

int main(void) {
    int x, y;
    printf("Enter two integers (x y): ");
    if (scanf("%d %d", &x, &y) != 2) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    printf("Before swap: x = %d, y = %d\n", x, y);
    swapBits(&x, &y);
    printf("After  swap: x = %d, y = %d\n", x, y);
    return EXIT_SUCCESS;
}
