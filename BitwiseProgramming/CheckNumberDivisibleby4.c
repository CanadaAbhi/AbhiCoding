#include <stdio.h>
#include <stdlib.h>

int isMultipleOf4(int n) {
    return (n & 3) == 0;
}

int main(void) {
    int x;
    printf("Enter an integer: ");
    if (scanf("%d", &x) != 1) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    if (isMultipleOf4(x)) {
        printf("%d is a multiple of 4.\n", x);
    } else {
        printf("%d is NOT a multiple of 4.\n", x);
    }
    return EXIT_SUCCESS;
}
