#include <stdio.h>
#include <stdlib.h>

int oppositeSigns(int x, int y) {
    return ((x ^ y) < 0);
}

int main(void) {
    int a, b;
    printf("Enter two integers: ");
    if (scanf("%d %d", &a, &b) != 2) {
        printf("Invalid input\n");
        return EXIT_FAILURE;
    }
    if (oppositeSigns(a, b)) {
        printf("%d and %d have opposite signs.\n", a, b);
    } else {
        printf("%d and %d have the same sign (or one is zero).\n", a, b);
    }
    return EXIT_SUCCESS;
}
