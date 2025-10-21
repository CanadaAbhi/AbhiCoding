#include <stdio.h>

int add(int a, int b) {
    while (b != 0) {
        int carry = a & b;
        a = a ^ b;
        b = carry << 1;
    }
    return a;
}

int main() {
    int num1, num2;
    printf("Enter two integers to add: ");
    if (scanf("%d %d", &num1, &num2) != 2) {
        printf("Invalid input\n");
        return 1;
    }
    int sum = add(num1, num2);
    printf("Sum: %d\n", sum);
    return 0;
}