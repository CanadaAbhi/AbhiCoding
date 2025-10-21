#include <stdio.h>
#include <stdbool.h>
bool isEven(int n) { return (n & 1) == 0; }


int main() {
    int number;
    printf( "Enter an integer number");
    scanf("%d \n",&number);

    if (isEven(number))
       printf( " is an even number.\n");
       else
       printf( " is an odd number \n");

    return 0;
}