#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void *my_memcpy(void *dest, const void *src, size_t n) {
    char *d = (char*)dest;
    const char *s = (const char*)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

int main() {
    const char source[] = "Hello, World!";
    char destination[20];

    my_memcpy(destination, source, strlen(source) + 1); // +1 to include null terminator

    printf("Source: %s\n", source);
    printf("Destination: %s\n", destination);

    return 0;
}