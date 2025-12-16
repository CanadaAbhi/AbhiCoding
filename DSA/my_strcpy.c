#include <stdio.h>
#include <string.h>
#include <stddef.h>

char *my_strcpy(char *dest, const char *src) {
    char *d = dest;
    while ((*d++ = *src++) != '\0')
        ;
    return dest;
}

size_t my_strlcpy(char *dest, const char *src, size_t size) {
    size_t i = 0;
    for (; i + 1 < size && src[i] != '\0'; i++) dest[i] = src[i];
    if (size > 0) dest[i] = '\0';
    while (src[i]) i++;
    return i;
}

int main(void) {
    char dst[32];
    my_strcpy(dst, "Hello, embedded world!");
    printf("my_strcpy: '%s'\n", dst);

    char small[8];
    size_t needed = my_strlcpy(small, "TruncateMePlease", sizeof(small));
    printf("my_strlcpy: '%s' (truncated, full length was %zu)\n", small, needed);

    return 0;
}
