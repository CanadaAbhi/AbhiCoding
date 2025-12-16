#include <stdio.h>
#include <string.h>
#include <stddef.h>

void *my_memmove(void *dest, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    if (d == s || n == 0) return dest;

    if (d < s) {
        for (size_t i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
    }
    return dest;
}

int main(void) {
    char buf1[] = "ABCDEFGHIJ";
    // Overlap: shift right by 2 (dest > src)
    my_memmove(buf1 + 2, buf1, 8);
    printf("Shift-right result: %s (expected ABABCDEFGH)\n", buf1);

    char buf2[] = "ABCDEFGHIJ";
    // Overlap: shift left by 2 (dest < src)
    my_memmove(buf2, buf2 + 2, 8);
    printf("Shift-left result:  %s (expected CDEFGHIJIJ)\n", buf2);

    return 0;
}
