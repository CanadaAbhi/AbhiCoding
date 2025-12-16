#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

void *my_memcpy(void *dest, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;

    if (((uintptr_t)d % sizeof(long) == 0) &&
        ((uintptr_t)s % sizeof(long) == 0) &&
        n >= sizeof(long)) {
        long *dl = (long *)d;
        const long *sl = (const long *)s;
        size_t words = n / sizeof(long);
        for (size_t i = 0; i < words; i++) dl[i] = sl[i];
        d += words * sizeof(long);
        s += words * sizeof(long);
        n -= words * sizeof(long);
    }
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dest;
}

int main(void) {
    char src[64];
    char dst[64];
    for (int i = 0; i < 64; i++) src[i] = (char)('A' + (i % 26));

    memset(dst, 0, sizeof(dst));
    my_memcpy(dst, src, sizeof(src));

    int ok = (memcmp(src, dst, sizeof(src)) == 0);
    printf("my_memcpy full-buffer test: %s\n", ok ? "PASS" : "FAIL");

    // Unaligned / odd-size test
    char dst2[10] = {0};
    my_memcpy(dst2, "HELLOWRLD", 9);
    dst2[9] = '\0';
    printf("my_memcpy small test: '%s' (expected HELLOWRLD)\n", dst2);

    return 0;
}
