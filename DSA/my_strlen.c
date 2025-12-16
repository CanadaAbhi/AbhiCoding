#include <stdio.h>
#include <stddef.h>
#include <string.h>

size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}

int main(void) {
    const char *tests[] = { "", "a", "hello world", "1234567890" };
    int all_pass = 1;

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        size_t got = my_strlen(tests[i]);
        size_t expected = strlen(tests[i]);
        printf("strlen(\"%s\") = %zu (expected %zu) %s\n",
               tests[i], got, expected, got == expected ? "PASS" : "FAIL");
        if (got != expected) all_pass = 0;
    }

    printf("Overall: %s\n", all_pass ? "PASS" : "FAIL");
    return 0;
}
