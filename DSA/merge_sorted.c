#include <stdio.h>

void merge_sorted(const int *a, int n, const int *b, int m, int *out) {
    int i = 0, j = 0, k = 0;
    while (i < n && j < m) {
        out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    }
    while (i < n) out[k++] = a[i++];
    while (j < m) out[k++] = b[j++];
}

void merge_sorted_inplace(int *a, int n, const int *b, int m) {
    int i = n - 1, j = m - 1, k = n + m - 1;
    while (j >= 0) {
        if (i >= 0 && a[i] > b[j]) a[k--] = a[i--];
        else a[k--] = b[j--];
    }
}

int main(void) {
    int a[] = { 1, 3, 5, 7 };
    int b[] = { 2, 4, 6, 8, 10 };
    int out[9];

    merge_sorted(a, 4, b, 5, out);
    printf("Merged (new array): ");
    for (int i = 0; i < 9; i++) printf("%d ", out[i]);
    printf("\n");

    // In-place variant: a has capacity for n+m elements
    int a2[9] = { 1, 3, 5, 7, 0, 0, 0, 0, 0 }; // trailing zeros = capacity
    int b2[] = { 2, 4, 6, 8, 10 };
    merge_sorted_inplace(a2, 4, b2, 5);
    printf("Merged (in-place):  ");
    for (int i = 0; i < 9; i++) printf("%d ", a2[i]);
    printf("\n");

    return 0;
}
