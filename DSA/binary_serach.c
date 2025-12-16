#include <stdio.h>

int binary_search(const int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int main(void) {
    int arr[] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 };
    int n = sizeof(arr) / sizeof(arr[0]);

    int targets[] = { 1, 19, 7, 8, -5 };
    for (size_t i = 0; i < sizeof(targets) / sizeof(targets[0]); i++) {
        int idx = binary_search(arr, n, targets[i]);
        printf("search(%d) -> index %d\n", targets[i], idx);
    }

    return 0;
}
