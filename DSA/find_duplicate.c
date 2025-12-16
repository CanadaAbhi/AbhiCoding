#include <stdio.h>

int find_duplicate(int *nums, int n) {
    int slow = nums[0], fast = nums[0];
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);

    slow = nums[0];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}

int main(void) {
    int nums1[] = { 1, 3, 4, 2, 2 }; // n=4, array size n+1=5
    int nums2[] = { 3, 1, 3, 4, 2 };

    printf("Duplicate in [1,3,4,2,2] = %d (expected 2)\n",
           find_duplicate(nums1, 4));
    printf("Duplicate in [3,1,3,4,2] = %d (expected 3)\n",
           find_duplicate(nums2, 4));

    return 0;
}
