#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>  // std::greater

int main() {
    std::vector<int> nums = {1, 3, 5, 2, 4};

    // Sort in descending order using std::greater
    std::sort(nums.begin(), nums.end(), std::greater<int>());

    for (auto n : nums) std::cout << n << " ";
    // Output: 5 4 3 2 1
}
