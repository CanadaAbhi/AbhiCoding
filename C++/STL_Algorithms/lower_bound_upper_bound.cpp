#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 3, 3, 5, 7};
    auto lb = std::lower_bound(v.begin(), v.end(), 3);
    auto ub = std::upper_bound(v.begin(), v.end(), 3);
    std::cout << "Lower bound of 3 at index: " << (lb - v.begin()) << std::endl;
    std::cout << "Upper bound of 3 at index: " << (ub - v.begin()) << std::endl;
}
