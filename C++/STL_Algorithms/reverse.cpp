#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::reverse(v.begin(), v.end());
    for (int n : v) std::cout << n << " ";
}
