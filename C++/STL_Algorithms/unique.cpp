#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 2, 3, 3, 3};
    auto it = std::unique(v.begin(), v.end());
    v.erase(it, v.end());
    for(int n : v) std::cout << n << " ";
}
