#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> copy_v(v.size());
    std::copy(v.begin(), v.end(), copy_v.begin());
    for(int n : copy_v) std::cout << n << " ";
}
