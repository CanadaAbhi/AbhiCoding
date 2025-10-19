#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> res(v.size());
    std::transform(v.begin(), v.end(), res.begin(), [](int x){ return x * x; }); // square

    for(int n : res) std::cout << n << " ";
}
