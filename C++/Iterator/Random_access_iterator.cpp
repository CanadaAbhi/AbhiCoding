#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {5, 10, 15, 20, 25};

    auto it = vec.begin();
    std::cout << *(it + 3) << std::endl;  // Move iterator 3 positions: 20

    it += 2;
    std::cout << *it << std::endl;        // 15
}
