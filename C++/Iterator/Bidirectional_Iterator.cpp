#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {1, 2, 3};

    auto it = lst.begin();
    std::cout << *it << std::endl;  // 1
    ++it;
    std::cout << *it << std::endl;  // 2
    --it;
    std::cout << *it << std::endl;  // 1
}
