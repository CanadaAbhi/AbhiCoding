#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {10, 20, 30, 40};

    // Using iterator to traverse
    for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";  // dereference iterator to get element
    }
    std::cout << std::endl;
}
