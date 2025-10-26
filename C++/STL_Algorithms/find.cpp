#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {5, 10, 15, 20};
    auto it = std::find(v.begin(), v.end(), 15);
    if(it != v.end()) std::cout << "Found " << *it << std::endl;
}
