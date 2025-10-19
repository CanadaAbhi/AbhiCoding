#include <iostream>
#include <vector>
#include <iterator>

int main() {
    std::vector<int> data = {10, 20, 30, 40, 50};
    auto it = data.begin();

    std::advance(it, 3);   // Move it forward by 3 elements (points to 40)
    std::cout << *it << "\n";

    auto it2 = std::next(it);  // Returns iterator to next element (50)
    std::cout << *it2 << "\n";

    auto it3 = std::prev(it, 2); // Move back 2 to 20
    std::cout << *it3 << "\n";

    std::cout << "Distance: " << std::distance(data.begin(), it2) << std::endl;
}
