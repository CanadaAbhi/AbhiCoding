#include <iostream>
#include <unordered_set>

int main() {
    std::unordered_set<int> my_set;

    // Inserting values
    my_set.insert(1);
    my_set.insert(2);
    my_set.insert(3);
    my_set.insert(3); // Duplicate will not be added

    // Checking size
    std::cout << "Set size: " << my_set.size() << std::endl;

    // Searching for an element
    int search_value = 2;
    if (my_set.find(search_value) != my_set.end()) {
        std::cout << search_value << " found in the set." << std::endl;
    } else {
        std::cout << search_value << " not found in the set." << std::endl;
    }

    // Displaying elements
    std::cout << "Elements in set: ";
    for (const auto& value : my_set) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
