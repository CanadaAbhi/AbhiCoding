#include <iostream>
#include <unordered_set>

int main() {
    std::unordered_multiset<int> multi_set;

    // Inserting values
    multi_set.insert(1);
    multi_set.insert(2);
    multi_set.insert(2); // Duplicate countable
    multi_set.insert(3);
    multi_set.insert(1); // Another duplicate

    // Counting occurrences
    int count_value = 1;
    std::cout << "Count of " << count_value << ": " << multi_set.count(count_value) << std::endl;

    // Displaying elements
    std::cout << "Elements in multi set: ";
    for (const auto& value : multi_set) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
