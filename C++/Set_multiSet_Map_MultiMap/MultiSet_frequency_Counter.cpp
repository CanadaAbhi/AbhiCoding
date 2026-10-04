#include <iostream>
#include <set>

int main() {
    std::multiset<int> numbers = {1, 2, 2, 3, 4, 4, 4, 5};

    std::cout << "Number Frequencies:" << std::endl;
    // Count occurrences
    for (const auto& num : numbers) {
        int count = numbers.count(num);
        std::cout << num << ": " << count << " time(s)" << std::endl;
        // Erase duplicates from multiset
        numbers.erase(num);
    }

    return 0;
}
