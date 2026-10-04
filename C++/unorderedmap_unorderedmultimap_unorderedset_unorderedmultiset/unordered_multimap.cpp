#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

int main() {
    std::unordered_multimap<std::string, int> multi_map;

    // Inserting key-multiple value pairs
    multi_map.emplace("Fruit", 1);
    multi_map.emplace("Fruit", 2);
    multi_map.emplace("Vegetable", 1);
    multi_map.emplace("Fruit", 3);

    // Retrieving and displaying values
    std::string search_key = "Fruit";
    auto range = multi_map.equal_range(search_key);
    std::cout << "Values for '" << search_key << "': ";
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << it->second << " ";
    }
    std::cout << std::endl;

    return 0;
}
