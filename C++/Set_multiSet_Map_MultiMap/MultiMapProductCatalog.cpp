#include <iostream>
#include <map>
#include <string>

int main() {
    std::multimap<std::string, double> products;

    // Insert products with prices
    products.insert({"Laptop", 1200.99});
    products.insert({"Laptop", 1150.49});
    products.insert({"Smartphone", 800.00});
    products.insert({"Tablet", 400.00});

    std::cout << "Product Catalog:" << std::endl;

    // Display all products and their prices
    for (const auto& pair : products) {
        std::cout << pair.first << ": $" << pair.second << std::endl;
    }

    // Find all prices for "Laptop"
    std::cout << "Prices for Laptop:" << std::endl;
    auto range = products.equal_range("Laptop");
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << "$" << it->second << std::endl;
    }

    return 0;
}
