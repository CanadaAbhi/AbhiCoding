#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> grades;

    // Insert student grades
    grades["Alice"] = 90;
    grades["Bob"] = 85;
    grades["Charlie"] = 88;

    // Display all grades
    std::cout << "Student Grades:" << std::endl;
    for (const auto& pair : grades) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Update Charlie's grade
    grades["Charlie"] = 93;
    std::cout << "Charlie's updated grade: " << grades["Charlie"] << std::endl;

    return 0;
}
