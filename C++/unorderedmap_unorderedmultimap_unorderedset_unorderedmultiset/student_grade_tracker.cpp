#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <unordered_set>
#include <string>

int main() {
    std::unordered_map<std::string, std::unordered_multiset<int>> student_grades;

    // Adding grades for students
    student_grades["Alice"].insert(90);
    student_grades["Alice"].insert(85);
    student_grades["Bob"].insert(72);
    student_grades["Charlie"].insert(88);
    student_grades["Alice"].insert(87); // Another grade for Alice

    // Displaying grades
    for (const auto& pair : student_grades) {
        std::cout << pair.first << "'s grades: ";
        for (const auto& grade : pair.second) {
            std::cout << grade << " ";
        }
        std::cout << std::endl;
    }

    // Calculating average grade for Alice
    int total = 0;
    int count = 0;
    for (const auto& grade : student_grades["Alice"]) {
        total += grade;
        count++;
    }
    std::cout << "Average grade for Alice: " << (count ? total / count : 0) << std::endl;

    return 0;
}
