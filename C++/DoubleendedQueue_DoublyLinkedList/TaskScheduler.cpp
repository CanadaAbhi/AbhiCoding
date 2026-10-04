#include <iostream>
#include <deque>
#include <string>

int main() {
    std::deque<std::string> tasks;

    // Adding tasks to the front and back
    tasks.push_back("Task 1");
    tasks.push_back("Task 2");
    tasks.push_front("Urgent Task");

    std::cout << "Current tasks: ";
    for (const auto& task : tasks) {
        std::cout << task << " ";
    }
    std::cout << std::endl;

    // Completing the first task
    std::cout << "Completing: " << tasks.front() << std::endl;
    tasks.pop_front();

    std::cout << "Tasks after completing one: ";
    for (const auto& task : tasks) {
        std::cout << task << " ";
    }
    std::cout << std::endl;

    return 0;
}
