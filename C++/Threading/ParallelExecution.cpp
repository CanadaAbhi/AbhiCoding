#include <iostream>
#include <thread>

void taskA() {
    std::cout << "Task A running\n";
}

void taskB() {
    std::cout << "Task B running\n";
}

int main() {
    std::thread t1(taskA);
    std::thread t2(taskB);
    t1.join();
    t2.join();
}
