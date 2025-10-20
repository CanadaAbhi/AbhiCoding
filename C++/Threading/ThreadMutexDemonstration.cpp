#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void printSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << msg << "\n";
}

int main() {
    std::thread t1(printSafe, "Thread 1");
    std::thread t2(printSafe, "Thread 2");
    t1.join();
    t2.join();
}
