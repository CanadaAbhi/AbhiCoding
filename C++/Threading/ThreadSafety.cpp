#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> safeCounter{0};

void increment() {
    for (int i = 0; i < 100000; ++i)
        ++safeCounter;
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();
    std::cout << "Thread-safe counter: " << safeCounter << "\n";
}
