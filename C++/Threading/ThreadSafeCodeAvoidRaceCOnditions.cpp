#include <atomic>
#include <thread>
#include <iostream>

std::atomic<int> safeCounter{0};

void atomicIncrement() {
    for (int i = 0; i < 100000; ++i)
        ++safeCounter;
}

int main() {
    std::thread t1(atomicIncrement);
    std::thread t2(atomicIncrement);
    t1.join();
    t2.join();
    std::cout << "Thread-safe counter: " << safeCounter << "\n";
}
