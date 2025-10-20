#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void safeIncrement() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        ++counter;
    }
}

int main() {
    std::thread t1(safeIncrement);
    std::thread t2(safeIncrement);
    t1.join();
    t2.join();
    std::cout << "Safe counter: " << counter << "\n";
}
