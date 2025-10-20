#include <iostream>
#include <thread>
#include <mutex>

std::mutex m1, m2;

void safeThread() {
    std::lock(m1, m2);  // Locks both mutexes atomically
    std::lock_guard<std::mutex> lock1(m1, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(m2, std::adopt_lock);
    std::cout << "Safe from deadlock\n";
}

int main() {
    std::thread t1(safeThread);
    std::thread t2(safeThread);
    t1.join();
    t2.join();
}
