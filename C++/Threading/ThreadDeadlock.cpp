#include <iostream>
#include <thread>
#include <mutex>

std::mutex m1, m2;

void threadA() {
    std::lock_guard<std::mutex> lock1(m1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::lock_guard<std::mutex> lock2(m2);  // waiting for m2
}

void threadB() {
    std::lock_guard<std::mutex> lock2(m2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::lock_guard<std::mutex> lock1(m1);  // waiting for m1
}

int main() {
    std::thread t1(threadA);
    std::thread t2(threadB);
    t1.join();
    t2.join();  // DEADLOCK: both threads wait forever
}
