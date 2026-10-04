#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx; // Mutex for synchronizing access
int counter = 0;

void increment_counter() {
    for (int i = 0; i < 1000; ++i) {
        std::lock_guard<std::mutex> lock(mtx);  // Locks the mutex
        ++counter;  // Critical section
    }
}

int main() {
    std::vector<std::thread> threads;

    // Create 10 threads to increment the counter
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(increment_counter);
    }

    // Join all threads
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final counter value: " << counter << std::endl; // Should be 10000
    return 0;
}
