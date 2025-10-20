#include <iostream>
#include <thread>

void work() {
    std::cout << "Working...\n";
}

int main() {
    std::thread t1(work);
    std::thread t2(work);

    t1.join();   // Main waits
    //t2.detach(); // Runs independently  (becomes daemon)
    //Detached threads must not access resources owned by main(), or risk undefined behavior.
    std::cout << "Main thread done.\n";
}
