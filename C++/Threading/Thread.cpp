#include <iostream>
#include <thread>

void printNumbers() {
    for (int i = 0; i < 5; ++i)
        std::cout << "Thread: " << i << "\n";
}

int main() {
    std::thread t(printNumbers);
    for (int i = 0; i < 5; ++i)
        std::cout << "Main: " << i << "\n";
    t.join();
    return 0;
}
