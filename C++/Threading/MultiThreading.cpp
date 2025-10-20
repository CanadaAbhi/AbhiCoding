#include <iostream>
#include <thread>

void sayHello() {
    std::cout << "Hello from thread!\n";
}

int main() {
    std::thread t(sayHello);
    t.join(); // Wait for thread to finish
    return 0;
}
