#include <iostream>
#include <future>

int computeSquare(int x) {
    return x * x;
}

int main() {
    std::future<int> result = std::async(std::launch::async, computeSquare, 10);

    std::cout << "Doing other work...\n";

    int value = result.get();  // Waits until computation is done
    std::cout << "Square is: " << value << "\n";
}
