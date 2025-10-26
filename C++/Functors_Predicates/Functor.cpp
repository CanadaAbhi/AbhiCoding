#include <iostream>

class Greet {
public:
    void operator()() const {
        std::cout << "Hello, Functor!" << std::endl;
    }
};

int main() {
    Greet greet;
    greet(); // Calls operator(), prints greeting
    return 0;
}
