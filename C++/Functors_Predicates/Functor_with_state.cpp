#include <iostream>

class Add {
    int value;
public:
    Add(int v) : value(v) {}
    int operator()(int x) const { return x + value; }
};

int main() {
    Add add_five(5);
    std::cout << add_five(10) << std::endl; // Output: 15
    return 0;
}
