#include <iostream>
using namespace std;

int divide(int a, int b) {
    if (b == 0)
        throw "Division by zero!";  // Throwing a string literal
    return a / b;
}

int main() {
    try {
        // risky code
    } catch (int e) {
        // handle integer exceptions
    } catch (const std::exception& e) {
        cout << "Standard exception: " << e.what() << endl;
    } catch (...) {
        // Catch-all handler
        cout << "Unknown exception caught!" << endl;
    }
}


