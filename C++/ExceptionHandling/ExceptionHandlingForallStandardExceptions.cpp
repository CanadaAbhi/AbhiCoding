#include <iostream>
#include <vector>
#include <stdexcept>  // Required for standard exceptions
using namespace std;

void throwStdException() {
    throw std::exception();  // Base class
}

void throwRuntimeError() {
    throw std::runtime_error("Runtime error occurred");
}

void throwLogicError() {
    throw std::logic_error("Logic error occurred");
}

void throwOutOfRange() {
    vector<int> v = {1, 2, 3};
    cout << v.at(5);  // Throws std::out_of_range
}

void throwInvalidArgument() {
    throw std::invalid_argument("Invalid argument passed");
}

void throwLengthError() {
    string s;
    s.reserve(s.max_size() + 1);  // Throws std::length_error
}

int main() {
    try {
        throwStdException();
    } catch (const std::exception& e) {
        cout << "[std::exception] " << e.what() << endl;
    }

    try {
        throwRuntimeError();
    } catch (const std::runtime_error& e) {
        cout << "[std::runtime_error] " << e.what() << endl;
    }

    try {
        throwLogicError();
    } catch (const std::logic_error& e) {
        cout << "[std::logic_error] " << e.what() << endl;
    }

    try {
        throwOutOfRange();
    } catch (const std::out_of_range& e) {
        cout << "[std::out_of_range] " << e.what() << endl;
    }

    try {
        throwInvalidArgument();
    } catch (const std::invalid_argument& e) {
        cout << "[std::invalid_argument] " << e.what() << endl;
    }

    try {
        throwLengthError();
    } catch (const std::length_error& e) {
        cout << "[std::length_error] " << e.what() << endl;
    }

    return 0;
}
