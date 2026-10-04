#include <bits/stdc++.h>
using namespace std;

// cpp14_demo.cpp
// Compile:  g++ -std=c++14 -Wall -Wextra cpp14_demo.cpp -o cpp14_demo
#include <iostream>
#include <memory>
#include <string>

// --- 6. Variable templates ---
template<typename T>
constexpr T pi = T(3.1415926535897932385L);

// --- 5. Relaxed constexpr (loops/locals allowed) ---
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i)
        result *= i;
    return result;
}

// --- 3. Return type deduction for normal functions ---
auto square(int x) {
    return x * x;
}

// --- 4. decltype(auto) ---
int global_x = 10;
int& getRef() { return global_x; }
decltype(auto) wrapGetRef() { return getRef(); } // deduced as int&

// --- 9. [[deprecated]] attribute ---
[[deprecated("use newFunc() instead")]]
void oldFunc() { std::cout << "oldFunc() called\n"; }
void newFunc() { std::cout << "newFunc() called\n"; }

// --- 2. Lambda init-capture (move capture) ---
auto make_printer() {
    auto ptr = std::make_unique<int>(42);
    return [p = std::move(ptr)]() {
        std::cout << "make_printer captured value: " << *p << "\n";
    };
}

int main() {
    std::cout << "=== C++14 Feature Demonstration ===\n\n";

    // 1. Generic lambdas
    auto add = [](auto a, auto b) { return a + b; };
    std::cout << "[Generic lambda] add(2,3)     = " << add(2, 3) << "\n";
    std::cout << "[Generic lambda] add(2.5,1.5) = " << add(2.5, 1.5) << "\n\n";

    // 2. Lambda init-capture
    auto printer = make_printer();
    printer();
    std::cout << "\n";

    // 3. Return type deduction
    std::cout << "[Return type deduction] square(5) = " << square(5) << "\n\n";

    // 4. decltype(auto)
    decltype(auto) r = wrapGetRef();
    std::cout << "[decltype(auto)] r = " << r << "\n";
    r = 99;
    std::cout << "[decltype(auto)] global_x after modifying r = " << global_x << "\n\n";

    // 5. Relaxed constexpr
    constexpr int f5 = factorial(5);
    static_assert(f5 == 120, "factorial(5) must equal 120");
    std::cout << "[Relaxed constexpr] factorial(5) = " << f5 << "\n\n";

    // 6. Variable templates
    std::cout << "[Variable template] pi<float>  = " << pi<float> << "\n";
    std::cout << "[Variable template] pi<double> = " << pi<double> << "\n\n";

    // 7. Binary literals & digit separators
    int flags = 0b1010'1100;
    long big  = 1'000'000'000L;
    std::cout << "[Binary literal]   flags = " << flags << "\n";
    std::cout << "[Digit separator]  big   = " << big << "\n\n";

    // 8. std::make_unique
    auto up = std::make_unique<int>(99);
    std::cout << "[make_unique] *up = " << *up << "\n\n";

    // 9. [[deprecated]] (compiler emits a warning here, program still runs)
    oldFunc();
    newFunc();

    std::cout << "\n=== End of C++14 Demonstration ===\n";
    return 0;
}

