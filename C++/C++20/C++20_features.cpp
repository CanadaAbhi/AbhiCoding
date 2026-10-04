// cpp20_demo.cpp
// Compile (GCC 13+/Clang 16+): g++ -std=c++20 -Wall -Wextra cpp20_demo.cpp -o cpp20_demo
// std::format requires GCC 13+ / Clang 17+ / MSVC 19.29+.
#include <iostream>
#include <concepts>
#include <ranges>
#include <vector>
#include <coroutine>
#include <compare>
#include <span>
#include <format>
#include <type_traits>

// --- 1. Concepts ---
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }

// --- 5. Three-way comparison (spaceship operator) ---
struct Point {
    int x, y;
    auto operator<=>(const Point&) const = default;
};

// --- 6. consteval / constinit ---
consteval int ct_square(int x) { return x * x; }
constinit int global_counter = 42;

// --- 3. Coroutines ---
struct Generator {
    struct promise_type {
        int value;
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(int v) { value = v; return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
        Generator get_return_object() {
            return Generator{ std::coroutine_handle<promise_type>::from_promise(*this) };
        }
    };
    std::coroutine_handle<promise_type> handle;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    bool next() {
        if (handle.done()) return false;
        handle.resume();
        return !handle.done();
    }
    int value() const { return handle.promise().value; }
};

Generator counter(int limit) {
    for (int i = 0; i < limit; ++i)
        co_yield i;
}

// --- 7. Designated initializers ---
struct WindowConfig {
    int width;
    int height;
    bool fullscreen;
};

// --- 9. Template lambda (explicit template parameter) ---
auto vecSize = []<typename T>(const std::vector<T>& v) { return v.size(); };

int main() {
    std::cout << "=== C++20 Feature Demonstration ===\n\n";

    // 1. Concepts
    std::cout << "[Concepts] add(2,3)     = " << add(2, 3) << "\n";
    std::cout << "[Concepts] add(2.5,1.5) = " << add(2.5, 1.5) << "\n\n";

    // 2. Ranges
    std::vector<int> nums{1, 2, 3, 4, 5, 6, 7, 8};
    auto even_squares = nums
        | std::views::filter([](int n) { return n % 2 == 0; })
        | std::views::transform([](int n) { return n * n; });
    std::cout << "[Ranges] even squares: ";
    for (int n : even_squares) std::cout << n << " ";
    std::cout << "\n\n";

    // 3. Coroutines (guaranteed copy elision makes this move/copy-free since C++17)
    std::cout << "[Coroutines] counter(5) values: ";
    Generator gen = counter(5);
    while (gen.next()) std::cout << gen.value() << " ";
    std::cout << "\n\n";

    // 4. Modules — see the separate two-file example below this program.
    std::cout << "[Modules] see separate math.cppm / main_modules.cpp example below\n\n";

    // 5. Spaceship operator
    Point p1{1, 2}, p2{1, 3};
    std::cout << std::boolalpha;
    std::cout << "[Spaceship] p1 < p2  = " << (p1 < p2) << "\n";
    std::cout << "[Spaceship] p1 == p2 = " << (p1 == p2) << "\n\n";

    // 6. consteval / constinit
    constexpr int sq = ct_square(5);
    std::cout << "[consteval] ct_square(5) = " << sq << "\n";
    std::cout << "[constinit] global_counter = " << global_counter << "\n\n";

    // 7. Designated initializers
    WindowConfig cfg{ .width = 1920, .height = 1080, .fullscreen = true };
    std::cout << "[Designated init] " << cfg.width << "x" << cfg.height
              << " fullscreen=" << cfg.fullscreen << "\n\n";

    // 8. std::span
    int arr[]{1, 2, 3, 4, 5};
    std::span<int> sp(arr);
    for (auto& x : sp) x *= 2;
    std::cout << "[span] doubled array: ";
    for (auto x : sp) std::cout << x << " ";
    std::cout << "\n\n";

    // 9. Template lambda
    std::vector<double> dv{1.1, 2.2, 3.3};
    std::cout << "[Template lambda] vecSize(dv) = " << vecSize(dv) << "\n\n";

    // 10. std::format
    std::string formatted = std::format("{} is {} years old", "Alice", 30);
    std::cout << "[std::format] " << formatted << "\n";

    std::cout << "\n=== End of C++20 Demonstration ===\n";
    return 0;
}
