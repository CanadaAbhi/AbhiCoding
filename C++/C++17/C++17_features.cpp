// cpp17_demo.cpp
// Compile:  g++ -std=c++17 -Wall -Wextra cpp17_demo.cpp -o cpp17_demo -lstdc++fs -ltbb
// (-lstdc++fs needed on older GCC for <filesystem>; -ltbb needed for <execution> parallel policies)
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <string_view>
#include <optional>
#include <variant>
#include <any>
#include <filesystem>
#include <algorithm>
#include <execution>
#include <functional>
#include <tuple>
#include <numeric>
#include <type_traits>

namespace fs = std::filesystem;

// --- 3. Inline variables ---
struct Config {
    static inline int version = 1;
};

// --- 2. if constexpr ---
template<typename T>
auto getValue(T t) {
    if constexpr (std::is_pointer_v<T>)
        return *t;
    else
        return t;
}

// --- 4. Fold expressions ---
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

// --- 10. Nested namespace definition ---
namespace Company::Project::Module {
    void run() { std::cout << "[Nested namespace] Company::Project::Module::run() called\n"; }
}

// --- 11. [[nodiscard]] / [[maybe_unused]] ---
[[nodiscard]] int computeImportant() { return 42; }
void debugFunc([[maybe_unused]] int debugOnlyParam) {
    std::cout << "[maybe_unused] debugFunc called\n";
}

// --- 13. std::invoke / std::apply ---
int addFn(int a, int b) { return a + b; }

int main() {
    std::cout << "=== C++17 Feature Demonstration ===\n\n";

    // 1. Structured bindings
    std::map<int, std::string> m{{1, "one"}, {2, "two"}};
    for (auto& [key, value] : m)
        std::cout << "[Structured binding] " << key << " => " << value << "\n";

    auto divmod = [](int a, int b) { return std::pair<int, int>{a / b, a % b}; };
    auto [q, r] = divmod(17, 5);
    std::cout << "[Structured binding] 17/5 => q=" << q << " r=" << r << "\n\n";

    // 2. if constexpr
    int val = 10;
    int* pval = &val;
    std::cout << "[if constexpr] getValue(pval) = " << getValue(pval) << "\n";
    std::cout << "[if constexpr] getValue(val)  = " << getValue(val) << "\n\n";

    // 3. Inline variables
    std::cout << "[Inline variable] Config::version = " << Config::version << "\n\n";

    // 4. Fold expressions
    std::cout << "[Fold expression] sum(1,2,3,4) = " << sum(1, 2, 3, 4) << "\n\n";

    // 5. Class Template Argument Deduction (CTAD)
    std::vector v{1, 2, 3};
    std::pair p{1, std::string("hello")};
    std::cout << "[CTAD] vector<int> size = " << v.size()
              << ", pair = (" << p.first << ", " << p.second << ")\n\n";

    // 6. optional, variant, any
    std::optional<int> maybeVal = std::nullopt;
    std::cout << "[optional] has_value = " << std::boolalpha << maybeVal.has_value() << "\n";
    maybeVal = 42;
    std::cout << "[optional] value = " << maybeVal.value() << "\n";

    std::variant<int, std::string> var = std::string("text");
    std::cout << "[variant] holds string: " << std::get<std::string>(var) << "\n";
    var = 10;
    std::cout << "[variant] holds int: " << std::get<int>(var) << "\n";

    std::any a = 5;
    a = std::string("now a string");
    std::cout << "[any] value = " << std::any_cast<std::string>(a) << "\n\n";

    // 7. string_view
    auto printSV = [](std::string_view sv) {
        std::cout << "[string_view] " << sv << " (len=" << sv.size() << ")\n";
    };
    printSV("literal");
    std::string s = "hello";
    printSV(s);
    std::cout << "\n";

    // 8. filesystem
    fs::path cwd = fs::current_path();
    std::cout << "[filesystem] current_path = " << cwd.string() << "\n";
    int count = 0;
    for (auto& entry : fs::directory_iterator(cwd)) { (void)entry; ++count; }
    std::cout << "[filesystem] entries in current directory = " << count << "\n\n";

    // 9. Parallel algorithms
    std::vector<int> bigVec(100000);
    std::iota(bigVec.rbegin(), bigVec.rend(), 1); // fills descending: 100000 ... 1
    std::sort(std::execution::par, bigVec.begin(), bigVec.end());
    std::cout << "[Parallel algorithm] sorted, first=" << bigVec.front()
              << " last=" << bigVec.back() << "\n\n";

    // 10. Nested namespace
    Company::Project::Module::run();
    std::cout << "\n";

    // 11. [[nodiscard]] / [[maybe_unused]]
    int important = computeImportant();
    std::cout << "[nodiscard] computeImportant() = " << important << "\n";
    debugFunc(123);
    std::cout << "\n";

    // 12. init-statement in if
    if (auto it = m.find(1); it != m.end())
        std::cout << "[init-statement if] found key 1 -> " << it->second << "\n\n";

    // 13. std::invoke / std::apply
    std::cout << "[std::invoke] invoke(addFn,2,3) = " << std::invoke(addFn, 2, 3) << "\n";
    auto t = std::make_tuple(2, 3);
    std::cout << "[std::apply] apply(addFn,t) = " << std::apply(addFn, t) << "\n";

    std::cout << "\n=== End of C++17 Demonstration ===\n";
    return 0;
}
