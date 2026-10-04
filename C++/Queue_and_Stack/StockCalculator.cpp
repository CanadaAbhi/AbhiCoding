#include <iostream>
#include <stack>
#include <string>
#include <sstream>

int main() {
    std::stack<int> s;
    std::string expression = "3 4 + 2 * 7 /"; // Example Postfix: ((3 + 4) * 2) / 7
    std::istringstream stream(expression);
    std::string token;

    while (stream >> token) {
        if (isdigit(token[0])) { // If token is a number
            s.push(std::stoi(token));
        } else {
            int b = s.top(); s.pop();
            int a = s.top(); s.pop();
            switch (token[0]) {
                case '+': s.push(a + b); break;
                case '-': s.push(a - b); break;
                case '*': s.push(a * b); break;
                case '/': s.push(a / b); break;
            }
        }
    }

    std::cout << "Result: " << s.top() << std::endl; // The final result
    return 0;
}
