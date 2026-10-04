#include <iostream>
#include <stack>
#include <string>

bool isPalindrome(const std::string& str) {
    std::stack<char> s;
    for (char c : str) {
        s.push(c);
    }
    
    std::string reversed;
    while (!s.empty()) {
        reversed += s.top();
        s.pop();
    }
    
    return str == reversed;
}

int main() {
    std::string input = "madam";
    if (isPalindrome(input)) {
        std::cout << input << " is a palindrome." << std::endl;
    } else {
        std::cout << input << " is not a palindrome." << std::endl;
    }
    return 0;
}
