#include <iostream>
#include <deque>
#include <string>

bool isPalindrome(const std::string& str) {
    std::deque<char> dq;
    for (char c : str) {
        dq.push_back(c);
    }

    while (dq.size() > 1) {
        if (dq.front() != dq.back()) {
            return false;
        }
        dq.pop_front();
        dq.pop_back();
    }
    return true;
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
