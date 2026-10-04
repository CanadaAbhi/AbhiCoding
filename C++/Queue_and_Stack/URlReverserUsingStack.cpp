#include <iostream>
#include <stack>
#include <sstream>

int main() {
    std::string url = "www.example.com/path/to/resource";
    std::stack<std::string> segments;
    std::istringstream stream(url);
    std::string segment;

    // Split the URL by '/'
    while (std::getline(stream, segment, '/')) {
        segments.push(segment);
    }

    std::cout << "Reversed URL: ";
    while (!segments.empty()) {
        std::cout << segments.top();
        segments.pop();
        if (!segments.empty()) {
            std::cout << "/";
        }
    }
    std::cout << std::endl;

    return 0;
}
