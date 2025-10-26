#include <queue>
#include <iostream>

int main() {
    std::queue<std::string> q;

    q.push("first");
    q.push("second");
    q.push("third");

    std::cout << "Queue size: " << q.size() << "\n";
    while (!q.empty()) {
        std::cout << q.front() << " "; // access front
        q.pop();                       // remove front
    }
}
