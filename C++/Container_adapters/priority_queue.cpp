#include <queue>
#include <iostream>

int main() {
    std::priority_queue<int> pq;

    pq.push(5);
    pq.push(1);
    pq.push(10);

    std::cout << "Priority Queue size: " << pq.size() << "\n";
    while (!pq.empty()) {
        std::cout << pq.top() << " ";  // largest
        pq.pop();
    }
}
