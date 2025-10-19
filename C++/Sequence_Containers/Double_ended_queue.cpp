#include <iostream>
#include <deque>
using namespace std;

int main() {
    deque<int> dq = {10, 20, 30};

    // Front and back insertion/removal
    dq.push_front(5);
    dq.push_back(40);
    dq.pop_front();
    dq.pop_back();

    // Access
    cout << "Front: " << dq.front() << " | Back: " << dq.back() << endl;
    cout << "Element at index 1: " << dq.at(1) << endl;

    // Iteration
    cout << "Deque: ";
    for (int n : dq) cout << n << " ";
    cout << endl;

    // Capacity
    cout << "Size: " << dq.size() << endl;
}
