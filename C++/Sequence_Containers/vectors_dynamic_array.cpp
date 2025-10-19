#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> v = {1, 2, 3};

    // Insertions
    v.push_back(4);
    v.insert(v.begin() + 1, 10);  // Insert at position 1

    // Access
    cout << "Element at index 2: " << v.at(2) << endl;
    cout << "Front: " << v.front() << " | Back: " << v.back() << endl;

    // Deletion
    v.pop_back();
    v.erase(v.begin());  // Remove first element

    // Iteration
    cout << "Vector: ";
    for (auto x : v) cout << x << " ";
    cout << endl;

    // Capacity functions
    cout << "Size: " << v.size() << " | Capacity: " << v.capacity() << endl;
}
