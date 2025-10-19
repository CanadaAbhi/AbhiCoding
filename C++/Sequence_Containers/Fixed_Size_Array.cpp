#include <iostream>
#include <array>
#include <algorithm>
using namespace std;

int main() {
    array<int, 5> arr = {5, 2, 3, 1, 4};

    // Access
    cout << "First: " << arr.front() << ", Last: " << arr.back() << endl;
    cout << "At(2): " << arr.at(2) << endl;

    // Modify
    arr.fill(7);
    sort(arr.begin(), arr.end());

    // Iteration
    cout << "Array: ";
    for (auto num : arr) cout << num << " ";
    cout << "\nSize: " << arr.size() << endl;
}
