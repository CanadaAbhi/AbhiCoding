#include <iostream>
#include <set>
using namespace std;

int main() {
    multiset<int> ms = {5, 2, 2, 4, 5};

    // Insertion
    ms.insert(3);
    ms.insert(3);

    cout << "Multiset: ";
    for (int x : ms) cout << x << " ";
    cout << endl;

    // Count occurrences
    cout << "Count of 3: " << ms.count(3) << endl;

    // Erase all 5s
    ms.erase(5);
    cout << "After erase(5): ";
    for (int x : ms) cout << x << " ";
    cout << endl;

    // Find and erase one occurrence
    auto it = ms.find(2);
    if (it != ms.end()) ms.erase(it);

    cout << "After erasing one 2: ";
    for (int x : ms) cout << x << " ";
}
