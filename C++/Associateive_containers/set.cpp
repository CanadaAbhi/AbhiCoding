#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s = {5, 1, 4, 2, 2, 3};  // duplicates auto-removed

    // Insertion
    s.insert(7);
    s.insert(0);

    // Access / Traversal (ascending order)
    cout << "Set contents: ";
    for (int x : s) cout << x << " ";
    cout << endl;

    // Search
    if (s.find(3) != s.end())
        cout << "3 is found in set." << endl;

    // Count — returns 1 for found, 0 if not
    cout << "Count of 4: " << s.count(4) << endl;

    // Erase
    s.erase(2);
    cout << "After erase(2): ";
    for (int x : s) cout << x << " ";
    cout << endl;

    // Lower/Upper Bound
    cout << "Lower bound of 4: " << *s.lower_bound(4) << endl;
    cout << "Upper bound of 4: " << *s.upper_bound(4) << endl;
}
