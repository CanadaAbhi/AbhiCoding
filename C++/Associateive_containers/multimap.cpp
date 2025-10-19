#include <iostream>
#include <map>
using namespace std;

int main() {
    multimap<string, int> grades;

    // Insert elements (using same keys)
    grades.insert({"Sam", 80});
    grades.insert({"Sam", 85});
    grades.insert({"John", 90});

    // Traversal
    cout << "Multimap Contents:\n";
    for (auto &p : grades)
        cout << p.first << ": " << p.second << endl;

    // Range query for one key
    auto range = grades.equal_range("Sam");
    cout << "All grades for Sam: ";
    for (auto it = range.first; it != range.second; ++it)
        cout << it->second << " ";
    cout << endl;

    // Erase duplicates
    grades.erase("Sam");
    cout << "After erasing Sam:\n";
    for (auto &p : grades)
        cout << p.first << ": " << p.second << endl;
}
