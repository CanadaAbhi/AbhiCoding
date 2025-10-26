#include <iostream>
#include <map>
using namespace std;

int main() {
    map<string, int> marks;

    // Insert using operator[]
    marks["Alice"] = 88;
    marks["Bob"] = 75;

    // Insert using insert()
    marks.insert({"Charlie", 92});
    marks.insert(make_pair("Dave", 85));

    // Traversal
    cout << "Marks:\n";
    for (auto &p : marks)
        cout << p.first << ": " << p.second << endl;

    // Access / Modify
    marks["Alice"] = 90;

    // Find key
    if (marks.find("Charlie") != marks.end())
        cout << "Charlie scored " << marks["Charlie"] << endl;

    // Count (0 or 1)
    cout << "Count of key 'Eve': " << marks.count("Eve") << endl;

    // Erase key
    marks.erase("Bob");
    cout << "After removing Bob:\n";
    for (auto &p : marks)
        cout << p.first << ": " << p.second << endl;

    // Lower and upper bounds
    cout << "Lower bound: " << marks.lower_bound("Charlie")->first << endl;
}
