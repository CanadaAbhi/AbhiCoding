#include <iostream>
#include <list>
using namespace std;

int main() {
    list<int> lst = {1, 2, 3};

    // Insertions
    lst.push_front(0);
    lst.push_back(4);
    auto it = lst.begin();
    advance(it, 2);
    lst.insert(it, 99);

    // Deletions
    lst.pop_front();
    lst.remove(3); // Remove element with value 3

    // Operations
    lst.reverse();
    lst.sort();

    cout << "List: ";
    for (auto val : lst) cout << val << " ";
    cout << "\nSize: " << lst.size() << endl;
}
