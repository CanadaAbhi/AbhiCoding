#include <iostream>
#include <forward_list>
using namespace std;

int main() {
    forward_list<int> fl = {2, 3, 4};

    // Insertions
    fl.push_front(1);
    auto it = fl.begin();
    fl.insert_after(it, 10);  // Insert after first position

    // Deletions
    fl.pop_front();
    fl.remove(3);  // Remove element value 3

    // Operations
    fl.reverse();
    fl.sort();

    cout << "Forward_list: ";
    for (int x : fl) cout << x << " ";
    cout << endl;
}
