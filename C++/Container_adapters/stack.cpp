#include <stack>
#include <iostream>

int main() {
    std::stack<int> st;

    st.push(10);
    st.push(20);
    st.push(30);

    std::cout << "Stack size: " << st.size() << "\n";
    while (!st.empty()) {
        std::cout << st.top() << " "; // access top
        st.pop();                     // remove top
    }
}
