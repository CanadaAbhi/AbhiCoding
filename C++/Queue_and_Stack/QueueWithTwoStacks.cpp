#include <iostream>
#include <stack>

class Queue {
public:
    void enqueue(int x) {
        stack1.push(x);
    }

    int dequeue() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        if (stack2.empty()) {
            throw std::runtime_error("Queue is empty!");
        }
        int front = stack2.top();
        stack2.pop();
        return front;
    }

private:
    std::stack<int> stack1, stack2;
};

int main() {
    Queue q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);

    std::cout << "Dequeue: " << q.dequeue() << std::endl; // 1
    std::cout << "Dequeue: " << q.dequeue() << std::endl; // 2

    return 0;
}
