#include <iostream>

struct Node {
    int data;
    Node* next;
};

class SinglyLinkedList {
public:
    SinglyLinkedList() : head(nullptr) {}

    void append(int value) {
        Node* newNode = new Node{value, nullptr};
        if (!head) {
            head = newNode;
        } else {
            Node* current = head;
            while (current->next) {
                current = current->next;
            }
            current->next = newNode;
        }
    }

    void reverse() {
        Node* prev = nullptr;
        Node* current = head;
        Node* next = nullptr;

        while (current) {
            next = current->next;  // Store next node
            current->next = prev;  // Reverse current node's pointer
            prev = current;        // Move pointers one position ahead
            current = next;
        }
        head = prev;  // Update head to the new first element
    }

    void display() const {
        Node* current = head;
        while (current) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

private:
    Node* head;
};

int main() {
    SinglyLinkedList list;
    list.append(1);
    list.append(2);
    list.append(3);
    
    std::cout << "Initial linked list: ";
    list.display();

    list.reverse();
    std::cout << "Reversed linked list: ";
    list.display();

    return 0;
}
