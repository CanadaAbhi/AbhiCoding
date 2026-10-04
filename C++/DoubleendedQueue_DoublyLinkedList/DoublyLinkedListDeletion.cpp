#include <iostream>

struct Node {
    int data;
    Node* prev;
    Node* next;
};

class DoublyLinkedList {
public:
    DoublyLinkedList() : head(nullptr), tail(nullptr) {}
    
    void append(int value) {
        Node* newNode = new Node{value, nullptr, nullptr};
        if (!head) {
            head = tail = newNode;
        } else {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        }
    }

    void deleteValue(int value) {
        Node* current = head;
        while (current) {
            if (current->data == value) {
                if (current->prev) current->prev->next = current->next;
                if (current->next) current->next->prev = current->prev;
                if (current == head) head = current->next;
                if (current == tail) tail = current->prev;
                
                delete current;
                return;
            }
            current = current->next;
        }
    }

    void display() {
        Node* current = head;
        while (current) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

private:
    Node* head;
    Node* tail;
};

int main() {
    DoublyLinkedList dll;
    dll.append(1);
    dll.append(2);
    dll.append(3);
    
    std::cout << "Initial Doubly Linked List: ";
    dll.display();

    dll.deleteValue(2);
    std::cout << "After deleting value 2: ";
    dll.display();

    return 0;
}
