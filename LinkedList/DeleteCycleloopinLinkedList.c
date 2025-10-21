#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

int has_cycle(Node *head) {
    Node *slow = head;
    Node *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return 1;  // cycle found
        }
    }
    return 0;  // no cycle
}

int main(void) {
    // Create a list: 1->2->3->4 and then make 4->2 to introduce a cycle
    Node *n1 = (Node*)malloc(sizeof(Node)); n1->data=1;
    Node *n2 = (Node*)malloc(sizeof(Node)); n2->data=2;
    Node *n3 = (Node*)malloc(sizeof(Node)); n3->data=3;
    Node *n4 = (Node*)malloc(sizeof(Node)); n4->data=4;

    n1->next = n2;
    n2->next = n3;
    n3->next = n4;
    n4->next = n2;  // cycle here

    printf("Has cycle? %s\n", has_cycle(n1) ? "Yes" : "No");

    // Note: can't free easily due to cycle. In real embedded, you'd avoid cycles or handle differently
    return 0;
}
