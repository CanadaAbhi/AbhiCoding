#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node* find_middle(Node *head) {
    Node *slow = head;
    Node *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;  // slow will be middle (for even number it will be 2nd of the two middle)
}

void print_list(Node *head) {
    Node *cur = head;
    while (cur) {
        printf("%d -> ", cur->data);
        cur = cur->next;
    }
    printf("NULL\n");
}

int main(void) {
    Node *head = (Node*)malloc(sizeof(Node)); head->data=1; head->next=NULL;
    head->next = (Node*)malloc(sizeof(Node)); head->next->data=2; head->next->next=NULL;
    head->next->next = (Node*)malloc(sizeof(Node)); head->next->next->data=3; head->next->next->next=NULL;
    head->next->next->next = (Node*)malloc(sizeof(Node)); head->next->next->next->data=4; head->next->next->next->next=NULL;
    head->next->next->next->next = (Node*)malloc(sizeof(Node)); head->next->next->next->next->data=5; head->next->next->next->next->next=NULL;

    print_list(head);
    Node *mid = find_middle(head);
    printf("Middle node data = %d\n", mid->data);

    // Free nodes omitted
    return 0;
}
