#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node* reverse_list(Node *head) {
    Node *prev = NULL;
    Node *cur = head;
    Node *next = NULL;
    while (cur) {
        next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
    }
    return prev;  // New head
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

    printf("Original: ");
    print_list(head);

    head = reverse_list(head);

    printf("Reversed: ");
    print_list(head);

    // Free nodes omitted for brevity
    return 0;
}
