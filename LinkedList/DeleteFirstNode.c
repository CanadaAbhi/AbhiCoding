#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

void delete_value(Node **head, int value) {
    Node *cur = *head;
    Node *prev = NULL;
    while (cur) {
        if (cur->data == value) {
            if (prev) {
                prev->next = cur->next;
            } else {
                *head = cur->next;
            }
            free(cur);
            return;
        }
        prev = cur;
        cur = cur->next;
    }
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
    Node *head = NULL;
    // Creating list 1->2->3
    head = (Node*)malloc(sizeof(Node)); head->data=1; head->next=NULL;
    head->next = (Node*)malloc(sizeof(Node)); head->next->data=2; head->next->next=NULL;
    head->next->next = (Node*)malloc(sizeof(Node)); head->next->next->data=3; head->next->next->next=NULL;

    printf("Before delete: ");
    print_list(head);

    delete_value(&head, 2);

    printf("After delete: ");
    print_list(head);

    // Free remaining nodes omitted for brevity
    return 0;
}
