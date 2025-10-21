#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Insert a new node at the front
void insert_front(Node **head, int value) {
    Node *new_node = (Node*)malloc(sizeof(Node));
    if (new_node == NULL) {
        // allocation error
        return;
    }
    new_node->data = value;
    new_node->next = *head;
    *head = new_node;
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
    insert_front(&head, 10);
    insert_front(&head, 20);
    insert_front(&head, 30);
    print_list(head);  // Should print: 30 -> 20 -> 10 -> NULL
    // Freeing omitted for brevity
    return 0;
}
