#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

Node *reverse_list(Node *head) {
    Node *prev = NULL, *curr = head;
    while (curr) {
        Node *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

Node *make_node(int val) {
    Node *n = malloc(sizeof(Node));
    n->data = val;
    n->next = NULL;
    return n;
}

void print_list(Node *head) {
    while (head) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

void free_list(Node *head) {
    while (head) {
        Node *next = head->next;
        free(head);
        head = next;
    }
}

int main(void) {
    Node *head = make_node(1);
    head->next = make_node(2);
    head->next->next = make_node(3);
    head->next->next->next = make_node(4);
    head->next->next->next->next = make_node(5);

    printf("Original: ");
    print_list(head);

    head = reverse_list(head);

    printf("Reversed: ");
    print_list(head);

    free_list(head);
    return 0;
}
