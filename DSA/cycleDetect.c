#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

int has_cycle(Node *head) {
    Node *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return 1;
    }
    return 0;
}

Node *find_cycle_start(Node *head) {
    Node *slow = head, *fast = head;
    int found = 0;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) { found = 1; break; }
    }
    if (!found) return NULL;
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    return slow;
}

Node *make_node(int val) {
    Node *n = malloc(sizeof(Node));
    n->data = val;
    n->next = NULL;
    return n;
}

int main(void) {
    Node *n1 = make_node(1);
    Node *n2 = make_node(2);
    Node *n3 = make_node(3);
    Node *n4 = make_node(4);
    Node *n5 = make_node(5);
    n1->next = n2; n2->next = n3; n3->next = n4; n4->next = n5;

    printf("No-cycle list has_cycle = %d\n", has_cycle(n1));

    n5->next = n3; // create cycle back to n3

    printf("Cyclic list has_cycle = %d\n", has_cycle(n1));

    Node *start = find_cycle_start(n1);
    printf("Cycle starts at node with data = %d (expected 3)\n",
           start ? start->data : -1);

    // NOTE: not freeing memory here since list is cyclic (would need to
    // break the cycle first in a real program).
    n5->next = NULL;
    free(n1); free(n2); free(n3); free(n4); free(n5);
    return 0;
}
