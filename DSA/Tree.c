/* ============================================================
 * EMBEDDED TREE DATA STRUCTURES - COMPLETE C IMPLEMENTATION
 * Compile: gcc -O2 -Wall -Wextra tree_impl.c -o tree_impl
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

/* ============================================================
 * SECTION 1: BINARY TREE TRAVERSAL (ITERATIVE)
 * ============================================================
 * Why iterative > recursive in embedded:
 * - Recursion uses O(h) call-stack space where h = tree height
 * - Embedded stacks are typically 4KB-64KB; a skewed tree of
 *   ~5000 nodes can blow a 64KB stack (each frame ~64-128 bytes)
 * - Iterative traversal uses an explicit, BOUNDED stack that we
 *   control and can size-check at compile time
 * - No hidden stack frames, no risk of silent corruption
 * ============================================================ */

typedef struct TreeNode {
    int32_t value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

#define MAX_STACK_SIZE 1024
#define MAX_QUEUE_SIZE 2048

/* ---------- Bounded Stack ---------- */
typedef struct {
    TreeNode *data[MAX_STACK_SIZE];
    int32_t top;
} Stack;

static Stack* stack_create(void) {
    Stack *st = (Stack *)malloc(sizeof(Stack));
    if (!st) { fprintf(stderr, "FATAL: stack alloc failed\n"); exit(1); }
    st->top = -1;
    return st;
}

static bool stack_push(Stack *st, TreeNode *node) {
    if (st->top >= MAX_STACK_SIZE - 1) {
        fprintf(stderr, "ERROR: stack overflow (max=%d)\n", MAX_STACK_SIZE);
        return false;
    }
    st->data[++st->top] = node;
    return true;
}

static TreeNode* stack_pop(Stack *st) {
    if (st->top < 0) return NULL;
    return st->data[st->top--];
}

static bool stack_empty(Stack *st) {
    return st->top < 0;
}

static void stack_destroy(Stack *st) {
    free(st);
}

/* ---------- Bounded Queue (circular-safe, simple linear here) ---------- */
typedef struct {
    TreeNode *data[MAX_QUEUE_SIZE];
    int32_t front, rear;
} Queue;

static Queue* queue_create(void) {
    Queue *q = (Queue *)malloc(sizeof(Queue));
    if (!q) { fprintf(stderr, "FATAL: queue alloc failed\n"); exit(1); }
    q->front = 0;
    q->rear = 0;
    return q;
}

static bool queue_enqueue(Queue *q, TreeNode *node) {
    if (q->rear >= MAX_QUEUE_SIZE) {
        fprintf(stderr, "ERROR: queue overflow (max=%d)\n", MAX_QUEUE_SIZE);
        return false;
    }
    q->data[q->rear++] = node;
    return true;
}

static TreeNode* queue_dequeue(Queue *q) {
    if (q->front >= q->rear) return NULL;
    return q->data[q->front++];
}

static bool queue_empty(Queue *q) {
    return q->front >= q->rear;
}

static void queue_destroy(Queue *q) {
    free(q);
}

/* ---------- Node helpers ---------- */
static TreeNode* create_node(int32_t value) {
    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    if (!node) { fprintf(stderr, "FATAL: node alloc failed\n"); exit(1); }
    node->value = value;
    node->left = NULL;
    node->right = NULL;
    return node;
}

static void destroy_tree(TreeNode *root) {
    if (!root) return;
    destroy_tree(root->left);
    destroy_tree(root->right);
    free(root);
}

/* ---------- 1.1 In-order (Iterative) ---------- */
static void inorder_iterative(TreeNode *root) {
    if (!root) { printf("\n"); return; }
    Stack *st = stack_create();
    TreeNode *curr = root;

    while (curr || !stack_empty(st)) {
        while (curr) {
            stack_push(st, curr);
            curr = curr->left;
        }
        curr = stack_pop(st);
        printf("%d ", curr->value);
        curr = curr->right;
    }
    printf("\n");
    stack_destroy(st);
}

/* ---------- 1.2 Pre-order (Iterative) ---------- */
static void preorder_iterative(TreeNode *root) {
    if (!root) { printf("\n"); return; }
    Stack *st = stack_create();
    stack_push(st, root);

    while (!stack_empty(st)) {
        TreeNode *curr = stack_pop(st);
        printf("%d ", curr->value);
        if (curr->right) stack_push(st, curr->right);
        if (curr->left)  stack_push(st, curr->left);
    }
    printf("\n");
    stack_destroy(st);
}

/* ---------- 1.3 Post-order (Iterative, two-stack) ---------- */
static void postorder_iterative(TreeNode *root) {
    if (!root) { printf("\n"); return; }
    Stack *st1 = stack_create();
    Stack *st2 = stack_create();
    stack_push(st1, root);

    while (!stack_empty(st1)) {
        TreeNode *curr = stack_pop(st1);
        stack_push(st2, curr);
        if (curr->left)  stack_push(st1, curr->left);
        if (curr->right) stack_push(st1, curr->right);
    }
    while (!stack_empty(st2)) {
        printf("%d ", stack_pop(st2)->value);
    }
    printf("\n");
    stack_destroy(st1);
    stack_destroy(st2);
}

/* ---------- 1.4 Level-order (BFS, Queue-based, bounded) ---------- */
static void levelorder_traversal(TreeNode *root) {
    if (!root) { printf("\n"); return; }
    Queue *q = queue_create();
    queue_enqueue(q, root);

    while (!queue_empty(q)) {
        TreeNode *curr = queue_dequeue(q);
        printf("%d ", curr->value);
        if (curr->left)  queue_enqueue(q, curr->left);
        if (curr->right) queue_enqueue(q, curr->right);
    }
    printf("\n");
    queue_destroy(q);
}

/* ---------- 1.5 Morris In-order (O(1) space, no stack at all) ----------
 * Uses threaded right pointers temporarily to eliminate the need
 * for an explicit stack. Restores tree structure after traversal.
 * Ideal for RAM-starved microcontrollers.
 */
static void morris_inorder(TreeNode *root) {
    TreeNode *curr = root;

    while (curr) {
        if (!curr->left) {
            printf("%d ", curr->value);
            curr = curr->right;
        } else {
            TreeNode *pred = curr->left;
            while (pred->right && pred->right != curr) {
                pred = pred->right;
            }
            if (!pred->right) {
                pred->right = curr;   /* thread */
                curr = curr->left;
            } else {
                pred->right = NULL;   /* unthread, restore */
                printf("%d ", curr->value);
                curr = curr->right;
            }
        }
    }
    printf("\n");
}

/* ============================================================
 * SECTION 2: BINARY SEARCH TREE VALIDATION
 * ============================================================ */

/* ---------- 2.1 NAIVE (WRONG) — only checks immediate children ---------- */
static bool isValidBST_naive(TreeNode *root) {
    if (!root) return true;
    if (root->left  && root->left->value  >= root->value) return false;
    if (root->right && root->right->value <= root->value) return false;
    return isValidBST_naive(root->left) && isValidBST_naive(root->right);
}
/* FAILS on: [5,1,15,null,null,6,20] -> node 6 < 15 (ok locally)
   but 6 < 5 violates ancestor bound -> should be invalid, naive says valid */

/* ---------- 2.2 CORRECT (recursive, min/max bounds) ---------- */
static bool isValidBST_bounds(TreeNode *root, int64_t min, int64_t max) {
    if (!root) return true;
    if (root->value <= min || root->value >= max) return false;
    return isValidBST_bounds(root->left, min, root->value) &&
           isValidBST_bounds(root->right, root->value, max);
}

static bool isValidBST(TreeNode *root) {
    return isValidBST_bounds(root, INT64_MIN, INT64_MAX);
}

/* ---------- 2.3 CORRECT (iterative, explicit stack — embedded-safe) ---------- */
typedef struct {
    TreeNode *node;
    int64_t min;
    int64_t max;
} ValidationFrame;

static bool isValidBST_iterative(TreeNode *root) {
    if (!root) return true;

    ValidationFrame *frames = (ValidationFrame *)malloc(sizeof(ValidationFrame) * MAX_STACK_SIZE);
    int32_t top = -1;

    frames[++top] = (ValidationFrame){ root, INT64_MIN, INT64_MAX };

    while (top >= 0) {
        ValidationFrame f = frames[top--];
        TreeNode *curr = f.node;
        if (!curr) continue;

        if (curr->value <= f.min || curr->value >= f.max) {
            free(frames);
            return false;
        }
        if (curr->left) {
            if (top >= MAX_STACK_SIZE - 1) { free(frames); return false; }
            frames[++top] = (ValidationFrame){ curr->left, f.min, curr->value };
        }
        if (curr->right) {
            if (top >= MAX_STACK_SIZE - 1) { free(frames); return false; }
            frames[++top] = (ValidationFrame){ curr->right, curr->value, f.max };
        }
    }
    free(frames);
    return true;
}

/* ---------- 2.4 In-order strictly-increasing validation ---------- */
static bool isValidBST_inorder(TreeNode *root) {
    if (!root) return true;
    Stack *st = stack_create();
    TreeNode *curr = root;
    int64_t prev = INT64_MIN;
    bool valid = true;

    while (curr || !stack_empty(st)) {
        while (curr) {
            stack_push(st, curr);
            curr = curr->left;
        }
        curr = stack_pop(st);
        if (curr->value <= prev) { valid = false; break; }
        prev = curr->value;
        curr = curr->right;
    }
    stack_destroy(st);
    return valid;
}

/* ============================================================
 * SECTION 3: LOWEST COMMON ANCESTOR (LCA)
 * ============================================================ */

/* ---------- 3.1 General binary tree, recursive, O(h) space ---------- */
static TreeNode* lca_general_recursive(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (!root || root == p || root == q) return root;

    TreeNode *left  = lca_general_recursive(root->left, p, q);
    TreeNode *right = lca_general_recursive(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

/* ---------- 3.2 BST, iterative, O(log n) time, O(1) space ---------- */
static TreeNode* lca_bst_iterative(TreeNode *root, TreeNode *p, TreeNode *q) {
    while (root) {
        if (p->value < root->value && q->value < root->value) {
            root = root->left;
        } else if (p->value > root->value && q->value > root->value) {
            root = root->right;
        } else {
            return root;
        }
    }
    return NULL;
}

/* ---------- 3.3 LCA with parent pointers (iterative, cross-reference scenario) ---------- */
typedef struct TreeNodeP {
    int32_t value;
    struct TreeNodeP *left, *right, *parent;
} TreeNodeP;

static int32_t node_depth(TreeNodeP *node) {
    int32_t d = 0;
    while (node) { d++; node = node->parent; }
    return d;
}

static TreeNodeP* lca_with_parent(TreeNodeP *p, TreeNodeP *q) {
    int32_t dp = node_depth(p);
    int32_t dq = node_depth(q);

    while (dp > dq) { p = p->parent; dp--; }
    while (dq > dp) { q = q->parent; dq--; }
    while (p != q)  { p = p->parent; q = q->parent; }
    return p;
}

/* ============================================================
 * SECTION 4: DEVICE TREE PARSING & NAVIGATION
 * ============================================================ */

#define MAX_CHILDREN        32
#define MAX_PROPERTIES      16
#define MAX_NAME_LEN        64
#define MAX_VALUE_LEN       256
#define MAX_PATH_LEN        512
#define DT_CACHE_SIZE       10000   /* supports 10K nodes */

typedef struct {
    char name[MAX_NAME_LEN];
    char value[MAX_VALUE_LEN];
} DeviceProperty;

typedef struct DeviceTreeNode {
    char name[MAX_NAME_LEN];
    DeviceProperty properties[MAX_PROPERTIES];
    int32_t property_count;
    struct DeviceTreeNode *children[MAX_CHILDREN];
    int32_t child_count;
    struct DeviceTreeNode *parent;
} DeviceTreeNode;

static DeviceTreeNode* dt_create_node(const char *name) {
    DeviceTreeNode *node = (DeviceTreeNode *)malloc(sizeof(DeviceTreeNode));
    strncpy(node->name, name, MAX_NAME_LEN - 1);
    node->name[MAX_NAME_LEN - 1] = '\0';
    node->property_count = 0;
    node->child_count = 0;
    node->parent = NULL;
    return node;
}

static void dt_add_property(DeviceTreeNode *node, const char *pname, const char *pvalue) {
    if (node->property_count >= MAX_PROPERTIES) return;
    strncpy(node->properties[node->property_count].name, pname, MAX_NAME_LEN - 1);
    strncpy(node->properties[node->property_count].value, pvalue, MAX_VALUE_LEN - 1);
    node->property_count++;
}

static void dt_add_child(DeviceTreeNode *parent, DeviceTreeNode *child) {
    if (parent->child_count >= MAX_CHILDREN) return;
    parent->children[parent->child_count++] = child;
    child->parent = parent;
}

/* ---------- 4.1 Find node by path: /soc/uart@0 ---------- */
static DeviceTreeNode* dt_find_by_path(DeviceTreeNode *root, const char *path) {
    if (!root || !path) return NULL;

    char path_copy[MAX_PATH_LEN];
    strncpy(path_copy, path, sizeof(path_copy) - 1);
    path_copy[sizeof(path_copy) - 1] = '\0';

    char *token = strtok(path_copy, "/");
    DeviceTreeNode *curr = root;

    while (token) {
        DeviceTreeNode *found = NULL;
        for (int32_t i = 0; i < curr->child_count; i++) {
            if (strcmp(curr->children[i]->name, token) == 0) {
                found = curr->children[i];
                break;
            }
        }
        if (!found) return NULL;
        curr = found;
        token = strtok(NULL, "/");
    }
    return curr;
}

/* ---------- 4.2 DFS: find all nodes matching a "compatible" property ---------- */
typedef struct {
    DeviceTreeNode *nodes[DT_CACHE_SIZE];
    int32_t count;
} DTSearchResults;

static void dt_find_by_compatible_dfs(DeviceTreeNode *node, const char *compat,
                                       DTSearchResults *results) {
    if (!node) return;

    for (int32_t i = 0; i < node->property_count; i++) {
        if (strcmp(node->properties[i].name, "compatible") == 0 &&
            strcmp(node->properties[i].value, compat) == 0) {
            if (results->count < DT_CACHE_SIZE) {
                results->nodes[results->count++] = node;
            }
            break;
        }
    }
    for (int32_t i = 0; i < node->child_count; i++) {
        dt_find_by_compatible_dfs(node->children[i], compat, results);
    }
}

/* ---------- 4.3 Real-time constraint: cache all nodes via BFS for O(1) lookup ---------- */
typedef struct {
    DeviceTreeNode *node;
} DTCacheEntry;

typedef struct {
    DTCacheEntry entries[DT_CACHE_SIZE];
    int32_t count;
} DTCache;

/* Bounded BFS queue sized for large device trees */
typedef struct {
    DeviceTreeNode *data[DT_CACHE_SIZE];
    int32_t front, rear;
} DTQueue;

static DTQueue* dt_queue_create(void) {
    DTQueue *q = (DTQueue *)malloc(sizeof(DTQueue));
    q->front = 0; q->rear = 0;
    return q;
}
static void dt_queue_enqueue(DTQueue *q, DeviceTreeNode *n) {
    if (q->rear < DT_CACHE_SIZE) q->data[q->rear++] = n;
}
static DeviceTreeNode* dt_queue_dequeue(DTQueue *q) {
    if (q->front >= q->rear) return NULL;
    return q->data[q->front++];
}
static bool dt_queue_empty(DTQueue *q) { return q->front >= q->rear; }

static DTCache* dt_build_cache(DeviceTreeNode *root) {
    DTCache *cache = (DTCache *)malloc(sizeof(DTCache));
    cache->count = 0;

    DTQueue *q = dt_queue_create();
    dt_queue_enqueue(q, root);

    while (!dt_queue_empty(q)) {
        DeviceTreeNode *curr = dt_queue_dequeue(q);
        if (cache->count < DT_CACHE_SIZE) {
            cache->entries[cache->count++].node = curr;
        }
        for (int32_t i = 0; i < curr->child_count; i++) {
            dt_queue_enqueue(q, curr->children[i]);
        }
    }
    free(q);
    return cache;
}

static void dt_destroy_tree(DeviceTreeNode *root) {
    if (!root) return;
    for (int32_t i = 0; i < root->child_count; i++) {
        dt_destroy_tree(root->children[i]);
    }
    free(root);
}

/* ============================================================
 * SECTION 5: BINARY INDEXED TREE (FENWICK TREE)
 * ============================================================ */

#define MAX_BIT_SIZE 4096

typedef struct {
    int32_t tree[MAX_BIT_SIZE + 1]; /* 1-indexed */
    int32_t n;
} FenwickTree;

static FenwickTree* fenwick_create(int32_t size) {
    FenwickTree *ft = (FenwickTree *)malloc(sizeof(FenwickTree));
    ft->n = size;
    memset(ft->tree, 0, sizeof(ft->tree));
    return ft;
}

static inline int32_t lowbit(int32_t x) { return x & (-x); }

/* O(log n), no rebalancing -> deterministic latency */
static void fenwick_update(FenwickTree *ft, int32_t idx, int32_t delta) {
    for (int32_t i = idx; i <= ft->n; i += lowbit(i)) {
        ft->tree[i] += delta;
    }
}

static int32_t fenwick_query(FenwickTree *ft, int32_t idx) {
    int32_t sum = 0;
    for (int32_t i = idx; i > 0; i -= lowbit(i)) {
        sum += ft->tree[i];
    }
    return sum;
}

static int32_t fenwick_range_query(FenwickTree *ft, int32_t l, int32_t r) {
    return fenwick_query(ft, r) - (l > 1 ? fenwick_query(ft, l - 1) : 0);
}

static void fenwick_destroy(FenwickTree *ft) { free(ft); }

/* ---------- 5.1 Real-time sensor aggregation ---------- */
#define NUM_SENSORS 1000

typedef struct {
    FenwickTree *sum_tree;
    int32_t latest[NUM_SENSORS];
} SensorAggregator;

static SensorAggregator* sensor_aggregator_create(void) {
    SensorAggregator *agg = (SensorAggregator *)malloc(sizeof(SensorAggregator));
    agg->sum_tree = fenwick_create(NUM_SENSORS);
    memset(agg->latest, 0, sizeof(agg->latest));
    return agg;
}

/* O(log n) per update — 1ms sensor tick budget */
static void sensor_update(SensorAggregator *agg, int32_t sensor_id, int32_t temp) {
    int32_t delta = temp - agg->latest[sensor_id];
    fenwick_update(agg->sum_tree, sensor_id + 1, delta);
    agg->latest[sensor_id] = temp;
}

/* O(log n) per query — 10ms window average */
static int32_t sensor_average(SensorAggregator *agg, int32_t start_id, int32_t end_id) {
    int32_t total = fenwick_range_query(agg->sum_tree, start_id + 1, end_id + 1);
    int32_t count = end_id - start_id + 1;
    return total / count;
}

static void sensor_aggregator_destroy(SensorAggregator *agg) {
    fenwick_destroy(agg->sum_tree);
    free(agg);
}

/* ============================================================
 * SECTION 6: TRIE FOR CONFIGURATION KEYWORDS / CLI AUTOCOMPLETE
 * ============================================================ */

#define ALPHABET_SIZE 26

typedef struct TrieNode {
    struct TrieNode *children[ALPHABET_SIZE];
    bool is_word_end;
    char *description;
} TrieNode;

static TrieNode* trie_node_create(void) {
    TrieNode *node = (TrieNode *)malloc(sizeof(TrieNode));
    memset(node->children, 0, sizeof(node->children));
    node->is_word_end = false;
    node->description = NULL;
    return node;
}

typedef struct { TrieNode *root; } Trie;

static Trie* trie_create(void) {
    Trie *t = (Trie *)malloc(sizeof(Trie));
    t->root = trie_node_create();
    return t;
}

static void trie_insert(Trie *t, const char *word, const char *description) {
    TrieNode *curr = t->root;
    for (int32_t i = 0; word[i]; i++) {
        int32_t idx = word[i] - 'a';
        if (idx < 0 || idx >= ALPHABET_SIZE) continue;
        if (!curr->children[idx]) curr->children[idx] = trie_node_create();
        curr = curr->children[idx];
    }
    curr->is_word_end = true;
    if (description) {
        curr->description = (char *)malloc(strlen(description) + 1);
        strcpy(curr->description, description);
    }
}

static bool trie_search(Trie *t, const char *word) {
    TrieNode *curr = t->root;
    for (int32_t i = 0; word[i]; i++) {
        int32_t idx = word[i] - 'a';
        if (idx < 0 || idx >= ALPHABET_SIZE || !curr->children[idx]) return false;
        curr = curr->children[idx];
    }
    return curr->is_word_end;
}

#define MAX_AUTOCOMPLETE_RESULTS 100
#define MAX_WORD_LEN 64

typedef struct {
    char words[MAX_AUTOCOMPLETE_RESULTS][MAX_WORD_LEN];
    int32_t count;
} AutocompleteResults;

static void autocomplete_dfs(TrieNode *node, char *prefix, int32_t len,
                              AutocompleteResults *results) {
    if (!node || results->count >= MAX_AUTOCOMPLETE_RESULTS) return;

    if (node->is_word_end) {
        prefix[len] = '\0';
        strncpy(results->words[results->count++], prefix, MAX_WORD_LEN - 1);
    }
    for (int32_t i = 0; i < ALPHABET_SIZE; i++) {
        if (node->children[i]) {
            prefix[len] = (char)('a' + i);
            autocomplete_dfs(node->children[i], prefix, len + 1, results);
        }
    }
}

static AutocompleteResults trie_autocomplete(Trie *t, const char *prefix) {
    AutocompleteResults results = {0};
    TrieNode *curr = t->root;

    for (int32_t i = 0; prefix[i]; i++) {
        int32_t idx = prefix[i] - 'a';
        if (idx < 0 || idx >= ALPHABET_SIZE || !curr->children[idx]) return results;
        curr = curr->children[idx];
    }
    char buffer[MAX_WORD_LEN];
    strncpy(buffer, prefix, MAX_WORD_LEN - 1);
    autocomplete_dfs(curr, buffer, (int32_t)strlen(prefix), &results);
    return results;
}

/* ---------- CLI Command Handler ---------- */
typedef struct {
    Trie *commands;
} CLIContext;

static CLIContext* cli_create(void) {
    CLIContext *ctx = (CLIContext *)malloc(sizeof(CLIContext));
    ctx->commands = trie_create();
    trie_insert(ctx->commands, "help",   "Show help message");
    trie_insert(ctx->commands, "status", "Print system status");
    trie_insert(ctx->commands, "reset",  "Perform system reset");
    trie_insert(ctx->commands, "memory", "Print memory stats");
    return ctx;
}

static void cli_handle_input(CLIContext *ctx, const char *input) {
    if (trie_search(ctx->commands, input)) {
        printf("Executing command: %s\n", input);
    } else {
        AutocompleteResults results = trie_autocomplete(ctx->commands, input);
        if (results.count > 0) {
            printf("Did you mean:\n");
            for (int32_t i = 0; i < results.count; i++) {
                printf("  %s\n", results.words[i]);
            }
        } else {
            printf("Command not found: %s\n", input);
        }
    }
}

/* ============================================================
 * MAIN: DEMONSTRATES ALL 6 SECTIONS
 * ============================================================ */
int main(void) {

    /* ---- 1. Binary Tree Traversals ---- */
    printf("=== 1. BINARY TREE TRAVERSALS ===\n");
    TreeNode *root = create_node(4);
    root->left = create_node(2);
    root->right = create_node(6);
    root->left->left = create_node(1);
    root->left->right = create_node(3);
    root->right->left = create_node(5);
    root->right->right = create_node(7);

    printf("In-order   : "); inorder_iterative(root);
    printf("Pre-order  : "); preorder_iterative(root);
    printf("Post-order : "); postorder_iterative(root);
    printf("Level-order: "); levelorder_traversal(root);
    printf("Morris IO  : "); morris_inorder(root);

    /* ---- 2. BST Validation ---- */
    printf("\n=== 2. BST VALIDATION ===\n");
    printf("Naive check     : %s\n", isValidBST_naive(root) ? "VALID" : "INVALID");
    printf("Bounds check    : %s\n", isValidBST(root) ? "VALID" : "INVALID");
    printf("Iterative check : %s\n", isValidBST_iterative(root) ? "VALID" : "INVALID");
    printf("Inorder check   : %s\n", isValidBST_inorder(root) ? "VALID" : "INVALID");

    /* Construct an invalid BST to show naive fails */
    TreeNode *bad = create_node(5);
    bad->left = create_node(1);
    bad->right = create_node(15);
    bad->right->left = create_node(6);   /* 6 < 5, violates root, but 6<15 locally OK */
    bad->right->right = create_node(20);
    printf("Bad tree: naive=%s, correct=%s\n",
           isValidBST_naive(bad) ? "VALID(WRONG!)" : "INVALID",
           isValidBST(bad) ? "VALID" : "INVALID(correct)");
    destroy_tree(bad);

    /* ---- 3. Lowest Common Ancestor ---- */
    printf("\n=== 3. LOWEST COMMON ANCESTOR ===\n");
    TreeNode *lca1 = lca_general_recursive(root, root->left->left, root->left->right);
    printf("General LCA(1,3) = %d\n", lca1->value);

    TreeNode *lca2 = lca_bst_iterative(root, root->left->left, root->right->right);
    printf("BST LCA(1,7)     = %d\n", lca2->value);

    /* ---- 4. Device Tree Parsing ---- */
    printf("\n=== 4. DEVICE TREE PARSING ===\n");
    DeviceTreeNode *dt_root = dt_create_node("root");
    DeviceTreeNode *soc = dt_create_node("soc");
    DeviceTreeNode *uart0 = dt_create_node("uart@0");
    dt_add_property(uart0, "compatible", "arm,pl011");
    DeviceTreeNode *gpio0 = dt_create_node("gpio@1000");
    dt_add_property(gpio0, "compatible", "arm,pl061");

    dt_add_child(dt_root, soc);
    dt_add_child(soc, uart0);
    dt_add_child(soc, gpio0);

    DeviceTreeNode *found = dt_find_by_path(dt_root, "/soc/uart@0");
    printf("Path lookup /soc/uart@0 -> %s\n", found ? found->name : "NOT FOUND");

    DTSearchResults dt_results = {0};
    dt_find_by_compatible_dfs(dt_root, "arm,pl011", &dt_results);
    printf("Nodes matching compatible=arm,pl011: %d\n", dt_results.count);

    DTCache *cache = dt_build_cache(dt_root);
    printf("Cached %d device tree nodes (BFS)\n", cache->count);
    free(cache);
    dt_destroy_tree(dt_root);

    /* ---- 5. Fenwick Tree (Sensor Aggregation) ---- */
    printf("\n=== 5. FENWICK TREE / SENSOR AGGREGATION ===\n");
    SensorAggregator *agg = sensor_aggregator_create();
    sensor_update(agg, 50, 2500);   /* 25.00 C */
    sensor_update(agg, 75, 2600);   /* 26.00 C */
    sensor_update(agg, 100, 2400);  /* 24.00 C */
    int32_t avg = sensor_average(agg, 50, 100);
    printf("Average temp sensors[50-100] = %d (x100 C)\n", avg);
    sensor_aggregator_destroy(agg);

    FenwickTree *ft = fenwick_create(10);
    fenwick_update(ft, 1, 3);
    fenwick_update(ft, 2, 2);
    fenwick_update(ft, 3, -1);
    printf("Fenwick prefix sum[1..3] = %d\n", fenwick_query(ft, 3));
    printf("Fenwick range  sum[2..3] = %d\n", fenwick_range_query(ft, 2, 3));
    fenwick_destroy(ft);

    /* ---- 6. Trie / CLI Autocomplete ---- */
    printf("\n=== 6. TRIE / CLI AUTOCOMPLETE ===\n");
    CLIContext *cli = cli_create();
    cli_handle_input(cli, "help");
    cli_handle_input(cli, "he");    /* should suggest help */
    cli_handle_input(cli, "xyz");   /* not found */

    free(cli->commands->root->description);
    free(cli);

    /* ---- Cleanup ---- */
    destroy_tree(root);

    printf("\n=== ALL TESTS COMPLETE ===\n");
    return 0;
}
