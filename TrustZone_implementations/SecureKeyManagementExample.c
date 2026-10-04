// trustzone_key_management.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/**
 * Secure Key Hierarchy
 * 
 * Device Key (HUK - Hardware Unique Key)
 *     ↓
 * Master Key (derived from HUK)
 *     ├→ Encryption Key
 *     ├→ Authentication Key
 *     └→ Attestation Key
 */

#define SECURE_KEY_SIZE 32
#define MAX_KEY_SLOTS 16

typedef enum {
    KEY_TYPE_MASTER = 0,
    KEY_TYPE_ENCRYPTION = 1,
    KEY_TYPE_AUTHENTICATION = 2,
    KEY_TYPE_ATTESTATION = 3,
    KEY_TYPE_SEALING = 4
} key_type_t;

typedef struct {
    uint32_t key_id;
    key_type_t key_type;
    uint8_t key_material[SECURE_KEY_SIZE];
    uint32_t creation_time;
    uint32_t expiration_time;
    uint32_t access_count;
    uint8_t is_exportable;
    uint8_t is_sensitive;
} secure_key_slot_t;

typedef struct {
    uint8_t hardware_unique_key[SECURE_KEY_SIZE]; // Never exported
    uint32_t key_version;
    secure_key_slot_t key_slots[MAX_KEY_SLOTS];
    uint32_t slot_count;
} key_hierarchy_t;

static key_hierarchy_t g_key_hierarchy;
static uint32_t g_key_id_counter = 1000;

/**
 * Initialize Key Hierarchy
 */
void key_init_hierarchy(void) {
    printf("[SECURE WORLD] Initializing Key Hierarchy...\n");
    
    memset(&g_key_hierarchy, 0, sizeof(key_hierarchy_t));
    
    // Simulate HUK (from secure fuse/OTP)
    for (int i = 0; i < SECURE_KEY_SIZE; i++) {
        g_key_hierarchy.hardware_unique_key[i] = (uint8_t)(i * 19 % 256);
    }
    
    g_key_hierarchy.key_version = 1;
    g_key_hierarchy.slot_count = 0;
    
    printf("[SECURE WORLD] ✓ Key Hierarchy initialized\n");
    printf("[SECURE WORLD] HUK Version: %u\n", g_key_hierarchy.key_version);
}

/**
 * KDF - Key Derivation Function
 * Derives child keys from master key
 */
void key_derive(const uint8_t *parent_key, 
                const char *derivation_path, 
                uint8_t *derived_key) {
    printf("[SECURE WORLD] Deriving key from path: %s\n", derivation_path);
    
    // Simplified KDF (in real implementation: use HKDF or similar)
    uint8_t temp[SECURE_KEY_SIZE];
    memcpy(temp, parent_key, SECURE_KEY_SIZE);
    
    // Mix with derivation path
    for (int i = 0; i < SECURE_KEY_SIZE && derivation_path[i]; i++) {
        temp[i] ^= (uint8_t)derivation_path[i];
    }
    
    memcpy(derived_key, temp, SECURE_KEY_SIZE);
}

/**
 * Import Key into Secure Storage
 */
uint32_t key_import(key_type_t type, 
                    const uint8_t *key_material,
                    uint32_t expiration_time,
                    uint8_t exportable) {
    printf("[SECURE WORLD] Importing key (type: %u)...\n", type);
    
    if (g_key_hierarchy.slot_count >= MAX_KEY_SLOTS) {
        printf("[SECURE WORLD] ✗ No available key slots\n");
        return 0;
    }
    
    int slot = g_key_hierarchy.slot_count++;
    uint32_t key_id = g_key_id_counter++;
    
    g_key_hierarchy.key_slots[slot].key_id = key_id;
    g_key_hierarchy.key_slots[slot].key_type = type;
    memcpy(g_key_hierarchy.key_slots[slot].key_material, key_material, SECURE_KEY_SIZE);
    g_key_hierarchy.key_slots[slot].creation_time = (uint32_t)time(NULL);
    g_key_hierarchy.key_slots[slot].expiration_time = expiration_time;
    g_key_hierarchy.key_slots[slot].access_count = 0;
    g_key_hierarchy.key_slots[slot].is_exportable = exportable;
    g_key_hierarchy.key_slots[slot].is_sensitive = (type != KEY_TYPE_ENCRYPTION);
    
    printf("[SECURE WORLD] ✓ Key 0x%X imported (slot %d)\n", key_id, slot);
    return key_id;
}

/**
 * Use Key for Cryptographic Operation
 */
int key_use(uint32_t key_id, const uint8_t *data, uint32_t data_len) {
    printf("[SECURE WORLD] Using key 0x%X for operation...\n", key_id);
    
    for (int i = 0; i < g_key_hierarchy.slot_count; i++) {
        if (g_key_hierarchy.key_slots[i].key_id == key_id) {
            // Check expiration
            if (g_key_hierarchy.key_slots[i].expiration_time != 0 &&
                g_key_hierarchy.key_slots[i].expiration_time < (uint32_t)time(NULL)) {
                printf("[SECURE WORLD] ✗ Key expired\n");
                return -1;
            }
            
            // Increment access count
            g_key_hierarchy.key_slots[i].access_count++;
            
            // Perform operation (simplified)
            printf("[SECURE WORLD] ✓ Operation completed (access count: %u)\n",
                   g_key_hierarchy.key_slots[i].access_count);
            
            return 0;
        }
    }
    
    printf("[SECURE WORLD] ✗ Key not found\n");
    return -1;
}

/**
 * Destroy Key
 */
int key_destroy(uint32_t key_id) {
    printf("[SECURE WORLD] Destroying key 0x%X...\n", key_id);
    
    for (int i = 0; i < g_key_hierarchy.slot_count; i++) {
        if (g_key_hierarchy.key_slots[i].key_id == key_id) {
            // Securely wipe key material
            volatile uint8_t *volatile_key = 
                (volatile uint8_t *volatile)g_key_hierarchy.key_slots[i].key_material;
            for (int j = 0; j < SECURE_KEY_SIZE; j++) {
                volatile_key[j] = 0;
            }
            
            // Mark slot as empty
            g_key_hierarchy.key_slots[i].key_id = 0;
            
            printf("[SECURE WORLD] ✓ Key destroyed\n");
            return 0;
        }
    }
    
    printf("[SECURE WORLD] ✗ Key not found\n");
    return -1;
}

/**
 * List Keys
 */
void key_list(void) {
    printf("\n[SECURE WORLD] Key Storage:\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║ Key ID     │ Type │ Accesses │ Exportable │ Sensitive║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    
    for (int i = 0; i < g_key_hierarchy.slot_count; i++) {
        if (g_key_hierarchy.key_slots[i].key_id != 0) {
            printf("║ 0x%08X │ %4u │ %8u │ %10u │ %9u║\n",
                   g_key_hierarchy.key_slots[i].key_id,
                   g_key_hierarchy.key_slots[i].key_type,
                   g_key_hierarchy.key_slots[i].access_count,
                   g_key_hierarchy.key_slots[i].is_exportable,
                   g_key_hierarchy.key_slots[i].is_sensitive);
        }
    }
    
    printf("╚══════════════════════════════════════════════════════╝\n");
}

/**
 * Example: Key Hierarchy Management
 */
void example_key_hierarchy(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║        Example: Key Hierarchy Management                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    key_init_hierarchy();
    
    // Derive master key from HUK
    uint8_t master_key[SECURE_KEY_SIZE];
    key_derive(g_key_hierarchy.hardware_unique_key, "master", master_key);
    
    // Derive child keys
    uint8_t enc_key[SECURE_KEY_SIZE];
    uint8_t auth_key[SECURE_KEY_SIZE];
    uint8_t attest_key[SECURE_KEY_SIZE];
    
    key_derive(master_key, "encryption", enc_key);
    key_derive(master_key, "authentication", auth_key);
    key_derive(master_key, "attestation", attest_key);
    
    printf("\n[SECURE WORLD] Importing derived keys...\n");
    
    // Import keys with expiration
    uint32_t enc_key_id = key_import(KEY_TYPE_ENCRYPTION, enc_key, 
                                     (uint32_t)time(NULL) + 86400, 0);
    uint32_t auth_key_id = key_import(KEY_TYPE_AUTHENTICATION, auth_key, 
                                      (uint32_t)time(NULL) + 86400, 0);
    uint32_t attest_key_id = key_import(KEY_TYPE_ATTESTATION, attest_key, 0, 0);
    
    key_list();
    
    // Use keys
    printf("\n[SECURE WORLD] Using keys for operations...\n");
    uint8_t data[32] = {0x01, 0x02, 0x03};
    key_use(enc_key_id, data, 32);
    key_use(auth_key_id, data, 32);
    key_use(attest_key_id, data, 32);
    key_use(attest_key_id, data, 32); // Use again
    
    key_list();
    
    // Destroy key
    printf("\n[SECURE WORLD] Destroying encryption key...\n");
    key_destroy(enc_key_id);
    
    key_list();
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║        TrustZone Key Management Example                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    example_key_hierarchy();
    
    return 0;
}
