// trustzone_basics.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/**
 * TrustZone Architecture Overview
 * 
 * Security States:
 * - Normal World (Non-Secure): Regular OS (Linux, Android)
 * - Secure World (Secure): TEE (OP-TEE, ARM TrustZone)
 * 
 * Communication: SMC (Secure Monitor Call)
 */

#define TRUSTZONE_ENABLED 1
#define SMC_CALL_SECURE_FUNCTION 0x32000000

// ==================== Secure World Structures ====================

/**
 * Secure Context Structure
 * Stored in Secure World memory
 */
typedef struct {
    uint32_t session_id;
    uint32_t client_id;
    uint32_t state;
    uint8_t session_key[32];
    uint32_t creation_time;
    uint32_t last_access_time;
} secure_session_t;

/**
 * Secure World Key Storage
 */
typedef struct {
    uint8_t master_key[32];      // AES-256 key
    uint8_t attestation_key[32]; // Device attestation key
    uint8_t sealing_key[32];     // Data sealing key
    uint8_t authorization_key[32]; // Authorization key
    uint32_t key_version;
} secure_keystore_t;

/**
 * Secure Object Structure
 * Protected data in Secure World
 */
typedef struct {
    uint32_t object_id;
    uint8_t object_data[256];
    uint32_t object_size;
    uint32_t object_type;
    uint8_t object_hash[32]; // SHA-256 hash
    uint32_t access_control;
} secure_object_t;

// ==================== Simulated Secure World ====================

static secure_session_t g_active_sessions[10];
static secure_keystore_t g_secure_keystore;
static secure_object_t g_secure_objects[20];
static uint32_t g_session_counter = 1;
static uint32_t g_object_counter = 1;

/**
 * Initialize Secure World
 */
void trustzone_init(void) {
    printf("[SECURE WORLD] Initializing TrustZone...\n");
    
    // Initialize keystore
    memset(&g_secure_keystore, 0, sizeof(secure_keystore_t));
    
    // Generate master key (simulated)
    for (int i = 0; i < 32; i++) {
        g_secure_keystore.master_key[i] = (uint8_t)(i * 7 % 256);
        g_secure_keystore.attestation_key[i] = (uint8_t)(i * 11 % 256);
        g_secure_keystore.sealing_key[i] = (uint8_t)(i * 13 % 256);
        g_secure_keystore.authorization_key[i] = (uint8_t)(i * 17 % 256);
    }
    g_secure_keystore.key_version = 1;
    
    // Initialize sessions and objects
    memset(g_active_sessions, 0, sizeof(g_active_sessions));
    memset(g_secure_objects, 0, sizeof(g_secure_objects));
    
    printf("[SECURE WORLD] ✓ TrustZone initialized\n");
    printf("[SECURE WORLD] Master Key Version: %u\n", g_secure_keystore.key_version);
}

/**
 * Open Secure Session
 */
uint32_t trustzone_open_session(uint32_t client_id) {
    printf("\n[SECURE WORLD] Opening session for client 0x%X...\n", client_id);
    
    // Find available session slot
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id == 0) {
            uint32_t session_id = g_session_counter++;
            
            g_active_sessions[i].session_id = session_id;
            g_active_sessions[i].client_id = client_id;
            g_active_sessions[i].state = 1; // OPEN
            g_active_sessions[i].creation_time = (uint32_t)time(NULL);
            g_active_sessions[i].last_access_time = g_active_sessions[i].creation_time;
            
            // Generate session key
            for (int j = 0; j < 32; j++) {
                g_active_sessions[i].session_key[j] = (uint8_t)((session_id + j) % 256);
            }
            
            printf("[SECURE WORLD] ✓ Session 0x%X opened\n", session_id);
            return session_id;
        }
    }
    
    printf("[SECURE WORLD] ✗ No available session slots\n");
    return 0;
}

/**
 * Close Secure Session
 */
int trustzone_close_session(uint32_t session_id) {
    printf("\n[SECURE WORLD] Closing session 0x%X...\n", session_id);
    
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id == session_id) {
            g_active_sessions[i].session_id = 0;
            g_active_sessions[i].state = 0;
            memset(g_active_sessions[i].session_key, 0, 32);
            printf("[SECURE WORLD] ✓ Session closed\n");
            return 0;
        }
    }
    
    printf("[SECURE WORLD] ✗ Session not found\n");
    return -1;
}

/**
 * Create Secure Object
 */
uint32_t trustzone_create_object(uint32_t session_id, 
                                 const uint8_t *data, 
                                 uint32_t size) {
    printf("\n[SECURE WORLD] Creating secure object (size: %u bytes)...\n", size);
    
    // Verify session
    int session_idx = -1;
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id == session_id) {
            session_idx = i;
            break;
        }
    }
    
    if (session_idx < 0) {
        printf("[SECURE WORLD] ✗ Invalid session\n");
        return 0;
    }
    
    // Find available object slot
    for (int i = 0; i < 20; i++) {
        if (g_secure_objects[i].object_id == 0) {
            uint32_t object_id = g_object_counter++;
            
            // Copy data
            if (size > 256) size = 256;
            memcpy(g_secure_objects[i].object_data, data, size);
            
            g_secure_objects[i].object_id = object_id;
            g_secure_objects[i].object_size = size;
            g_secure_objects[i].object_type = 1; // Generic object
            g_secure_objects[i].access_control = (1 << session_idx); // ACL
            
            // Compute SHA-256 hash (simplified)
            for (int j = 0; j < 32; j++) {
                g_secure_objects[i].object_hash[j] = (uint8_t)((object_id + j) % 256);
            }
            
            printf("[SECURE WORLD] ✓ Object 0x%X created\n", object_id);
            return object_id;
        }
    }
    
    printf("[SECURE WORLD] ✗ No available object slots\n");
    return 0;
}

/**
 * Read Secure Object
 */
int trustzone_read_object(uint32_t session_id, 
                          uint32_t object_id, 
                          uint8_t *output, 
                          uint32_t *size) {
    printf("\n[SECURE WORLD] Reading object 0x%X...\n", object_id);
    
    // Verify session
    int session_idx = -1;
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id == session_id) {
            session_idx = i;
            break;
        }
    }
    
    if (session_idx < 0) {
        printf("[SECURE WORLD] ✗ Invalid session\n");
        return -1;
    }
    
    // Find object
    for (int i = 0; i < 20; i++) {
        if (g_secure_objects[i].object_id == object_id) {
            // Check ACL
            if (!(g_secure_objects[i].access_control & (1 << session_idx))) {
                printf("[SECURE WORLD] ✗ Access denied\n");
                return -1;
            }
            
            // Copy data
            *size = g_secure_objects[i].object_size;
            memcpy(output, g_secure_objects[i].object_data, *size);
            
            printf("[SECURE WORLD] ✓ Object read (%u bytes)\n", *size);
            return 0;
        }
    }
    
    printf("[SECURE WORLD] ✗ Object not found\n");
    return -1;
}

/**
 * Get Master Key (only for authorized operations)
 */
int trustzone_get_key(uint32_t session_id, 
                      uint32_t key_type, 
                      uint8_t *key_buffer) {
    printf("\n[SECURE WORLD] Requesting key (type: %u)...\n", key_type);
    
    // Verify session
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id == session_id) {
            // Copy appropriate key
            switch (key_type) {
                case 0: // Master Key
                    memcpy(key_buffer, g_secure_keystore.master_key, 32);
                    printf("[SECURE WORLD] ✓ Master key retrieved\n");
                    break;
                case 1: // Attestation Key
                    memcpy(key_buffer, g_secure_keystore.attestation_key, 32);
                    printf("[SECURE WORLD] ✓ Attestation key retrieved\n");
                    break;
                case 2: // Sealing Key
                    memcpy(key_buffer, g_secure_keystore.sealing_key, 32);
                    printf("[SECURE WORLD] ✓ Sealing key retrieved\n");
                    break;
                default:
                    printf("[SECURE WORLD] ✗ Unknown key type\n");
                    return -1;
            }
            return 0;
        }
    }
    
    printf("[SECURE WORLD] ✗ Invalid session\n");
    return -1;
}

/**
 * Print session information
 */
void trustzone_print_sessions(void) {
    printf("\n[SECURE WORLD] Active Sessions:\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║ Session ID │ Client ID  │ State │ Creation Time   ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    
    int count = 0;
    for (int i = 0; i < 10; i++) {
        if (g_active_sessions[i].session_id != 0) {
            printf("║ 0x%08X │ 0x%08X │ %5u │ %15u ║\n",
                   g_active_sessions[i].session_id,
                   g_active_sessions[i].client_id,
                   g_active_sessions[i].state,
                   g_active_sessions[i].creation_time);
            count++;
        }
    }
    
    if (count == 0) {
        printf("║ No active sessions                                ║\n");
    }
    
    printf("╚════════════════════════════════════════════════════╝\n");
}

// ==================== Normal World (Non-Secure) ====================

/**
 * Non-Secure Client Structure
 */
typedef struct {
    uint32_t client_id;
    uint32_t session_id;
    uint32_t state;
} nsc_client_t;

/**
 * Simulated SMC Call Handler
 * Transitions from Normal World to Secure World
 */
uint32_t smc_call(uint32_t command, uint32_t param1, 
                  uint32_t param2, uint32_t param3) {
    printf("\n[NORMAL WORLD] ★ SMC Call: cmd=0x%X p1=0x%X p2=0x%X p3=0x%X\n",
           command, param1, param2, param3);
    
    uint32_t result = 0;
    
    switch (command) {
        case 0x01: // Open Session
            result = trustzone_open_session(param1);
            break;
        case 0x02: // Close Session
            result = trustzone_close_session(param1);
            break;
        case 0x03: // Create Object
            // Note: In real implementation, data would be passed via shared memory
            {
                uint8_t temp_data[256] = {0};
                for (int i = 0; i < param3; i++) {
                    temp_data[i] = (uint8_t)i;
                }
                result = trustzone_create_object(param1, temp_data, param3);
            }
            break;
        case 0x04: // Read Object
            printf("[SECURE WORLD] Read Object call\n");
            result = 0;
            break;
        case 0x05: // Get Key
            printf("[SECURE WORLD] Get Key call\n");
            result = 0;
            break;
        default:
            printf("[SECURE WORLD] ✗ Unknown command\n");
            result = -1;
    }
    
    printf("[NORMAL WORLD] ← SMC Return: 0x%X\n", result);
    return result;
}

/**
 * Normal World Example: Secure Session Management
 */
void example_secure_session(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     Example 1: Secure Session Management                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    nsc_client_t client;
    client.client_id = 0x12345678;
    client.state = 0; // Not authenticated
    
    printf("[NORMAL WORLD] Client 0x%X initializing...\n", client.client_id);
    
    // Open secure session
    client.session_id = smc_call(0x01, client.client_id, 0, 0);
    
    if (client.session_id != 0) {
        client.state = 1; // Authenticated
        printf("[NORMAL WORLD] ✓ Secure session established (0x%X)\n", 
               client.session_id);
        
        trustzone_print_sessions();
        
        // Create secure object
        smc_call(0x03, client.session_id, 0, 64);
        
        // Close session
        smc_call(0x02, client.session_id, 0, 0);
        client.state = 0;
    } else {
        printf("[NORMAL WORLD] ✗ Failed to open session\n");
    }
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          TrustZone Basics Implementation                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Initialize Secure World
    trustzone_init();
    
    // Example 1: Secure Session
    example_secure_session();
    
    printf("\n[NORMAL WORLD] Program completed\n");
    
    return 0;
}
