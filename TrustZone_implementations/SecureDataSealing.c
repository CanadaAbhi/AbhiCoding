// trustzone_sealing.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/**
 * Data Sealing
 * Protects sensitive data by encrypting it with device-specific keys
 * 
 * Use Cases:
 * - Secure storage of credentials
 * - Protection of sensitive application data
 * - Binding data to specific device
 */

#define SEALED_DATA_SIZE 256
#define TAG_SIZE 16

typedef struct {
    uint32_t version;
    uint32_t data_size;
    uint8_t iv[16];
    uint8_t sealed_data[SEALED_DATA_SIZE];
    uint8_t tag[TAG_SIZE];
} sealed_blob_t;

typedef enum {
    SEAL_POLICY_DEVICE = 0,      // Tied to device
    SEAL_POLICY_DEVICE_AND_OS = 1, // Tied to device + OS
    SEAL_POLICY_DEVICE_AND_TEE = 2 // Tied to device + TEE
} seal_policy_t;

/**
 * Seal Data (Secure World)
 */
int seal_data(const uint8_t *plaintext,
              uint32_t plaintext_len,
              seal_policy_t policy,
              sealed_blob_t *sealed) {
    printf("[SECURE WORLD] Sealing data (%u bytes, policy: %u)...\n",
           plaintext_len, policy);
    
    if (plaintext_len > SEALED_DATA_SIZE) {
        printf("[SECURE WORLD] ✗ Data too large\n");
        return -1;
    }
    
    // Initialize
    sealed->version = 1;
    sealed->data_size = plaintext_len;
    
    // Generate IV (simulated random)
    for (int i = 0; i < 16; i++) {
        sealed->iv[i] = (uint8_t)(i * 17);
    }
    
    // Encrypt plaintext (simplified XOR)
    // In real implementation: use AES-GCM
    uint8_t seal_key[32] = {0};
    
    // Derive seal key based on policy
    if (policy == SEAL_POLICY_DEVICE) {
        // Use device-specific key
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)(i * 7);
        }
    } else if (policy == SEAL_POLICY_DEVICE_AND_OS) {
        // Include OS version
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)((i * 7) ^ 14); // OS version 14
        }
    } else if (policy == SEAL_POLICY_DEVICE_AND_TEE) {
        // Include TEE version
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)((i * 7) ^ 1); // TEE version 1
        }
    }
    
    // Encrypt
    for (int i = 0; i < plaintext_len; i++) {
        sealed->sealed_data[i] = plaintext[i] ^ seal_key[i % 32];
    }
    
    // Generate authentication tag (simulated HMAC)
    for (int i = 0; i < TAG_SIZE; i++) {
        sealed->tag[i] = (uint8_t)((i * 11) ^ sealed->sealed_data[i % plaintext_len]);
    }
    
    printf("[SECURE WORLD] ✓ Data sealed\n");
    return 0;
}

/**
 * Unseal Data (Secure World)
 */
int unseal_data(const sealed_blob_t *sealed,
                seal_policy_t policy,
                uint8_t *plaintext) {
    printf("[SECURE WORLD] Unsealing data (policy: %u)...\n", policy);
    
    // Derive seal key (same as during sealing)
    uint8_t seal_key[32] = {0};
    
    if (policy == SEAL_POLICY_DEVICE) {
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)(i * 7);
        }
    } else if (policy == SEAL_POLICY_DEVICE_AND_OS) {
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)((i * 7) ^ 14);
        }
    } else if (policy == SEAL_POLICY_DEVICE_AND_TEE) {
        for (int i = 0; i < 32; i++) {
            seal_key[i] = (uint8_t)((i * 7) ^ 1);
        }
    }
    
    // Verify tag (simplified)
    uint8_t expected_tag[TAG_SIZE];
    for (int i = 0; i < TAG_SIZE; i++) {
        expected_tag[i] = (uint8_t)((i * 11) ^ sealed->sealed_data[i % sealed->data_size]);
    }
    
    if (memcmp(expected_tag, sealed->tag, TAG_SIZE) != 0) {
        printf("[SECURE WORLD] ✗ Authentication tag verification failed\n");
        return -1;
    }
    
    printf("[SECURE WORLD] ✓ Tag verified\n");
    
    // Decrypt
    for (int i = 0; i < sealed->data_size; i++) {
        plaintext[i] = sealed->sealed_data[i] ^ seal_key[i % 32];
    }
    
    printf("[SECURE WORLD] ✓ Data unsealed\n");
    return 0;
}

/**
 * Print hex data
 */
void print_hex(const uint8_t *data, uint32_t len, const char *label) {
    printf("%s: ", label);
    for (uint32_t i = 0; i < len && i < 16; i++) {
        printf("%02x", data[i]);
    }
    if (len > 16) printf("...");
    printf("\n");
}

/**
 * Example: Data Sealing and Unsealing
 */
void example_data_sealing(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          Example: Secure Data Sealing                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Sensitive data to seal
    uint8_t plaintext[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
    };
    uint32_t plaintext_len = sizeof(plaintext);
    
    printf("\n[NORMAL WORLD] Sensitive data to protect:\n");
    print_hex(plaintext, plaintext_len, "Plaintext");
    
    // Seal with different policies
    sealed_blob_t sealed_device;
    sealed_blob_t sealed_device_os;
    sealed_blob_t sealed_device_tee;
    
    printf("\n[SECURE WORLD] Sealing with different policies...\n");
    
    seal_data(plaintext, plaintext_len, SEAL_POLICY_DEVICE, &sealed_device);
    seal_data(plaintext, plaintext_len, SEAL_POLICY_DEVICE_AND_OS, &sealed_device_os);
    seal_data(plaintext, plaintext_len, SEAL_POLICY_DEVICE_AND_TEE, &sealed_device_tee);
    
    printf("\nSealed blobs:\n");
    print_hex(sealed_device.sealed_data, plaintext_len, "DEVICE");
    print_hex(sealed_device_os.sealed_data, plaintext_len, "DEVICE_AND_OS");
    print_hex(sealed_device_tee.sealed_data, plaintext_len, "DEVICE_AND_TEE");
    
    // Unseal
    printf("\n[SECURE WORLD] Unsealing data...\n");
    
    uint8_t decrypted[SEALED_DATA_SIZE];
    
    if (unseal_data(&sealed_device, SEAL_POLICY_DEVICE, decrypted) == 0) {
        print_hex(decrypted, plaintext_len, "Decrypted (DEVICE)");
        
        // Verify
        if (memcmp(plaintext, decrypted, plaintext_len) == 0) {
            printf("✓ Unsealing successful!\n");
        }
    }
    
    // Try to unseal with wrong policy
    printf("\n[SECURE WORLD] Attempting to unseal with wrong policy...\n");
    if (unseal_data(&sealed_device, SEAL_POLICY_DEVICE_AND_OS, decrypted) != 0) {
        printf("✗ Unsealing failed (expected)\n");
    }
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║        TrustZone Data Sealing Example                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    example_data_sealing();
    
    return 0;
}
