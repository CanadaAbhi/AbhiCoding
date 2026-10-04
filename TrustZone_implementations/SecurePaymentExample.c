// trustzone_secure_payment.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/**
 * Secure Payment Processing
 * Demonstrates protected transaction handling in TrustZone
 */

#define TRANSACTION_ID_SIZE 8
#define PIN_SIZE 4
#define SIGNATURE_SIZE 64

typedef struct {
    uint8_t card_number[16];
    uint8_t card_name[32];
    uint16_t expiry_month;
    uint16_t expiry_year;
    uint8_t cvv[3];
} card_info_t;

typedef struct {
    uint8_t transaction_id[TRANSACTION_ID_SIZE];
    uint32_t amount;
    uint32_t merchant_id;
    uint32_t timestamp;
    uint8_t transaction_hash[32];
} transaction_t;

typedef struct {
    card_info_t card;
    uint8_t pin_hash[32];
    uint32_t daily_limit;
    uint32_t daily_spent;
    uint8_t is_locked;
} payment_context_t;

static payment_context_t g_payment_context;

/**
 * Initialize Payment Context
 */
void payment_init(void) {
    printf("[SECURE WORLD] Initializing Secure Payment Processing...\n");
    
    // Initialize card (in real system, loaded from hardware)
    memset(&g_payment_context.card, 0, sizeof(card_info_t));
    strncpy((char *)g_payment_context.card.card_name, "JOHN DOE", 31);
    g_payment_context.card.expiry_month = 12;
    g_payment_context.card.expiry_year = 2025;
    
    // Set card number (masked in output)
    const uint8_t card[] = {0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                            0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50};
    memcpy(g_payment_context.card.card_number, card, 16);
    
    // Set PIN hash (simulated)
    for (int i = 0; i < 32; i++) {
        g_payment_context.pin_hash[i] = (uint8_t)(i * 13 % 256);
    }
    
    g_payment_context.daily_limit = 1000000; // $10,000
    g_payment_context.daily_spent = 0;
    g_payment_context.is_locked = 0;
    
    printf("[SECURE WORLD] ✓ Payment initialized\n");
    printf("[SECURE WORLD] Card Holder: %s\n", g_payment_context.card.card_name);
    printf("[SECURE WORLD] Daily Limit: %u cents\n", g_payment_context.daily_limit);
}

/**
 * Verify PIN
 */
int payment_verify_pin(const uint8_t *pin, uint32_t pin_len) {
    printf("[SECURE WORLD] Verifying PIN...\n");
    
    if (g_payment_context.is_locked) {
        printf("[SECURE WORLD] ✗ Card is locked\n");
        return -1;
    }
    
    // Compute PIN hash (simplified)
    uint8_t computed_hash[32] = {0};
    for (int i = 0; i < pin_len && i < 32; i++) {
        computed_hash[i] = pin[i] * 13 % 256;
    }
    
    // Verify with timing-safe comparison
    int match = 1;
    for (int i = 0; i < 32; i++) {
        if (computed_hash[i] != g_payment_context.pin_hash[i]) {
            match = 0;
        }
    }
    
    if (!match) {
        printf("[SECURE WORLD] ✗ Invalid PIN\n");
        return -1;
    }
    
    printf("[SECURE WORLD] ✓ PIN verified\n");
    return 0;
}

/**
 * Process Transaction
 */
int payment_process_transaction(uint32_t amount,
                                uint32_t merchant_id,
                                transaction_t *transaction) {
    printf("[SECURE WORLD] Processing transaction: %u cents from 0x%X...\n",
           amount, merchant_id);
    
    // Check daily limit
    if (g_payment_context.daily_spent + amount > g_payment_context.daily_limit) {
        printf("[SECURE WORLD] ✗ Exceeds daily limit\n");
        return -1;
    }
    
    // Generate transaction ID
    for (int i = 0; i < TRANSACTION_ID_SIZE; i++) {
        transaction->transaction_id[i] = (uint8_t)((time(NULL) + i) % 256);
    }
    
    transaction->amount = amount;
    transaction->merchant_id = merchant_id;
    transaction->timestamp = (uint32_t)time(NULL);
    
    // Compute transaction hash
    uint32_t hash = 0;
    for (int i = 0; i < TRANSACTION_ID_SIZE; i++) {
        hash = ((hash << 5) ^ transaction->transaction_id[i]) & 0xFFFFFFFF;
    }
    hash ^= amount;
    hash ^= merchant_id;
    
    for (int i = 0; i < 32; i++) {
        transaction->transaction_hash[i] = (uint8_t)((hash >> (i % 4)) & 0xFF);
    }
    
    // Update daily spent
    g_payment_context.daily_spent += amount;
    
    printf("[SECURE WORLD] ✓ Transaction approved\n");
    printf("[SECURE WORLD] Daily spent: %u / %u cents\n",
           g_payment_context.daily_spent,
           g_payment_context.daily_limit);
    
    return 0;
}

/**
 * Lock Card
 */
void payment_lock_card(void) {
    printf("[SECURE WORLD] Locking card due to suspicious activity...\n");
    g_payment_context.is_locked = 1;
    printf("[SECURE WORLD] ✗ Card locked\n");
}

/**
 * Get Transaction Receipt
 */
void payment_print_receipt(const transaction_t *transaction) {
    printf("\n╔══════════════════════════════════╗\n");
    printf("║      TRANSACTION RECEIPT         ║\n");
    printf("╠══════════════════════════════════╣\n");
    printf("║ Amount:     $%.2f                ║\n",
           transaction->amount / 100.0);
    printf("║ Merchant:   0x%X               ║\n",
           transaction->merchant_id);
    printf("║ Time:       %u                ║\n",
           transaction->timestamp);
    printf("║ Approved:   YES                  ║\n");
    printf("╚══════════════════════════════════╝\n");
}

/**
 * Example: Secure Payment Transaction
 */
void example_payment_transaction(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         Example: Secure Payment Transaction               ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    payment_init();
    
    // Normal World sends transaction request with PIN
    printf("\n[NORMAL WORLD] Initiating payment transaction...\n");
    printf("[NORMAL WORLD] Amount: $50.00\n");
    printf("[NORMAL WORLD] Merchant: 0x98765432\n");
    
    // PIN (simulated - in real system, obtained from user input in Normal World)
    uint8_t pin[] = {0x01, 0x02, 0x03, 0x04};
    
    printf("\n");
    
    // Verify PIN
    if (payment_verify_pin(pin, sizeof(pin)) != 0) {
        printf("[NORMAL WORLD] ✗ Transaction rejected\n");
        payment_lock_card();
        return;
    }
    
    // Process transaction
    transaction_t transaction;
    if (payment_process_transaction(5000, 0x98765432, &transaction) == 0) {
        printf("\n[NORMAL WORLD] ✓ Transaction approved\n");
        payment_print_receipt(&transaction);
    } else {
        printf("[NORMAL WORLD] ✗ Transaction declined\n");
    }
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     TrustZone Secure Payment Example                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    example_payment_transaction();
    
    return 0;
}
