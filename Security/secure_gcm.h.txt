/* aes_gcm.h
 *
 * AES-256-GCM AEAD (Authenticated Encryption with Associated Data)
 * Public API for embedded ARM Linux targets.
 *
 * Security Properties:
 *   - Confidentiality:  AES-256 (NIST FIPS 197)
 *   - Integrity:        GHASH (GCM authentication tag, 128-bit)
 *   - IV Uniqueness:    Enforced via monotonic counter + random
 *   - Key Freshness:    HKDF-derived per-operation subkeys (optional)
 *
 * Thread Safety: NOT thread-safe. Each thread must maintain its own
 *                aes_gcm_ctx_t instance.
 */

#ifndef AES_GCM_H
#define AES_GCM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */
#define AES_GCM_KEY_SIZE        (32U)   /* AES-256: 256 bits        */
#define AES_GCM_IV_SIZE         (12U)   /* GCM standard: 96-bit IV  */
#define AES_GCM_TAG_SIZE        (16U)   /* 128-bit authentication tag*/
#define AES_GCM_AAD_MAX_SIZE    (256U)  /* Max Additional Auth Data */
#define AES_GCM_PLAINTEXT_MAX   (1U << 24)  /* 16MB per operation   */

/*
 * SECURITY: IV size is fixed at 12 bytes (96 bits).
 * GCM with 96-bit IV uses the counter directly (no hashing),
 * which is more efficient AND avoids the GHASH collision risk
 * present with non-96-bit IV sizes. Never use other IV sizes.
 */

/* ----------------------------------------------------------------
 * Error Codes
 * ---------------------------------------------------------------- */
typedef enum {
    AES_GCM_OK              =  0,   /* Success                     */
    AES_GCM_ERR_NULL        = -1,   /* NULL pointer argument       */
    AES_GCM_ERR_SIZE        = -2,   /* Invalid buffer size         */
    AES_GCM_ERR_KEY_INIT    = -3,   /* Key schedule failed         */
    AES_GCM_ERR_IV_GEN      = -4,   /* IV generation failed        */
    AES_GCM_ERR_ENCRYPT     = -5,   /* Encryption operation failed */
    AES_GCM_ERR_DECRYPT     = -6,   /* Decryption operation failed */
    AES_GCM_ERR_AUTH        = -7,   /* Authentication tag mismatch */
    AES_GCM_ERR_IV_REUSE    = -8,   /* IV reuse detected           */
    AES_GCM_ERR_KEY_WRAP    = -9,   /* Key wrapping failed         */
    AES_GCM_ERR_RNG         = -10,  /* RNG failure                 */
    AES_GCM_ERR_OVERFLOW    = -11,  /* Counter overflow            */
} aes_gcm_error_t;

/* ----------------------------------------------------------------
 * IV State Machine
 *
 * SECURITY: IV reuse with the same key is catastrophic for GCM.
 * It allows an attacker to XOR ciphertexts to recover XOR of
 * plaintexts AND recover the authentication key (H), breaking
 * both confidentiality and integrity completely.
 *
 * Mitigation: Hybrid IV construction:
 *   [4-byte random prefix][8-byte monotonic counter]
 *
 * - Random prefix: changes on each context initialization
 * - Monotonic counter: increments on each encryption
 * - Counter overflow: context MUST be re-initialized (new key)
 *
 * This guarantees uniqueness for up to 2^64 messages per key.
 * ---------------------------------------------------------------- */
typedef struct {
    uint8_t  random_prefix[4];      /* Per-context random bytes     */
    uint64_t counter;               /* Monotonic operation counter  */
    uint64_t max_invocations;       /* Hard limit (default 2^32)    */
} aes_gcm_iv_state_t;

/* ----------------------------------------------------------------
 * AES-GCM Context
 *
 * SECURITY: Never copy this struct. It contains key schedule
 * material. Use secure_free() when done.
 * ---------------------------------------------------------------- */
typedef struct {
    /* mbedTLS GCM context (contains expanded key schedule) */
    void                   *mbedtls_gcm_ctx;    /* Opaque handle    */

    /* IV management */
    aes_gcm_iv_state_t      iv_state;

    /* Key metadata */
    uint8_t                 key_id[16];         /* Key identifier   */
    bool                    is_initialized;
    bool                    key_loaded;

    /* Operation statistics (for audit logging) */
    uint64_t                encrypt_count;
    uint64_t                decrypt_count;
    uint64_t                auth_fail_count;    /* Tag mismatches   */
} aes_gcm_ctx_t;

/* ----------------------------------------------------------------
 * Encrypted packet structure
 *
 * Wire format:
 * ┌──────────────────────────────────────────────────────┐
 * │ IV (12 bytes) │ Tag (16 bytes) │ Ciphertext (N bytes)│
 * └──────────────────────────────────────────────────────┘
 *
 * SECURITY: Tag is transmitted with ciphertext (not after).
 * Decryption MUST verify tag before returning any plaintext.
 * (Encrypt-then-MAC order; GCM enforces this internally.)
 * ---------------------------------------------------------------- */
typedef struct {
    uint8_t  iv[AES_GCM_IV_SIZE];
    uint8_t  tag[AES_GCM_TAG_SIZE];
    uint8_t *ciphertext;
    size_t   ciphertext_len;
} aes_gcm_packet_t;

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

/*
 * aes_gcm_init() — Initialize context and RNG subsystem.
 * Must be called before any other API function.
 */
aes_gcm_error_t aes_gcm_init(aes_gcm_ctx_t *ctx,
                              const uint8_t *key_id,
                              size_t         key_id_len);

/*
 * aes_gcm_load_key() — Load AES-256 key into context.
 *
 * SECURITY: Key material is copied into the mbedTLS context
 * and the caller's buffer should be zeroed after this call.
 * The key is expanded into a key schedule and the raw key
 * bytes are cleared from the mbedTLS context after setup.
 */
aes_gcm_error_t aes_gcm_load_key(aes_gcm_ctx_t *ctx,
                                  const uint8_t *key,
                                  size_t         key_len);

/*
 * aes_gcm_encrypt() — Encrypt and authenticate plaintext.
 *
 * Parameters:
 *   ctx          - Initialized context with loaded key
 *   plaintext    - Input data
 *   pt_len       - Input data length
 *   aad          - Additional Authenticated Data (may be NULL)
 *   aad_len      - AAD length (0 if aad is NULL)
 *   packet       - Output packet (IV, tag, ciphertext)
 *                  packet->ciphertext must be pre-allocated (pt_len bytes)
 *
 * SECURITY: IV is generated internally and stored in packet->iv.
 * Caller must NEVER reuse an IV externally.
 */
aes_gcm_error_t aes_gcm_encrypt(aes_gcm_ctx_t       *ctx,
                                 const uint8_t       *plaintext,
                                 size_t               pt_len,
                                 const uint8_t       *aad,
                                 size_t               aad_len,
                                 aes_gcm_packet_t    *packet);

/*
 * aes_gcm_decrypt() — Verify authentication tag and decrypt.
 *
 * SECURITY CRITICAL:
 * 1. Tag verification happens BEFORE any plaintext is output.
 * 2. If tag verification fails, output buffer is zeroed.
 * 3. Returns AES_GCM_ERR_AUTH on tag mismatch — caller must
 *    NOT use output_plaintext in this case.
 * 4. Comparison is constant-time to prevent timing oracle.
 */
aes_gcm_error_t aes_gcm_decrypt(aes_gcm_ctx_t       *ctx,
                                 const aes_gcm_packet_t *packet,
                                 const uint8_t        *aad,
                                 size_t                aad_len,
                                 uint8_t              *output_plaintext,
                                 size_t                output_buf_size,
                                 size_t               *output_len);

/*
 * aes_gcm_destroy() — Securely wipe and free context.
 * MUST be called when context is no longer needed.
 */
void aes_gcm_destroy(aes_gcm_ctx_t *ctx);

/*
 * aes_gcm_strerror() — Human-readable error string.
 */
const char *aes_gcm_strerror(aes_gcm_error_t err);

#ifdef __cplusplus
}
#endif

#endif /* AES_GCM_H */