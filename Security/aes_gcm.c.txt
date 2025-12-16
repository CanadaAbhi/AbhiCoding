/* aes_gcm.c
 *
 * AES-256-GCM implementation using mbedTLS 3.x
 *
 * Compile: see Makefile
 * Target:  ARM Cortex-A Linux (aarch64 / armv7l)
 */

#define _GNU_SOURCE
#include "aes_gcm.h"
#include "secure_mem.h"

/* mbedTLS headers */
#include "mbedtls/gcm.h"
#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"
#include "mbedtls/error.h"
#include "mbedtls/platform.h"
#include "mbedtls/constant_time.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

/* ----------------------------------------------------------------
 * Internal RNG state
 * SECURITY: Single DRBG instance per process. The CTR_DRBG is
 * seeded from /dev/urandom via mbedTLS entropy module.
 * Reseeding happens automatically every 10,000 calls (default).
 * ---------------------------------------------------------------- */
static mbedtls_entropy_context  g_entropy;
static mbedtls_ctr_drbg_context g_ctr_drbg;
static bool                      g_rng_initialized = false;

/* Personalization string for DRBG — unique per application */
static const uint8_t g_drbg_pers[] =
    "aes_gcm_arm_embedded_v1.0_" __DATE__ "__" __TIME__;

/* ----------------------------------------------------------------
 * Internal: Initialize global RNG (call once at startup)
 * ---------------------------------------------------------------- */
static aes_gcm_error_t rng_init_once(void)
{
    int ret;

    if (g_rng_initialized) {
        return AES_GCM_OK;
    }

    mbedtls_entropy_init(&g_entropy);
    mbedtls_ctr_drbg_init(&g_ctr_drbg);

    /*
     * SECURITY: Seed CTR-DRBG from /dev/urandom.
     * The personalization string provides domain separation —
     * even if two processes share the same entropy source state,
     * their DRBG outputs will differ.
     */
    ret = mbedtls_ctr_drbg_seed(
            &g_ctr_drbg,
            mbedtls_entropy_func,
            &g_entropy,
            g_drbg_pers,
            sizeof(g_drbg_pers) - 1U);

    if (ret != 0) {
        mbedtls_ctr_drbg_free(&g_ctr_drbg);
        mbedtls_entropy_free(&g_entropy);
        return AES_GCM_ERR_RNG;
    }

    /*
     * SECURITY: Enable prediction resistance mode.
     * Forces a reseed from /dev/urandom before EVERY random
     * generation call. Slower but prevents state compromise
     * attacks on the DRBG internal state.
     *
     * For high-throughput use cases, remove this and rely on
     * automatic reseeding every 10,000 calls.
     */
    mbedtls_ctr_drbg_set_prediction_resistance(
        &g_ctr_drbg,
        MBEDTLS_CTR_DRBG_PR_ON);

    g_rng_initialized = true;
    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * Internal: Generate cryptographically secure random bytes
 * ---------------------------------------------------------------- */
static aes_gcm_error_t rng_generate(uint8_t *buf, size_t len)
{
    aes_gcm_error_t init_err = rng_init_once();
    if (init_err != AES_GCM_OK) {
        return init_err;
    }

    int ret = mbedtls_ctr_drbg_random(&g_ctr_drbg, buf, len);
    if (ret != 0) {
        secure_memzero(buf, len);   /* Don't expose partial output */
        return AES_GCM_ERR_RNG;
    }

    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * Internal: Build GCM IV from IV state
 *
 * IV Layout (12 bytes / 96 bits):
 * ┌────────────────┬────────────────────────────────────┐
 * │ Random (4B)    │ Monotonic Counter (8B, big-endian) │
 * └────────────────┴────────────────────────────────────┘
 *
 * SECURITY:
 * - Random prefix ensures IVs are unique across context instances
 *   even if counter state is reset (e.g., power cycle without
 *   persistent counter storage).
 * - Monotonic counter guarantees within-session uniqueness.
 * - Big-endian counter is more natural for GCM counter mode
 *   and easier to audit in packet captures.
 * ---------------------------------------------------------------- */
static aes_gcm_error_t build_iv(aes_gcm_iv_state_t *state,
                                 uint8_t            *iv_out)
{
    /* Check counter overflow before incrementing */
    if (state->counter >= state->max_invocations) {
        return AES_GCM_ERR_OVERFLOW;
    }

    /* Copy 4-byte random prefix */
    memcpy(iv_out, state->random_prefix, 4U);

    /* Encode 8-byte counter in big-endian */
    uint64_t ctr = state->counter;
    iv_out[4]  = (uint8_t)((ctr >> 56U) & 0xFFU);
    iv_out[5]  = (uint8_t)((ctr >> 48U) & 0xFFU);
    iv_out[6]  = (uint8_t)((ctr >> 40U) & 0xFFU);
    iv_out[7]  = (uint8_t)((ctr >> 32U) & 0xFFU);
    iv_out[8]  = (uint8_t)((ctr >> 24U) & 0xFFU);
    iv_out[9]  = (uint8_t)((ctr >> 16U) & 0xFFU);
    iv_out[10] = (uint8_t)((ctr >>  8U) & 0xFFU);
    iv_out[11] = (uint8_t)((ctr        ) & 0xFFU);

    /* Increment AFTER successful IV construction */
    state->counter++;

    return AES_GCM_OK;
}

/* ================================================================
 * Public API Implementation
 * ================================================================ */

/* ----------------------------------------------------------------
 * aes_gcm_init()
 * ---------------------------------------------------------------- */
aes_gcm_error_t aes_gcm_init(aes_gcm_ctx_t *ctx,
                              const uint8_t *key_id,
                              size_t         key_id_len)
{
    if (ctx == NULL) {
        return AES_GCM_ERR_NULL;
    }

    /* Zero-initialize entire context */
    secure_memzero(ctx, sizeof(aes_gcm_ctx_t));

    /* Allocate mbedTLS GCM context in secure memory */
    ctx->mbedtls_gcm_ctx = secure_alloc(sizeof(mbedtls_gcm_context));
    if (ctx->mbedtls_gcm_ctx == NULL) {
        return AES_GCM_ERR_KEY_INIT;
    }

    mbedtls_gcm_init((mbedtls_gcm_context *)ctx->mbedtls_gcm_ctx);

    /* Initialize IV state */
    aes_gcm_error_t rng_err =
        rng_generate(ctx->iv_state.random_prefix, 4U);
    if (rng_err != AES_GCM_OK) {
        secure_free(ctx->mbedtls_gcm_ctx,
                    sizeof(mbedtls_gcm_context));
        ctx->mbedtls_gcm_ctx = NULL;
        return rng_err;
    }

    ctx->iv_state.counter          = 0ULL;
    /*
     * SECURITY: Limit to 2^32 encryptions per key per context.
     * GCM is secure for far more invocations per key, but this
     * limit encourages key rotation and catches counter bugs.
     * Adjust based on your threat model.
     */
    ctx->iv_state.max_invocations  = (1ULL << 32);

    /* Store key identifier (for audit logs, not crypto) */
    if (key_id != NULL && key_id_len > 0U) {
        size_t copy_len = (key_id_len < sizeof(ctx->key_id))
                        ? key_id_len : sizeof(ctx->key_id);
        memcpy(ctx->key_id, key_id, copy_len);
    }

    ctx->is_initialized = true;
    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * aes_gcm_load_key()
 * ---------------------------------------------------------------- */
aes_gcm_error_t aes_gcm_load_key(aes_gcm_ctx_t *ctx,
                                  const uint8_t *key,
                                  size_t         key_len)
{
    if (ctx == NULL || key == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (!ctx->is_initialized) {
        return AES_GCM_ERR_KEY_INIT;
    }
    if (key_len != AES_GCM_KEY_SIZE) {
        /*
         * SECURITY: Reject non-256-bit keys.
         * AES-128 and AES-192 are acceptable but this
         * implementation is hardened for AES-256 only.
         */
        return AES_GCM_ERR_SIZE;
    }

    mbedtls_gcm_context *gcm =
        (mbedtls_gcm_context *)ctx->mbedtls_gcm_ctx;

    /*
     * SECURITY: mbedtls_gcm_setkey() expands the raw key into
     * a key schedule (10/12/14 rounds for AES-128/192/256).
     * After this call, the key schedule is stored in the GCM
     * context and the raw key bytes are not retained by mbedTLS.
     *
     * Key length is specified in BITS per mbedTLS API.
     */
    int ret = mbedtls_gcm_setkey(gcm,
                                  MBEDTLS_CIPHER_ID_AES,
                                  key,
                                  (unsigned int)(key_len * 8U));
    if (ret != 0) {
        return AES_GCM_ERR_KEY_INIT;
    }

    ctx->key_loaded = true;

    /*
     * SECURITY REMINDER (not enforceable here):
     * The caller SHOULD zero their key buffer after this call:
     *   secure_memzero(key_buf, key_len);
     * See main.c for correct usage pattern.
     */
    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * aes_gcm_encrypt()
 * ---------------------------------------------------------------- */
aes_gcm_error_t aes_gcm_encrypt(aes_gcm_ctx_t       *ctx,
                                 const uint8_t       *plaintext,
                                 size_t               pt_len,
                                 const uint8_t       *aad,
                                 size_t               aad_len,
                                 aes_gcm_packet_t    *packet)
{
    /* ---- Input validation ---- */
    if (ctx == NULL || packet == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (plaintext == NULL && pt_len > 0U) {
        return AES_GCM_ERR_NULL;
    }
    if (aad == NULL && aad_len > 0U) {
        return AES_GCM_ERR_NULL;
    }
    if (pt_len > AES_GCM_PLAINTEXT_MAX) {
        return AES_GCM_ERR_SIZE;
    }
    if (aad_len > AES_GCM_AAD_MAX_SIZE) {
        return AES_GCM_ERR_SIZE;
    }
    if (packet->ciphertext == NULL && pt_len > 0U) {
        return AES_GCM_ERR_NULL;
    }
    if (!ctx->is_initialized || !ctx->key_loaded) {
        return AES_GCM_ERR_KEY_INIT;
    }

    /* ---- Generate IV ---- */
    aes_gcm_error_t iv_err = build_iv(&ctx->iv_state, packet->iv);
    if (iv_err != AES_GCM_OK) {
        return iv_err;
    }

    /* ---- Perform GCM encryption ---- */
    mbedtls_gcm_context *gcm =
        (mbedtls_gcm_context *)ctx->mbedtls_gcm_ctx;

    /*
     * mbedtls_gcm_crypt_and_tag():
     *   - Encrypts plaintext → ciphertext (AES-CTR mode internally)
     *   - Computes GHASH over (AAD || ciphertext) → tag
     *   - Single-call API ensures atomicity of encrypt+authenticate
     *
     * SECURITY: GCM mode — the same AES key drives both the
     * stream cipher (CTR) and the authenticator (GHASH via H = E(K,0)).
     * This is safe because CTR and GHASH use key in different ways.
     */
    int ret = mbedtls_gcm_crypt_and_tag(
                gcm,
                MBEDTLS_GCM_ENCRYPT,
                pt_len,
                packet->iv,  AES_GCM_IV_SIZE,
                aad,         aad_len,
                plaintext,
                packet->ciphertext,
                AES_GCM_TAG_SIZE,
                packet->tag);

    if (ret != 0) {
        /*
         * SECURITY: On any failure, zero output buffers
         * to prevent partial/unauthenticated data leakage.
         */
        secure_memzero(packet->ciphertext, pt_len);
        secure_memzero(packet->tag,        AES_GCM_TAG_SIZE);
        secure_memzero(packet->iv,         AES_GCM_IV_SIZE);
        return AES_GCM_ERR_ENCRYPT;
    }

    packet->ciphertext_len = pt_len;
    ctx->encrypt_count++;

    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * aes_gcm_decrypt()
 * ---------------------------------------------------------------- */
aes_gcm_error_t aes_gcm_decrypt(aes_gcm_ctx_t          *ctx,
                                 const aes_gcm_packet_t *packet,
                                 const uint8_t          *aad,
                                 size_t                  aad_len,
                                 uint8_t                *output_plaintext,
                                 size_t                  output_buf_size,
                                 size_t                 *output_len)
{
    /* ---- Input validation ---- */
    if (ctx == NULL || packet == NULL || output_plaintext == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (output_len == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (packet->ciphertext == NULL && packet->ciphertext_len > 0U) {
        return AES_GCM_ERR_NULL;
    }
    if (aad == NULL && aad_len > 0U) {
        return AES_GCM_ERR_NULL;
    }
    if (packet->ciphertext_len > AES_GCM_PLAINTEXT_MAX) {
        return AES_GCM_ERR_SIZE;
    }
    if (output_buf_size < packet->ciphertext_len) {
        return AES_GCM_ERR_SIZE;
    }
    if (!ctx->is_initialized || !ctx->key_loaded) {
        return AES_GCM_ERR_KEY_INIT;
    }

    *output_len = 0U;

    /* ---- Decrypt and verify authentication tag ---- */
    mbedtls_gcm_context *gcm =
        (mbedtls_gcm_context *)ctx->mbedtls_gcm_ctx;

    /*
     * SECURITY CRITICAL — Tag verification design:
     *
     * mbedtls_gcm_auth_decrypt() performs:
     * 1. Decrypt ciphertext → temp buffer (CTR mode)
     * 2. Compute expected tag over (IV, AAD, ciphertext)
     * 3. Compare computed tag vs packet->tag (constant-time)
     * 4. If mismatch: zero output buffer, return error
     * 5. If match: copy temp buffer to output
     *
     * This is FUNDAMENTALLY DIFFERENT from:
     *   decrypt first → verify tag second  (WRONG — timing oracle!)
     *
     * mbedTLS implements this correctly internally. However, we
     * additionally zero the output on any error as defense-in-depth.
     *
     * The tag comparison MUST be constant-time. mbedtls uses
     * mbedtls_ct_memcmp() internally since mbedTLS 3.0.
     */
    int ret = mbedtls_gcm_auth_decrypt(
                gcm,
                packet->ciphertext_len,
                packet->iv,  AES_GCM_IV_SIZE,
                aad,         aad_len,
                packet->tag, AES_GCM_TAG_SIZE,
                packet->ciphertext,
                output_plaintext);

    if (ret == MBEDTLS_ERR_GCM_AUTH_FAILED) {
        /*
         * SECURITY: Tag authentication FAILED.
         * This means the ciphertext was tampered with, the wrong
         * key was used, the AAD was modified, or the IV was wrong.
         *
         * MANDATORY: Zero output buffer before returning error.
         * Never let caller process unauthenticated plaintext.
         */
        secure_memzero(output_plaintext, packet->ciphertext_len);
        ctx->auth_fail_count++;

        /*
         * SECURITY: Do NOT return detailed error information.
         * In a network context, detailed errors enable padding
         * oracle and chosen-ciphertext attacks. Always return
         * the same generic auth-failure code.
         */
        return AES_GCM_ERR_AUTH;

    } else if (ret != 0) {
        secure_memzero(output_plaintext, packet->ciphertext_len);
        return AES_GCM_ERR_DECRYPT;
    }

    *output_len = packet->ciphertext_len;
    ctx->decrypt_count++;

    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * aes_gcm_destroy()
 * ---------------------------------------------------------------- */
void aes_gcm_destroy(aes_gcm_ctx_t *ctx)
{
    if (ctx == NULL) {
        return;
    }

    if (ctx->mbedtls_gcm_ctx != NULL) {
        /*
         * SECURITY: mbedtls_gcm_free() zeros the key schedule
         * inside the mbedTLS context structure.
         */
        mbedtls_gcm_free((mbedtls_gcm_context *)ctx->mbedtls_gcm_ctx);
        secure_free(ctx->mbedtls_gcm_ctx, sizeof(mbedtls_gcm_context));
        ctx->mbedtls_gcm_ctx = NULL;
    }

    /*
     * SECURITY: Zero the entire context struct including:
     *   - IV state (random prefix + counter)
     *   - Key identifier
     *   - Operation counters (audit data)
     */
    secure_memzero(ctx, sizeof(aes_gcm_ctx_t));
}

/* ----------------------------------------------------------------
 * aes_gcm_strerror()
 * ---------------------------------------------------------------- */
const char *aes_gcm_strerror(aes_gcm_error_t err)
{
    switch (err) {
        case AES_GCM_OK:            return "Success";
        case AES_GCM_ERR_NULL:      return "NULL pointer argument";
        case AES_GCM_ERR_SIZE:      return "Invalid buffer size";
        case AES_GCM_ERR_KEY_INIT:  return "Key initialization failed";
        case AES_GCM_ERR_IV_GEN:    return "IV generation failed";
        case AES_GCM_ERR_ENCRYPT:   return "Encryption failed";
        case AES_GCM_ERR_DECRYPT:   return "Decryption failed";
        case AES_GCM_ERR_AUTH:      return "Authentication tag mismatch";
        case AES_GCM_ERR_IV_REUSE:  return "IV reuse detected";
        case AES_GCM_ERR_KEY_WRAP:  return "Key wrapping failed";
        case AES_GCM_ERR_RNG:       return "RNG failure";
        case AES_GCM_ERR_OVERFLOW:  return "Counter overflow - re-key required";
        default:                    return "Unknown error";
    }
}