/* main.c — Production integration example
 *
 * Demonstrates secure encrypt/decrypt cycle with:
 * - Key derivation via HKDF
 * - AAD (Additional Authenticated Data) for context binding
 * - Proper error handling and key zeroization
 * - Packet serialization format
 */

#include "aes_gcm.h"
#include "key_manager.h"
#include "secure_mem.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ----------------------------------------------------------------
 * Utility: Print hex buffer (debug only — strip from production)
 * ---------------------------------------------------------------- */
#ifdef DEBUG_BUILD
static void print_hex(const char *label,
                      const uint8_t *buf, size_t len)
{
    printf("%-20s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02X", buf[i]);
        if ((i + 1) % 16 == 0 && i + 1 < len) printf("\n%22s", "");
    }
    printf("\n");
}
#else
#define print_hex(label, buf, len)  /* stripped in production */
#endif

/* ----------------------------------------------------------------
 * Utility: Serialize packet to wire format
 *
 * Wire format:
 * [IV:12][TAG:16][LEN:4 big-endian][CIPHERTEXT:LEN]
 * ---------------------------------------------------------------- */
static int serialize_packet(const aes_gcm_packet_t *pkt,
                             uint8_t *out_buf, size_t out_size,
                             size_t *out_len)
{
    size_t required = AES_GCM_IV_SIZE + AES_GCM_TAG_SIZE
                    + 4U + pkt->ciphertext_len;

    if (out_size < required) return -1;

    size_t offset = 0;

    memcpy(out_buf + offset, pkt->iv, AES_GCM_IV_SIZE);
    offset += AES_GCM_IV_SIZE;

    memcpy(out_buf + offset, pkt->tag, AES_GCM_TAG_SIZE);
    offset += AES_GCM_TAG_SIZE;

    /* Length in big-endian */
    uint32_t len_be = (uint32_t)pkt->ciphertext_len;
    out_buf[offset + 0] = (uint8_t)(len_be >> 24U);
    out_buf[offset + 1] = (uint8_t)(len_be >> 16U);
    out_buf[offset + 2] = (uint8_t)(len_be >>  8U);
    out_buf[offset + 3] = (uint8_t)(len_be        );
    offset += 4U;

    memcpy(out_buf + offset, pkt->ciphertext, pkt->ciphertext_len);
    offset += pkt->ciphertext_len;

    *out_len = offset;
    return 0;
}

/* ----------------------------------------------------------------
 * main()
 * ---------------------------------------------------------------- */
int main(void)
{
    aes_gcm_error_t  err;
    int              exit_code = EXIT_FAILURE;

    /* ---- Secure allocations ---- */
    uint8_t *master_key    = (uint8_t *)secure_alloc(KM_MASTER_KEY_SIZE);
    uint8_t *derived_key   = (uint8_t *)secure_alloc(AES_GCM_KEY_SIZE);
    uint8_t *ciphertext    = NULL;
    uint8_t *plaintext_out = NULL;

    if (master_key == NULL || derived_key == NULL) {
        fprintf(stderr, "[ERROR] Secure allocation failed\n");
        goto cleanup;
    }

    /* ================================================================
     * STEP 1: Key Generation and Derivation
     * ================================================================ */

    /* Generate random 256-bit master key */
    err = km_generate_master_key(master_key, KM_MASTER_KEY_SIZE);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Key generation: %s\n",
                aes_gcm_strerror(err));
        goto cleanup;
    }
    print_hex("Master Key", master_key, KM_MASTER_KEY_SIZE);

    /*
     * Derive AES-256 subkey via HKDF-SHA256.
     *
     * SECURITY: The 'info' string provides domain separation.
     * Use unique strings for different purposes:
     *   "firmware-update-encryption-v1"
     *   "secure-storage-v1"
     *   "session-key-v1"
     * This ensures keys derived for different purposes are
     * cryptographically independent, even from the same master key.
     */
    const uint8_t salt[] = "arm-embedded-aes-gcm-salt-2024";
    const uint8_t info[] = "firmware-data-encryption-v1";

    err = km_derive_key(master_key,    KM_MASTER_KEY_SIZE,
                        salt,          sizeof(salt) - 1U,
                        info,          sizeof(info) - 1U,
                        derived_key);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Key derivation: %s\n",
                aes_gcm_strerror(err));
        goto cleanup;
    }
    print_hex("Derived Key", derived_key, AES_GCM_KEY_SIZE);

    /*
     * SECURITY: Zero master key immediately after derivation.
     * The derived key is what we actually use for encryption.
     * Keeping the master key in memory longer than necessary
     * increases exposure window for memory scraping attacks.
     */
    secure_memzero(master_key, KM_MASTER_KEY_SIZE);
    printf("[INFO] Master key zeroized after derivation\n");

    /* ================================================================
     * STEP 2: Initialize AES-GCM Context
     * ================================================================ */

    aes_gcm_ctx_t ctx;
    const uint8_t key_id[] = "fw-enc-key-2024-001";

    err = aes_gcm_init(&ctx, key_id, sizeof(key_id) - 1U);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Context init: %s\n",
                aes_gcm_strerror(err));
        goto cleanup;
    }

    err = aes_gcm_load_key(&ctx, derived_key, AES_GCM_KEY_SIZE);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Key load: %s\n",
                aes_gcm_strerror(err));
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    /*
     * SECURITY: Zero derived key after loading into context.
     * Key schedule is now inside the secure GCM context.
     */
    secure_memzero(derived_key, AES_GCM_KEY_SIZE);
    printf("[INFO] Derived key zeroized after context load\n");

    /* ================================================================
     * STEP 3: Encrypt with AAD
     * ================================================================ */

    const char plaintext[] =
        "CONFIDENTIAL: ARM TrustZone firmware update package v2.1.0";

    /*
     * Additional Authenticated Data (AAD) — NOT encrypted but IS
     * authenticated. Use for headers, packet metadata, device IDs.
     *
     * SECURITY: AAD binds the ciphertext to its context.
     * An attacker cannot replay a ciphertext in a different
     * context (different device, different command) without
     * tag verification failing — even with the correct key.
     */
    const uint8_t aad[] = "DEVICE:ARM-CORTEX-A55:SLOT:0:VERSION:2";

    size_t pt_len = strlen(plaintext);
    ciphertext = (uint8_t *)secure_alloc(pt_len);
    if (ciphertext == NULL) {
        fprintf(stderr, "[ERROR] Ciphertext allocation failed\n");
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    aes_gcm_packet_t packet = {
        .ciphertext     = ciphertext,
        .ciphertext_len = 0U,
    };

    printf("\n[ENCRYPT] Plaintext (%zu bytes): %s\n", pt_len, plaintext);

    err = aes_gcm_encrypt(&ctx,
                          (const uint8_t *)plaintext, pt_len,
                          aad, sizeof(aad) - 1U,
                          &packet);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Encryption: %s\n",
                aes_gcm_strerror(err));
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    print_hex("IV (12B)",         packet.iv,         AES_GCM_IV_SIZE);
    print_hex("Auth Tag (16B)",   packet.tag,        AES_GCM_TAG_SIZE);
    print_hex("Ciphertext",       packet.ciphertext, packet.ciphertext_len);
    printf("[INFO] Encrypt count: %llu\n",
           (unsigned long long)ctx.encrypt_count);

    /* Serialize to wire format */
    uint8_t wire_buf[1024];
    size_t  wire_len = 0U;
    if (serialize_packet(&packet, wire_buf,
                         sizeof(wire_buf), &wire_len) != 0) {
        fprintf(stderr, "[ERROR] Serialization failed\n");
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }
    printf("[INFO] Wire packet: %zu bytes\n", wire_len);

    /* ================================================================
     * STEP 4: Decrypt and Verify
     * ================================================================ */

    plaintext_out = (uint8_t *)secure_alloc(pt_len + 1U);
    if (plaintext_out == NULL) {
        fprintf(stderr, "[ERROR] Output allocation failed\n");
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    size_t output_len = 0U;

    err = aes_gcm_decrypt(&ctx,
                          &packet,
                          aad, sizeof(aad) - 1U,
                          plaintext_out, pt_len + 1U,
                          &output_len);
    if (err != AES_GCM_OK) {
        fprintf(stderr, "[ERROR] Decryption: %s\n",
                aes_gcm_strerror(err));
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    plaintext_out[output_len] = '\0';
    printf("\n[DECRYPT] Recovered (%zu bytes): %s\n",
           output_len, plaintext_out);

    /* ================================================================
     * STEP 5: Tamper Detection Test
     * ================================================================ */
    printf("\n[TEST] Tamper detection...\n");

    /* Flip one bit in ciphertext */
    uint8_t tampered_ct[256];
    memcpy(tampered_ct, packet.ciphertext, packet.ciphertext_len);
    tampered_ct[0] ^= 0x01U;    /* Single bit flip */

    aes_gcm_packet_t tampered_packet = packet;
    tampered_packet.ciphertext = tampered_ct;

    uint8_t tampered_out[256] = {0};
    size_t  tampered_out_len  = 0U;

    err = aes_gcm_decrypt(&ctx,
                          &tampered_packet,
                          aad, sizeof(aad) - 1U,
                          tampered_out, sizeof(tampered_out),
                          &tampered_out_len);

    if (err == AES_GCM_ERR_AUTH) {
        printf("[PASS] Tampered ciphertext correctly REJECTED\n");
        printf("[INFO] Auth fail count: %llu\n",
               (unsigned long long)ctx.auth_fail_count);
    } else {
        fprintf(stderr, "[FAIL] Tamper detection FAILED!\n");
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    /* ================================================================
     * STEP 6: Verify plaintext integrity
     * ================================================================ */
    if (output_len != pt_len ||
        secure_memcmp(plaintext_out,
                      (const uint8_t *)plaintext,
                      pt_len) != 0) {
        fprintf(stderr, "[FAIL] Plaintext mismatch!\n");
        aes_gcm_destroy(&ctx);
        goto cleanup;
    }

    printf("\n[PASS] All security checks passed\n");
    printf("[INFO] Total encrypt: %llu  decrypt: %llu\n",
           (unsigned long long)ctx.encrypt_count,
           (unsigned long long)ctx.decrypt_count);

    exit_code = EXIT_SUCCESS;

    /* ================================================================
     * CLEANUP — Always executed
     * ================================================================ */
    aes_gcm_destroy(&ctx);

cleanup:
    if (master_key)    secure_free(master_key,    KM_MASTER_KEY_SIZE);
    if (derived_key)   secure_free(derived_key,   AES_GCM_KEY_SIZE);
    if (ciphertext)    secure_free(ciphertext,     pt_len);
    if (plaintext_out) secure_free(plaintext_out,  pt_len + 1U);

    return exit_code;
}