/* key_manager.c */

#include "key_manager.h"
#include "secure_mem.h"

#include "mbedtls/hkdf.h"
#include "mbedtls/md.h"
#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"

#include <string.h>

/* External RNG (defined in aes_gcm.c — shared instance) */
extern mbedtls_ctr_drbg_context g_ctr_drbg;
extern bool                      g_rng_initialized;

/* ----------------------------------------------------------------
 * km_derive_key() — HKDF-SHA256
 *
 * HKDF provides:
 * 1. Extract: HMAC-SHA256(salt, IKM) → PRK (pseudo-random key)
 * 2. Expand:  HMAC-SHA256(PRK, info || counter) → OKM (output key)
 *
 * SECURITY: Even if master_key has low entropy or structure,
 * HKDF's extract step normalizes it to a full-entropy PRK.
 * The 'info' string provides cryptographic domain separation.
 * ---------------------------------------------------------------- */
aes_gcm_error_t km_derive_key(const uint8_t *master_key,
                               size_t         master_key_len,
                               const uint8_t *salt,
                               size_t         salt_len,
                               const uint8_t *info,
                               size_t         info_len,
                               uint8_t       *derived_key)
{
    if (master_key == NULL || derived_key == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (master_key_len < 16U || master_key_len > 64U) {
        return AES_GCM_ERR_SIZE;
    }
    if (info_len > KM_INFO_MAX_SIZE) {
        return AES_GCM_ERR_SIZE;
    }

    const mbedtls_md_info_t *md_info =
        mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);

    if (md_info == NULL) {
        return AES_GCM_ERR_KEY_INIT;
    }

    int ret = mbedtls_hkdf(
                md_info,
                salt,        salt_len,
                master_key,  master_key_len,
                info,        info_len,
                derived_key, KM_DERIVED_KEY_SIZE);

    if (ret != 0) {
        secure_memzero(derived_key, KM_DERIVED_KEY_SIZE);
        return AES_GCM_ERR_KEY_INIT;
    }

    return AES_GCM_OK;
}

/* ----------------------------------------------------------------
 * km_generate_master_key()
 * ---------------------------------------------------------------- */
aes_gcm_error_t km_generate_master_key(uint8_t *key_out,
                                        size_t   key_out_size)
{
    if (key_out == NULL) {
        return AES_GCM_ERR_NULL;
    }
    if (key_out_size < KM_MASTER_KEY_SIZE) {
        return AES_GCM_ERR_SIZE;
    }
    if (!g_rng_initialized) {
        return AES_GCM_ERR_RNG;
    }

    int ret = mbedtls_ctr_drbg_random(
                &g_ctr_drbg, key_out, KM_MASTER_KEY_SIZE);

    if (ret != 0) {
        secure_memzero(key_out, key_out_size);
        return AES_GCM_ERR_RNG;
    }

    return AES_GCM_OK;
}