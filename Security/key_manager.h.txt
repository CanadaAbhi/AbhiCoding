/* key_manager.h */

#ifndef KEY_MANAGER_H
#define KEY_MANAGER_H

#include <stddef.h>
#include <stdint.h>
#include "aes_gcm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KM_MASTER_KEY_SIZE      (32U)   /* 256-bit master key       */
#define KM_DERIVED_KEY_SIZE     (32U)   /* 256-bit derived key      */
#define KM_SALT_SIZE            (32U)   /* HKDF salt size           */
#define KM_INFO_MAX_SIZE        (64U)   /* HKDF info string max     */

/*
 * km_derive_key() — HKDF-SHA256 key derivation (RFC 5869)
 *
 * Derives an AES-256 subkey from a master key using HKDF.
 * This provides domain separation between different uses of
 * the same master key (e.g., different data types or sessions).
 *
 * Parameters:
 *   master_key  - Input keying material (IKM)
 *   salt        - Random salt (use NULL for default salt)
 *   info        - Context/application string (domain separator)
 *   derived_key - Output 32-byte derived key
 */
aes_gcm_error_t km_derive_key(const uint8_t *master_key,
                               size_t         master_key_len,
                               const uint8_t *salt,
                               size_t         salt_len,
                               const uint8_t *info,
                               size_t         info_len,
                               uint8_t       *derived_key);

/*
 * km_generate_master_key() — Generate random 256-bit master key.
 * Store securely (HSM, secure storage, or key wrapping).
 */
aes_gcm_error_t km_generate_master_key(uint8_t *key_out,
                                        size_t   key_out_size);

#ifdef __cplusplus
}
#endif

#endif /* KEY_MANAGER_H */