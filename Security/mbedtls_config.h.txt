/* mbedtls_config.h
 *
 * SECURITY: Explicitly enable ONLY required modules.
 * Attack surface reduction through minimal feature set.
 * Disabling unused modules prevents supply-chain vulnerabilities
 * from pulling in untested code paths.
 */

#ifndef MBEDTLS_CONFIG_H
#define MBEDTLS_CONFIG_H

/* ----------------------------------------------------------------
 * Core System
 * ---------------------------------------------------------------- */
#define MBEDTLS_PLATFORM_C              /* Platform abstraction     */
#define MBEDTLS_PLATFORM_MEMORY         /* Custom allocator support */

/* ----------------------------------------------------------------
 * Entropy and RNG — CRITICAL SECURITY COMPONENTS
 * On ARM Linux: feeds from /dev/urandom (CSPRNG)
 * SECURITY: Never use MBEDTLS_NO_PLATFORM_ENTROPY in production
 * ---------------------------------------------------------------- */
#define MBEDTLS_ENTROPY_C
#define MBEDTLS_CTR_DRBG_C              /* NIST SP 800-90A DRBG    */
#define MBEDTLS_ENTROPY_HARDWARE_ALT    /* Hook for ARM TRNG if available */

/* ----------------------------------------------------------------
 * AES Engine
 * SECURITY: Enable hardware acceleration on ARMv8-A
 * Prevents timing attacks inherent in software AES T-table impl.
 * ---------------------------------------------------------------- */
#define MBEDTLS_AES_C
#define MBEDTLS_AESNI_C                 /* x86 AES-NI (fallback)   */
#define MBEDTLS_AESCE_C                 /* ARMv8 CE (primary)       */

/*
 * SECURITY: Do NOT define MBEDTLS_AES_ROM_TABLES — runtime tables
 * in RAM are harder to attack via cache-timing than ROM tables
 * that share cache lines with other processes.
 */

/* ----------------------------------------------------------------
 * GCM Mode
 * SECURITY: GCM provides authenticated encryption (AEAD).
 * Prevents padding oracle attacks impossible with CBC.
 * Authentication tag detects any ciphertext tampering.
 * ---------------------------------------------------------------- */
#define MBEDTLS_GCM_C
#define MBEDTLS_CIPHER_C

/* ----------------------------------------------------------------
 * SHA-256 — for key derivation (HKDF)
 * ---------------------------------------------------------------- */
#define MBEDTLS_SHA256_C
#define MBEDTLS_MD_C
#define MBEDTLS_HKDF_C                  /* RFC 5869 key derivation  */

/* ----------------------------------------------------------------
 * SECURITY HARDENING — DO NOT DISABLE
 * ---------------------------------------------------------------- */
#define MBEDTLS_ERROR_C                 /* Human-readable errors    */

/*
 * MBEDTLS_CHECK_PARAMS: Validate all function arguments.
 * Prevents exploitable undefined behavior from NULL/bad inputs.
 */
#define MBEDTLS_CHECK_PARAMS

/*
 * Timing-attack resistance for comparison operations.
 */
/* Note: Use mbedtls_ct_memcmp() from constant_time module */

/* ----------------------------------------------------------------
 * EXPLICITLY DISABLED (Security Reasons)
 * ---------------------------------------------------------------- */
/* #define MBEDTLS_DES_C       -- Weak cipher, 56-bit key         */
/* #define MBEDTLS_RC4_C       -- Broken stream cipher            */
/* #define MBEDTLS_MD5_C       -- Cryptographically broken        */
/* #define MBEDTLS_SHA1_C      -- Collision vulnerable            */
/* #define MBEDTLS_ECB_C       -- Mode leaks block patterns       */
/* #define MBEDTLS_CBC_C       -- Padding oracle risk             */
/* #define MBEDTLS_CFB_C       -- Not needed                      */
/* #define MBEDTLS_TIMING_C    -- Not needed (no TLS)             */
/* #define MBEDTLS_NET_SOCKETS -- Not needed                      */
/* #define MBEDTLS_DEBUG_C     -- NEVER in production firmware    */

/* ----------------------------------------------------------------
 * Size optimizations for embedded targets
 * ---------------------------------------------------------------- */
#define MBEDTLS_AES_FEWER_TABLES        /* Reduce ROM footprint     */

#include "mbedtls/check_config.h"

#endif /* MBEDTLS_CONFIG_H */