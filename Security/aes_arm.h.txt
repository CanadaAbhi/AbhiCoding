/**
 * aes256_arm.h - Lightweight AES-256 for ARM embedded systems
 *
 * Target  : ARMv7-M / ARMv8-M / ARMv8-A (Cortex-M4/M7/M33/A53+)
 * Mode    : ECB core + CTR streaming cipher
 * RAM     : ~560 bytes stack worst-case, zero heap
 * Compiler: GCC/Clang with -march=armv8-a+crypto or -march=armv7-m
 */

#ifndef AES256_ARM_H
#define AES256_ARM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ */
/*  Build-time capability detection                                    */
/* ------------------------------------------------------------------ */
#if defined(__ARM_FEATURE_CRYPTO) && defined(__aarch64__)
#  define AES_HW_AARCH64   1          /* ARMv8-A 64-bit crypto ext    */
#elif defined(__ARM_FEATURE_CRYPTO) && defined(__arm__)
#  define AES_HW_ARMV8M    1          /* ARMv8-M / Cortex-M33+        */
#else
#  define AES_SW_FALLBACK  1          /* Pure-C for Cortex-M4/M7      */
#endif

/* ------------------------------------------------------------------ */
/*  Benchmark hooks — cycle counter                                    */
/* ------------------------------------------------------------------ */
#if defined(__aarch64__)
#  define BENCH_READ_CYCLES(dst)                                        \
     do {                                                               \
         uint64_t _c;                                                   \
         __asm__ volatile("mrs %0, cntvct_el0" : "=r"(_c));            \
         (dst) = _c;                                                    \
     } while(0)

#elif defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__)
   /* Cortex-M DWT cycle counter — enable once in system init         */
#  define BENCH_DWT_ENABLE()                                            \
     do {                                                               \
         CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;               \
         DWT->CYCCNT       = 0;                                         \
         DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;                   \
     } while(0)
#  define BENCH_READ_CYCLES(dst)   ((dst) = DWT->CYCCNT)

#else
#  define BENCH_DWT_ENABLE()       do {} while(0)
#  define BENCH_READ_CYCLES(dst)   ((dst) = 0U)
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */
#define AES256_KEY_SIZE    32U   /* bytes                              */
#define AES256_BLOCK_SIZE  16U   /* bytes                              */
#define AES256_ROUNDS      14U   /* Nr for AES-256                     */
#define AES256_CTR_IV_SIZE 12U   /* 96-bit nonce for CTR              */

/* ------------------------------------------------------------------ */
/*  Context — lives on caller's stack, zero heap                      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t rk[60];             /* Round keys  (240 bytes)           */
    uint8_t  iv[AES256_CTR_IV_SIZE]; /* CTR nonce                     */
    uint32_t ctr;                /* 32-bit counter (big-endian in CTR)*/
    bool     hw_accel;           /* true → HW path chosen at init     */
} aes256_ctx_t;                  /* Total: ~257 bytes on stack         */

/* ------------------------------------------------------------------ */
/*  Return codes                                                       */
/* ------------------------------------------------------------------ */
typedef enum {
    AES256_OK            =  0,
    AES256_ERR_NULL_PTR  = -1,
    AES256_ERR_LENGTH    = -2,
    AES256_ERR_CTR_WRAP  = -3,   /* Counter overflow — must rekey     */
} aes256_err_t;

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * aes256_init - Expand key, detect HW acceleration, bind nonce.
 *
 * @ctx    Caller-allocated context (stack or BSS, never heap)
 * @key    32-byte AES-256 key
 * @iv     12-byte nonce for CTR mode (must be unique per key)
 *
 * Stack cost: ~64 bytes (key schedule temporaries, released on return)
 */
aes256_err_t aes256_init(aes256_ctx_t       *ctx,
                         const uint8_t       key[AES256_KEY_SIZE],
                         const uint8_t       iv[AES256_CTR_IV_SIZE]);

/**
 * aes256_ctr_crypt - Encrypt or decrypt a buffer (CTR is symmetric).
 *
 * @ctx    Initialised context
 * @in     Input  buffer (plain or cipher)
 * @out    Output buffer (may alias @in for in-place)
 * @len    Byte count (any length, not block-aligned required)
 *
 * Stack cost: ~80 bytes (keystream block + counters)
 */
aes256_err_t aes256_ctr_crypt(aes256_ctx_t *ctx,
                               const uint8_t *in,
                               uint8_t       *out,
                               size_t         len);

/**
 * aes256_ecb_block - Single-block ECB encrypt (raw primitive).
 *                    Use for CMAC, key wrapping, etc.
 *
 * @ctx    Initialised context
 * @in     16-byte plaintext
 * @out    16-byte ciphertext (must not alias @in)
 */
aes256_err_t aes256_ecb_block(const aes256_ctx_t *ctx,
                               const uint8_t       in[AES256_BLOCK_SIZE],
                               uint8_t             out[AES256_BLOCK_SIZE]);

/**
 * aes256_zeroize - Wipe key material from context.
 *                  Call before context goes out of scope.
 */
void aes256_zeroize(aes256_ctx_t *ctx);

/**
 * aes256_benchmark - Measure encrypt throughput.
 *
 * @ctx          Initialised context
 * @blocks       Number of 16-byte blocks to encrypt
 * @cycles_out   Cycle count of operation (0 if no HW counter)
 * @bytes_out    Bytes processed (= blocks × 16)
 */
aes256_err_t aes256_benchmark(aes256_ctx_t *ctx,
                               uint32_t      blocks,
                               uint64_t     *cycles_out,
                               uint32_t     *bytes_out);

#endif /* AES256_ARM_H */