/**
 * main_demo.c - Self-test and benchmark harness
 *
 * Runs on:
 *   - Bare-metal Cortex-M (semihosting printf)
 *   - Linux/ARM64 userspace
 *   - QEMU aarch64-softmmu
 */

#include "aes256_arm.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  NIST SP 800-38A — AES-256-CTR test vector F.5.5 / F.5.6          */
/* ------------------------------------------------------------------ */
static const uint8_t tv_key[32] = {
    0x60,0x3d,0xeb,0x10, 0x15,0xca,0x71,0xbe,
    0x2b,0x73,0xae,0xf0, 0x85,0x7d,0x77,0x81,
    0x1f,0x35,0x2c,0x07, 0x3b,0x61,0x08,0xd7,
    0x2d,0x98,0x10,0xa3, 0x09,0x14,0xdf,0xf4
};

/* CTR IV = NIST counter block initial value (first 12 of 16 bytes)  */
static const uint8_t tv_iv[AES256_CTR_IV_SIZE] = {
    0xf0,0xf1,0xf2,0xf3, 0xf4,0xf5,0xf6,0xf7,
    0xf8,0xf9,0xfa,0xfb
};

/* NIST plaintext (4 blocks = 64 bytes)                              */
static const uint8_t tv_plain[64] = {
    0x6b,0xc1,0xbe,0xe2, 0x2e,0x40,0x9f,0x96,
    0xe9,0x3d,0x7e,0x11, 0x73,0x93,0x17,0x2a,
    0xae,0x2d,0x8a,0x57, 0x1e,0x03,0xac,0x9c,
    0x9e,0xb7,0x6f,0xac, 0x45,0xaf,0x8e,0x51,
    0x30,0xc8,0x1c,0x46, 0xa3,0x5c,0xe4,0x11,
    0xe5,0xfb,0xc1,0x19, 0x1a,0x0a,0x52,0xef,
    0xf6,0x9f,0x24,0x45, 0xdf,0x4f,0x9b,0x17,
    0xad,0x2b,0x41,0x7b, 0xe6,0x6c,0x37,0x10
};

/* NIST expected ciphertext                                          */
static const uint8_t tv_cipher[64] = {
    0x60,0x1e,0xc3,0x13, 0x77,0x57,0x89,0xa5,
    0xb7,0xa7,0xf5,0x04, 0xbb,0xf3,0xd2,0x28,
    0xf4,0x43,0xe3,0xca, 0x4d,0x62,0xb5,0x9a,
    0xca,0x84,0xe9,0x90, 0xca,0xca,0xf5,0xc5,
    0x2b,0x09,0x30,0xda, 0xa2,0x3d,0xe9,0x4c,
    0xe8,0x70,0x17,0xba, 0x2d,0x84,0x98,0x8d,
    0xdf,0xc9,0xc5,0x8d, 0xb6,0x7a,0xad,0xa6,
    0x13,0xc2,0xdd,0x08, 0x45,0x79,0x41,0xa6
};

/* ------------------------------------------------------------------ */

static void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("%-18s ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
        if ((i & 15U) == 15U && i + 1 < len) printf("\n%-18s ", "");
    }
    printf("\n");
}

int main(void)
{
    /* Enable DWT on Cortex-M before first benchmark                 */
    BENCH_DWT_ENABLE();

    /* Context on stack — no heap allocation                         */
    aes256_ctx_t ctx;

    printf("=== AES-256-CTR ARM Embedded Demo ===\n\n");
    printf("Platform: %s\n\n",
#if defined(AES_HW_AARCH64)
           "ARMv8-A 64-bit (hardware AES)"
#elif defined(AES_HW_ARMV8M)
           "ARMv8-M (hardware AES)"
#else
           "ARMv7-M / software fallback"
#endif
    );

    /* -------------------------------------------------------------- */
    /*  TEST 1: Encrypt                                               */
    /* -------------------------------------------------------------- */
    printf("--- NIST SP 800-38A AES-256-CTR Test ---\n");

    aes256_err_t err = aes256_init(&ctx, tv_key, tv_iv);
    if (err != AES256_OK) {
        printf("FAIL: init returned %d\n", (int)err);
        return 1;
    }

    uint8_t out_enc[64] = {0};
    err = aes256_ctr_crypt(&ctx, tv_plain, out_enc, sizeof(tv_plain));
    if (err != AES256_OK) {
        printf("FAIL: encrypt returned %d\n", (int)err);
        return 1;
    }

    print_hex("Plaintext[0..15]:", tv_plain,  16);
    print_hex("Ciphertext[0..15]:", out_enc,   16);
    print_hex("Expected[0..15]:",  tv_cipher, 16);

    int enc_ok = (memcmp(out_enc, tv_cipher, 64) == 0);
    printf("Encrypt: %s\n\n", enc_ok ? "PASS" : "FAIL");

    /* -------------------------------------------------------------- */
    /*  TEST 2: Decrypt (CTR is symmetric — same operation)          */
    /* -------------------------------------------------------------- */
    aes256_zeroize(&ctx);
    err = aes256_init(&ctx, tv_key, tv_iv);   /* Reset counter       */
    if (err != AES256_OK) { return 1; }

    uint8_t out_dec[64] = {0};
    err = aes256_ctr_crypt(&ctx, out_enc, out_dec, sizeof(out_enc));
    if (err != AES256_OK) { return 1; }

    int dec_ok = (memcmp(out_dec, tv_plain, 64) == 0);
    printf("Decrypt (round-trip): %s\n\n", dec_ok ? "PASS" : "FAIL");

    /* -------------------------------------------------------------- */
    /*  TEST 3: Benchmark                                             */
    /* -------------------------------------------------------------- */
    printf("--- Benchmark ---\n");

    const uint32_t BENCH_BLOCKS = 1000U;
    uint64_t cycles  = 0;
    uint32_t bytes   = 0;

    err = aes256_benchmark(&ctx, BENCH_BLOCKS, &cycles, &bytes);
    if (err != AES256_OK) { return 1; }

    printf("Blocks     : %u\n",    BENCH_BLOCKS);
    printf("Bytes      : %u\n",    bytes);
    printf("Cycles     : %llu\n",  (unsigned long long)cycles);

    if (cycles > 0U) {
        uint64_t cyc_per_block = cycles / BENCH_BLOCKS;
        uint64_t cyc_per_byte  = cycles / bytes;
        printf("Cycles/blk : %llu\n", (unsigned long long)cyc_per_block);
        printf("Cycles/byte: %llu\n", (unsigned long long)cyc_per_byte);
    } else {
        printf("(Cycle counter not available on this platform)\n");
    }

    /* -------------------------------------------------------------- */
    /*  TEST 4: In-place encryption                                   */
    /* -------------------------------------------------------------- */
    printf("\n--- In-place (aliased) encrypt ---\n");
    aes256_zeroize(&ctx);
    aes256_init(&ctx, tv_key, tv_iv);

    uint8_t inplace[64];
    memcpy(inplace, tv_plain, 64);
    aes256_ctr_crypt(&ctx, inplace, inplace, 64);   /* in == out      */
    int ip_ok = (memcmp(inplace, tv_cipher, 64) == 0);
    printf("In-place: %s\n\n", ip_ok ? "PASS" : "FAIL");

    /* -------------------------------------------------------------- */
    /*  Cleanup                                                        */
    /* -------------------------------------------------------------- */
    aes256_zeroize(&ctx);

    int all_pass = enc_ok && dec_ok && ip_ok;
    printf("=== Overall: %s ===\n", all_pass ? "ALL PASS" : "FAIL");
    return all_pass ? 0 : 1;
}