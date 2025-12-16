/**
 * aes256_arm.c - Lightweight AES-256 implementation for ARM
 *
 * Paths compiled:
 *   AES_HW_AARCH64  — ARMv8-A intrinsics  (vaeseq_u8 / vaesdq_u8)
 *   AES_HW_ARMV8M   — ARMv8-M intrinsics  (__saes_enc / __saes_keyschedule)
 *   AES_SW_FALLBACK — Compact C (lookup-table, < 2 KB flash, ~560 B stack)
 *
 * Compile flags (GCC / Clang):
 *   aarch64:  -O2 -march=armv8-a+crypto -mfpu=crypto-neon-fp-armv8
 *   armv7-m:  -O2 -march=armv7-m (SW fallback)
 *   armv8-m:  -O2 -march=armv8-m.main+dsp+crypto
 */

#include "aes256_arm.h"

#include <string.h>     /* memcpy, memset                             */
#include <stdint.h>

/* ================================================================== */
/*  INTERNAL UTILITIES                                                 */
/* ================================================================== */

/* Volatile barrier prevents compiler from optimising away zeroise    */
static void secure_zero(volatile void *p, size_t n)
{
    volatile uint8_t *b = (volatile uint8_t *)p;
    while (n--) *b++ = 0;
}

/* Rotate left 32-bit — compiled to single ARM ROR/ROL instruction   */
static inline uint32_t rot32(uint32_t x, unsigned n)
{
    return (x << n) | (x >> (32U - n));
}

/* Pack / unpack big-endian 32-bit words                              */
static inline uint32_t be_load32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

static inline void be_store32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8);
    p[3] = (uint8_t)(v);
}

/* ================================================================== */
/*  PATH A — ARMv8-A 64-bit hardware acceleration                     */
/* ================================================================== */
#if defined(AES_HW_AARCH64)

#include <arm_neon.h>

/* ------------------------------------------------------------------ */
/*  HW key schedule — 15 round keys packed into uint8x16_t array     */
/*  AES-256 needs 60 × 32-bit words = 15 × 128-bit round keys        */
/* ------------------------------------------------------------------ */

/* AES-256 Rcon for key schedule                                      */
static const uint8_t rcon_hw[7] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40
};

/* Expand one 256-bit key into 15 round keys stored in ctx->rk[]     */
static void hw_key_expand(aes256_ctx_t *ctx, const uint8_t key[32])
{
    /*
     * AArch64 NEON intrinsics do not provide a key-schedule
     * instruction (unlike x86 AESKEYGENASSIST). We therefore use
     * the software schedule but store results into ctx->rk[] in the
     * same uint32_t format — the encrypt path will load them back
     * into uint8x16_t vectors.
     */

    /* Reuse SW schedule path — defined below                         */
    extern void sw_key_expand(uint32_t rk[60], const uint8_t key[32]);
    sw_key_expand(ctx->rk, key);
}

/* ------------------------------------------------------------------ */
/*  HW single-block ECB encrypt using vaeseq / vaesmcq intrinsics     */
/* ------------------------------------------------------------------ */
static void hw_ecb_encrypt(const uint32_t rk[60],
                            const uint8_t  in[16],
                            uint8_t        out[16])
{
    uint8x16_t b   = vld1q_u8(in);
    uint8x16_t rk0 = vld1q_u8((const uint8_t *)(rk +  0));
    uint8x16_t rk1 = vld1q_u8((const uint8_t *)(rk +  4));
    uint8x16_t rk2 = vld1q_u8((const uint8_t *)(rk +  8));
    uint8x16_t rk3 = vld1q_u8((const uint8_t *)(rk + 12));
    uint8x16_t rk4 = vld1q_u8((const uint8_t *)(rk + 16));
    uint8x16_t rk5 = vld1q_u8((const uint8_t *)(rk + 20));
    uint8x16_t rk6 = vld1q_u8((const uint8_t *)(rk + 24));
    uint8x16_t rk7 = vld1q_u8((const uint8_t *)(rk + 28));
    uint8x16_t rk8 = vld1q_u8((const uint8_t *)(rk + 32));
    uint8x16_t rk9 = vld1q_u8((const uint8_t *)(rk + 36));
    uint8x16_t rkA = vld1q_u8((const uint8_t *)(rk + 40));
    uint8x16_t rkB = vld1q_u8((const uint8_t *)(rk + 44));
    uint8x16_t rkC = vld1q_u8((const uint8_t *)(rk + 48));
    uint8x16_t rkD = vld1q_u8((const uint8_t *)(rk + 52));
    uint8x16_t rkE = vld1q_u8((const uint8_t *)(rk + 56));

    /* 13 rounds: SubBytes + ShiftRows + MixColumns + AddRoundKey     */
    b = vaeseq_u8(b, rk0);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk1);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk2);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk3);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk4);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk5);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk6);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk7);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk8);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rk9);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rkA);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rkB);  b = vaesmcq_u8(b);
    b = vaeseq_u8(b, rkC);  b = vaesmcq_u8(b);

    /* Final round: SubBytes + ShiftRows + AddRoundKey (no MixCols)  */
    b = vaeseq_u8(b, rkD);
    b = veorq_u8(b, rkE);

    vst1q_u8(out, b);
}