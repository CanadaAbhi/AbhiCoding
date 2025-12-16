#define PLATFORM_ECB_ENCRYPT(rk, in, out)   hw_ecb_encrypt(rk, in, out)
#define PLATFORM_KEY_EXPAND(ctx, key)       hw_key_expand(ctx, key)

/* ================================================================== */
/*  PATH B — ARMv8-M hardware (Cortex-M33/M35P/M55)                  */
/* ================================================================== */
#elif defined(AES_HW_ARMV8M)

#include <arm_acle.h>

/*
 * ARMv8-M AES instructions operate on 32-bit words.
 * This path wraps the __saes_enc / __saes_sub ACLE intrinsics.
 * Not all toolchains expose these — fallback to SW if missing.
 */
#if defined(__ARM_FEATURE_AES)

static void armv8m_ecb_encrypt(const uint32_t rk[60],
                                const uint8_t  in[16],
                                uint8_t        out[16])
{
    /* Load state as 4 × 32-bit words                                 */
    uint32_t s0 = be_load32(in +  0) ^ rk[0];
    uint32_t s1 = be_load32(in +  4) ^ rk[1];
    uint32_t s2 = be_load32(in +  8) ^ rk[2];
    uint32_t s3 = be_load32(in + 12) ^ rk[3];

    /*
     * Full AES-256 round loop using __saes_enc.
     * ARMv8-M toolchain support varies — guard with __ARM_FEATURE_AES.
     */
    for (unsigned r = 1; r < AES256_ROUNDS; r++) {
        uint32_t t0 = __saes_enc(s0, s1) ^ rk[4*r + 0];
        uint32_t t1 = __saes_enc(s1, s2) ^ rk[4*r + 1];
        uint32_t t2 = __saes_enc(s2, s3) ^ rk[4*r + 2];
        uint32_t t3 = __saes_enc(s3, s0) ^ rk[4*r + 3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }
    /* Final round                                                    */
    uint32_t idx = 4U * AES256_ROUNDS;
    s0 = __saes_sub(s0, s1) ^ rk[idx + 0];
    s1 = __saes_sub(s1, s2) ^ rk[idx + 1];
    s2 = __saes_sub(s2, s3) ^ rk[idx + 2];
    s3 = __saes_sub(s3, s0) ^ rk[idx + 3];

    be_store32(out +  0, s0);
    be_store32(out +  4, s1);
    be_store32(out +  8, s2);
    be_store32(out + 12, s3);
}

#  define PLATFORM_ECB_ENCRYPT(rk, in, out)   armv8m_ecb_encrypt(rk, in, out)

#else /* __ARM_FEATURE_AES not available on this ARMv8-M toolchain    */
#  define AES_SW_FALLBACK  1
#endif

#endif /* AES_HW_ARMV8M */