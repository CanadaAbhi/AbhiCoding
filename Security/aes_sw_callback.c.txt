
/* ================================================================== */
/*  HW path KEY EXPAND alias (aarch64 uses sw schedule internally)    */
/* ================================================================== */
#if defined(AES_HW_AARCH64) && !defined(PLATFORM_KEY_EXPAND)
#  define PLATFORM_KEY_EXPAND(ctx, key)   hw_key_expand(ctx, key)
#endif

#if defined(AES_HW_ARMV8M) && !defined(AES_SW_FALLBACK) && !defined(PLATFORM_KEY_EXPAND)
   /* ARMv8-M key schedule — reuse SW key expand                     */
   static void armv8m_key_expand(aes256_ctx_t *ctx, const uint8_t key[32])
   {
       sw_key_expand(ctx->rk, key);
   }
#  define PLATFORM_KEY_EXPAND(ctx, key)   armv8m_key_expand(ctx, key)
#endif

/* ================================================================== */
/*  CTR MODE — platform-agnostic, builds on PLATFORM_ECB_ENCRYPT     */
/* ================================================================== */

/*
 * Build a 16-byte CTR block from the 12-byte nonce and 32-bit counter.
 * Format: [nonce 12 bytes][counter 4 bytes big-endian]
 * Follows RFC 3686 / NIST SP 800-38A style.
 */
static void build_ctr_block(const uint8_t iv[12],
                              uint32_t      ctr,
                              uint8_t       block[16])
{
    memcpy(block, iv, 12U);
    be_store32(block + 12, ctr);
}

/* ================================================================== */
/*  PUBLIC API IMPLEMENTATION                                          */
/* ================================================================== */

aes256_err_t aes256_init(aes256_ctx_t       *ctx,
                          const uint8_t       key[AES256_KEY_SIZE],
                          const uint8_t       iv[AES256_CTR_IV_SIZE])
{
    if (!ctx || !key || !iv) {
        return AES256_ERR_NULL_PTR;
    }

    /* Detect hardware path at runtime                                */
#if defined(AES_HW_AARCH64) || defined(AES_HW_ARMV8M)
    ctx->hw_accel = true;
#else
    ctx->hw_accel = false;
#endif

    /* Key schedule expansion                                         */
    PLATFORM_KEY_EXPAND(ctx, key);

    /* Store CTR nonce and initialise counter to 1 (RFC 3686)        */
    memcpy(ctx->iv, iv, AES256_CTR_IV_SIZE);
    ctx->ctr = 1U;

    return AES256_OK;
}

/* ------------------------------------------------------------------ */

aes256_err_t aes256_ecb_block(const aes256_ctx_t *ctx,
                               const uint8_t       in[AES256_BLOCK_SIZE],
                               uint8_t             out[AES256_BLOCK_SIZE])
{
    if (!ctx || !in || !out) {
        return AES256_ERR_NULL_PTR;
    }
    PLATFORM_ECB_ENCRYPT(ctx->rk, in, out);
    return AES256_OK;
}

/* ------------------------------------------------------------------ */

aes256_err_t aes256_ctr_crypt(aes256_ctx_t *ctx,
                               const uint8_t *in,
                               uint8_t       *out,
                               size_t         len)
{
    if (!ctx || !in || !out) {
        return AES256_ERR_NULL_PTR;
    }

    uint8_t  ctr_block[AES256_BLOCK_SIZE];  /* 16 bytes stack        */
    uint8_t  keystream[AES256_BLOCK_SIZE];  /* 16 bytes stack        */
    size_t   offset = 0;

    while (offset < len) {

        /* Overflow check — must rekey before counter wraps          */
        if (ctx->ctr == 0xFFFFFFFFU) {
            secure_zero(keystream, sizeof(keystream));
            return AES256_ERR_CTR_WRAP;
        }

        /* Generate keystream block                                   */
        build_ctr_block(ctx->iv, ctx->ctr, ctr_block);
        PLATFORM_ECB_ENCRYPT(ctx->rk, ctr_block, keystream);
        ctx->ctr++;

        /* XOR with input — handle partial final block                */
        size_t blk = (len - offset) < AES256_BLOCK_SIZE
                   ? (len - offset)
                   : AES256_BLOCK_SIZE;

        for (size_t i = 0; i < blk; i++) {
            out[offset + i] = in[offset + i] ^ keystream[i];
        }
        offset += blk;
    }

    /* Wipe keystream — do not leave key material on stack           */
    secure_zero(keystream, sizeof(keystream));
    secure_zero(ctr_block, sizeof(ctr_block));

    return AES256_OK;
}

/* ------------------------------------------------------------------ */

void aes256_zeroize(aes256_ctx_t *ctx)
{
    if (ctx) {
        secure_zero(ctx, sizeof(aes256_ctx_t));
    }
}

/* ------------------------------------------------------------------ */

aes256_err_t aes256_benchmark(aes256_ctx_t *ctx,
                               uint32_t      blocks,
                               uint64_t     *cycles_out,
                               uint32_t     *bytes_out)
{
    if (!ctx || !cycles_out || !bytes_out || blocks == 0U) {
        return AES256_ERR_NULL_PTR;
    }

    /*
     * Fixed test vector — NIST FIPS-197 Appendix B plaintext.
     * Stack-allocated, never changes, no heap needed.
     */
    static const uint8_t test_in[AES256_BLOCK_SIZE] = {
        0x32,0x43,0xf6,0xa8, 0x88,0x5a,0x30,0x8d,
        0x31,0x31,0x98,0xa2, 0xe0,0x37,0x07,0x34
    };
    uint8_t test_out[AES256_BLOCK_SIZE];

    uint64_t t_start, t_end;
    BENCH_READ_CYCLES(t_start);

    for (uint32_t i = 0; i < blocks; i++) {
        PLATFORM_ECB_ENCRYPT(ctx->rk, test_in, test_out);
        /*
         * Prevent dead-code elimination: XOR accumulate into
         * a volatile sink so the compiler cannot remove the loop.
         */
        __asm__ volatile("" :: "r"(test_out[0]) : "memory");
    }

    BENCH_READ_CYCLES(t_end);

    *cycles_out = (t_end >= t_start) ? (t_end - t_start) : 0U;
    *bytes_out  = blocks * AES256_BLOCK_SIZE;

    secure_zero(test_out, sizeof(test_out));
    return AES256_OK;
}