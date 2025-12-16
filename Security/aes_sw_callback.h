/* ================================================================== */
/*  PATH C — Software fallback (ARMv7-M, Cortex-M4/M7, no crypto ext)*/
/* ================================================================== */
#if defined(AES_SW_FALLBACK)

/*
 * Compact AES S-box: 256 bytes in flash/ROM.
 * We use a single forward S-box only (encrypt path).
 * No 1 KB T-table — saves flash, costs ~3× cycles vs T-table AES.
 * For Cortex-M4 @168 MHz: ~120 cycles/block with T-tables,
 *                          ~380 cycles/block with S-box only.
 */
static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,
    0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
    0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,
    0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,
    0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
    0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,
    0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,
    0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
    0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,
    0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,
    0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
    0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,
    0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,
    0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
    0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,
    0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,
    0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

/* GF(2^8) multiply by 2 — used in MixColumns                        */
static inline uint8_t xtime(uint8_t a)
{
    return (uint8_t)((a << 1) ^ ((a >> 7) ? 0x1b : 0x00));
}

/* GF multiply by 3                                                   */
static inline uint8_t gmul3(uint8_t a) { return xtime(a) ^ a; }

/* ------------------------------------------------------------------ */
/*  Software key schedule — shared with HW paths for key expansion    */
/* ------------------------------------------------------------------ */
void sw_key_expand(uint32_t rk[60], const uint8_t key[32])
{
    static const uint32_t rcon[10] = {
        0x01000000U, 0x02000000U, 0x04000000U, 0x08000000U,
        0x10000000U, 0x20000000U, 0x40000000U, 0x80000000U,
        0x1b000000U, 0x36000000U
    };

    /* Load 8 key words                                               */
    for (unsigned i = 0; i < 8; i++) {
        rk[i] = be_load32(key + 4U * i);
    }

    for (unsigned i = 8; i < 60; i++) {
        uint32_t t = rk[i - 1];

        if ((i & 7U) == 0U) {
            /* RotWord + SubWord + Rcon                               */
            t = rot32(t, 8U);
            t = ((uint32_t)sbox[(t >> 24) & 0xff] << 24)
              | ((uint32_t)sbox[(t >> 16) & 0xff] << 16)
              | ((uint32_t)sbox[(t >>  8) & 0xff] <<  8)
              | ((uint32_t)sbox[ t        & 0xff]);
            t ^= rcon[(i >> 3) - 1U];

        } else if ((i & 7U) == 4U) {
            /* SubWord only (AES-256 extra step every 4th word)       */
            t = ((uint32_t)sbox[(t >> 24) & 0xff] << 24)
              | ((uint32_t)sbox[(t >> 16) & 0xff] << 16)
              | ((uint32_t)sbox[(t >>  8) & 0xff] <<  8)
              | ((uint32_t)sbox[ t        & 0xff]);
        }

        rk[i] = rk[i - 8U] ^ t;
    }
}

/* ------------------------------------------------------------------ */
/*  Software single-block ECB encrypt                                  */
/* ------------------------------------------------------------------ */
static void sw_ecb_encrypt(const uint32_t rk[60],
                            const uint8_t  in[16],
                            uint8_t        out[16])
{
    /* State array — 16 bytes on stack                                */
    uint8_t s[16];
    memcpy(s, in, 16);

    /* Initial round key add                                          */
    for (unsigned i = 0; i < 16; i++) {
        s[i] ^= ((const uint8_t *)rk)[i ^ 3U];   /* big-endian adapt*/
    }

    /*
     * Main rounds: SubBytes → ShiftRows → MixColumns → AddRoundKey
     * Unrolled into a single byte-oriented loop to keep code small.
     */
    for (unsigned round = 1; round < AES256_ROUNDS; round++) {

        /* SubBytes + ShiftRows combined                              */
        uint8_t t[16];
        t[ 0] = sbox[s[ 0]];  t[ 1] = sbox[s[ 5]];
        t[ 2] = sbox[s[10]];  t[ 3] = sbox[s[15]];
        t[ 4] = sbox[s[ 4]];  t[ 5] = sbox[s[ 9]];
        t[ 6] = sbox[s[14]];  t[ 7] = sbox[s[ 3]];
        t[ 8] = sbox[s[ 8]];  t[ 9] = sbox[s[13]];
        t[10] = sbox[s[ 2]];  t[11] = sbox[s[ 7]];
        t[12] = sbox[s[12]];  t[13] = sbox[s[ 1]];
        t[14] = sbox[s[ 6]];  t[15] = sbox[s[11]];

        /* MixColumns — operates on each 4-byte column               */
        for (unsigned c = 0; c < 4; c++) {
            uint8_t a0 = t[4*c+0], a1 = t[4*c+1];
            uint8_t a2 = t[4*c+2], a3 = t[4*c+3];
            s[4*c+0] = xtime(a0) ^ gmul3(a1) ^       a2  ^       a3;
            s[4*c+1] =       a0  ^ xtime(a1) ^ gmul3(a2) ^       a3;
            s[4*c+2] =       a0  ^       a1  ^ xtime(a2) ^ gmul3(a3);
            s[4*c+3] = gmul3(a0) ^       a1  ^       a2  ^ xtime(a3);
        }

        /* AddRoundKey                                                */
        const uint8_t *prk = (const uint8_t *)(rk + 4U * round);
        for (unsigned i = 0; i < 16; i++) {
            s[i] ^= prk[i ^ 3U];
        }
    }

    /* Final round: SubBytes + ShiftRows + AddRoundKey (no MixCols)  */
    uint8_t t[16];
    t[ 0] = sbox[s[ 0]];  t[ 1] = sbox[s[ 5]];
    t[ 2] = sbox[s[10]];  t[ 3] = sbox[s[15]];
    t[ 4] = sbox[s[ 4]];  t[ 5] = sbox[s[ 9]];
    t[ 6] = sbox[s[14]];  t[ 7] = sbox[s[ 3]];
    t[ 8] = sbox[s[ 8]];  t[ 9] = sbox[s[13]];
    t[10] = sbox[s[ 2]];  t[11] = sbox[s[ 7]];
    t[12] = sbox[s[12]];  t[13] = sbox[s[ 1]];
    t[14] = sbox[s[ 6]];  t[15] = sbox[s[11]];

    const uint8_t *prk = (const uint8_t *)(rk + 4U * AES256_ROUNDS);
    for (unsigned i = 0; i < 16; i++) {
        out[i] = t[i] ^ prk[i ^ 3U];
    }

    /* Wipe temporaries                                               */
    secure_zero(s, sizeof(s));
    secure_zero(t, sizeof(t));
}

#define PLATFORM_ECB_ENCRYPT(rk, in, out)   sw_ecb_encrypt(rk, in, out)
#define PLATFORM_KEY_EXPAND(ctx, key)       sw_key_expand((ctx)->rk, key)

#endif /* AES_SW_FALLBACK */