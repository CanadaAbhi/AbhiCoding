static inline uint32_t read_scr(void)
{
    uint32_t scr;
    asm volatile("mrc p15, 0, %0, c1, c1, 0" : "=r"(scr));
    return scr;
}

static inline void write_scr(uint32_t scr)
{
    asm volatile("mcr p15, 0, %0, c1, c1, 0" :: "r"(scr) : "memory");
}

void switch_to_nonsecure(void)
{
    uint32_t scr = read_scr();
    scr |= (1 << 0);     // NS bit = 1 -> Non-secure state
    write_scr(scr);
}

void switch_to_secure(void)
{
    uint32_t scr = read_scr();
    scr &= ~(1 << 0);    // NS bit = 0 -> Secure state
    write_scr(scr);
}
