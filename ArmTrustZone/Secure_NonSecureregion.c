#define TZASC_BASE                  0xF9020000UL

#define TZASC_REGION_LOW(n)   (*(volatile uint32_t *)(TZASC_BASE + 0x100 + (n)*0x10))
#define TZASC_REGION_HIGH(n)  (*(volatile uint32_t *)(TZASC_BASE + 0x104 + (n)*0x10))
#define TZASC_REGION_ATTR(n)  (*(volatile uint32_t *)(TZASC_BASE + 0x108 + (n)*0x10))

#define TZASC_ATTR_ENABLE     (1u << 0)
#define TZASC_ATTR_NS_ALLOW   (1u << 31)   // allow Non-secure masters

void tzasc_configure_secure_region(uint32_t region, uint32_t base_addr,
                                    uint32_t size_log2)
{
    TZASC_REGION_LOW(region)  = base_addr;
    TZASC_REGION_HIGH(region) = 0x0;

    uint32_t attr = (size_log2 << 1) | TZASC_ATTR_ENABLE;
    attr &= ~TZASC_ATTR_NS_ALLOW;   // Secure masters only
    TZASC_REGION_ATTR(region) = attr;
}

void tzasc_configure_nonsecure_region(uint32_t region, uint32_t base_addr,
                                       uint32_t size_log2)
{
    TZASC_REGION_LOW(region)  = base_addr;
    TZASC_REGION_HIGH(region) = 0x0;

    uint32_t attr = (size_log2 << 1) | TZASC_ATTR_ENABLE;
    attr |= TZASC_ATTR_NS_ALLOW;    // Non-secure masters permitted
    TZASC_REGION_ATTR(region) = attr;
}
