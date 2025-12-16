/* secure_mem.c */

#define _GNU_SOURCE             /* mlock, explicit_bzero            */
#include "secure_mem.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>           /* mlock, mprotect, MAP_ANONYMOUS   */
#include <errno.h>
#include <stdio.h>

/* ----------------------------------------------------------------
 * secure_memzero() — volatile-barrier wipe
 *
 * The cast through volatile pointer and compiler memory barrier
 * (via the empty asm block on GCC/Clang) together ensure this
 * store is never eliminated. On ARM this compiles to:
 *   STR  XZR, [X0]   (or equivalent STRB loop)
 * ---------------------------------------------------------------- */
void secure_memzero(void *ptr, size_t len)
{
    if (ptr == NULL || len == 0) {
        return;
    }

#if defined(__GLIBC__) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
    /*
     * SECURITY: explicit_bzero() is guaranteed by POSIX.1-2024
     * to not be optimized away. Preferred when available.
     */
    explicit_bzero(ptr, len);
#else
    /*
     * Fallback: volatile pointer prevents dead-store elimination.
     * The compiler memory barrier asm("" ::: "memory") prevents
     * reordering around this loop.
     */
    volatile unsigned char *vptr = (volatile unsigned char *)ptr;
    size_t i;
    for (i = 0; i < len; i++) {
        vptr[i] = 0U;
    }
    /* Compiler memory barrier — ARM: no instruction emitted,
     * but prevents compiler reordering of the store above */
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* ----------------------------------------------------------------
 * secure_memcmp() — constant-time comparison
 *
 * XOR difference accumulation:
 *   diff |= a[i] ^ b[i]  for all i
 *
 * The volatile qualifier on 'diff' prevents the compiler from
 * short-circuiting the loop once diff becomes non-zero.
 * The result is cast to bool (0 or 1) in constant time.
 *
 * IMPORTANT: This function assumes both buffers are the SAME
 * length. If comparing variable-length inputs, pad or hash first.
 * ---------------------------------------------------------------- */
int secure_memcmp(const void *a, const void *b, size_t len)
{
    const volatile unsigned char *pa =
        (const volatile unsigned char *)a;
    const volatile unsigned char *pb =
        (const volatile unsigned char *)b;
    volatile unsigned char diff = 0U;
    size_t i;

    if (a == NULL || b == NULL) {
        /*
         * SECURITY: Return non-zero (mismatch) on NULL inputs.
         * Never treat NULL == NULL as a successful auth comparison.
         */
        return 1;
    }

    for (i = 0; i < len; i++) {
        diff |= pa[i] ^ pb[i];
    }

    /*
     * Convert to 0 (equal) or 1 (not equal) without branching.
     * ((diff - 1) >> 8) & 1  evaluates to:
     *   diff == 0 → (0xFF...FF >> 8) & 1 → 1  (equal)
     *   diff != 0 → non-negative >> 8  & 1 → 0 (not equal)
     * Invert to match memcmp() convention (0 = equal).
     */
    return (int)(1U - (((unsigned int)(diff - 1U)) >> 8U & 1U));
}

/* ----------------------------------------------------------------
 * secure_memcpy_checked() — bounds-checked copy
 * ---------------------------------------------------------------- */
int secure_memcpy_checked(void       *dst, size_t dst_size,
                          const void *src, size_t copy_len)
{
    if (dst == NULL || src == NULL) {
        return SECURE_MEM_ERR_NULL;
    }
    if (copy_len > dst_size) {
        return SECURE_MEM_ERR_SIZE;
    }
    /* Overlap check — undefined behavior if regions overlap */
    const uintptr_t d = (uintptr_t)dst;
    const uintptr_t s = (uintptr_t)src;
    if ((d < s + copy_len) && (s < d + dst_size)) {
        return SECURE_MEM_ERR_SIZE;  /* Use memmove for overlaps   */
    }
    memcpy(dst, src, copy_len);
    return SECURE_MEM_OK;
}

/* ----------------------------------------------------------------
 * secure_alloc() / secure_free()
 *
 * SECURITY design decisions:
 * 1. mmap(MAP_ANONYMOUS|MAP_PRIVATE) — bypasses heap metadata
 *    attacks (heap spray, use-after-free on allocator internals)
 * 2. mlock() — prevents swap to disk (cold-boot / swap attack)
 * 3. PROT_READ|PROT_WRITE initially, PROT_NONE after free
 * 4. Overallocate by one page on each side for guard pages
 *    (catches buffer overflows into adjacent key material)
 * ---------------------------------------------------------------- */

#define PAGE_SIZE       (4096UL)
#define GUARD_SIZE      PAGE_SIZE

/* Header stored in the first guard page (readable metadata) */
typedef struct {
    size_t alloc_size;      /* Requested allocation size            */
    size_t total_size;      /* Total mmap size including guards     */
} secure_alloc_header_t;

void *secure_alloc(size_t size)
{
    if (size == 0 || size > (1UL << 24)) {     /* Max 16MB per alloc */
        return NULL;
    }

    /*
     * Layout:
     * [GUARD PAGE (PROT_NONE)] [HEADER PAGE] [DATA PAGES] [GUARD PAGE]
     */
    size_t data_pages  = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t total_size  = GUARD_SIZE            /* leading guard  */
                       + PAGE_SIZE             /* header page    */
                       + (data_pages * PAGE_SIZE)
                       + GUARD_SIZE;           /* trailing guard */

    uint8_t *base = (uint8_t *)mmap(NULL, total_size,
                                    PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE,
                                    -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    /* Protect leading guard page — any access triggers SIGSEGV */
    if (mprotect(base, GUARD_SIZE, PROT_NONE) != 0) {
        munmap(base, total_size);
        return NULL;
    }

    /* Protect trailing guard page */
    uint8_t *trail_guard = base + GUARD_SIZE + PAGE_SIZE
                         + (data_pages * PAGE_SIZE);
    if (mprotect(trail_guard, GUARD_SIZE, PROT_NONE) != 0) {
        munmap(base, total_size);
        return NULL;
    }

    /* Write header in header page (after leading guard) */
    secure_alloc_header_t *hdr =
        (secure_alloc_header_t *)(base + GUARD_SIZE);
    hdr->alloc_size = size;
    hdr->total_size = total_size;

    /* Data region starts after leading guard + header page */
    uint8_t *data = base + GUARD_SIZE + PAGE_SIZE;

    /*
     * SECURITY: mlock() the data region to prevent swap.
     * Failure is non-fatal (may lack CAP_IPC_LOCK), warn only.
     */
    if (mlock(data, data_pages * PAGE_SIZE) != 0) {
        /* In production: log warning, continue (best effort) */
        (void)0;
    }

    return (void *)data;
}

void secure_free(void *ptr, size_t size)
{
    if (ptr == NULL) {
        return;
    }

    /* Recover base from pointer arithmetic */
    uint8_t *data = (uint8_t *)ptr;
    uint8_t *hdr_page = data - PAGE_SIZE;
    secure_alloc_header_t *hdr = (secure_alloc_header_t *)hdr_page;
    uint8_t *base = hdr_page - GUARD_SIZE;

    size_t alloc_size = (size > 0) ? size : hdr->alloc_size;
    size_t total_size = hdr->total_size;

    /* SECURITY: Zeroize before unmapping */
    secure_memzero(data, alloc_size);

    /* Unlock and unmap the entire region */
    munmap(base, total_size);
    /* munmap implicitly unlocks mlocked pages per POSIX */
}