#ifndef LIBKMEM_H
#define LIBKMEM_H
#include <stddef.h>
#include <stdint.h>

typedef enum { KMEM_KMALLOC = 0, KMEM_PAGES = 1, KMEM_VMALLOC = 2, KMEM_DMA = 3 } kmem_type_t;

int   kmem_lib_init(void);           /* opens /dev/kmem_lab once */
void  kmem_lib_fini(void);

/* Allocates via the kernel driver and mmaps it; returns pointer or NULL.
 * out_actual_size (optional) reveals kernel-side rounding/fragmentation. */
void *kmem_alloc(kmem_type_t type, size_t size, unsigned flags, size_t *out_actual_size);
void  kmem_free(void *ptr);

#define KMEM_FLAG_UNCACHED 1u

#endif
