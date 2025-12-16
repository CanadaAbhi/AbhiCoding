// libkmem.c -- Custom Memory API: the application-facing layer that hides
// the ioctl-handle + mmap-offset dance behind malloc()/free()-shaped calls.
#include "libkmem.h"
#include "../driver/kmem_lab_uapi.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#define MAX_LIVE 4096
static int g_fd = -1;
static struct { void *ptr; size_t len; unsigned long handle; } g_live[MAX_LIVE];
static int g_nlive = 0;

int kmem_lib_init(void)
{
    if (g_fd >= 0) return 0;
    g_fd = open("/dev/kmem_lab", O_RDWR);
    return g_fd < 0 ? -1 : 0;
}

void kmem_lib_fini(void)
{
    if (g_fd >= 0) close(g_fd);
    g_fd = -1;
}

void *kmem_alloc(kmem_type_t type, size_t size, unsigned flags, size_t *out_actual_size)
{
    struct kmem_alloc_req req = { .type = type, .flags = flags, .size = size };
    void *ptr;

    if (ioctl(g_fd, KMEM_IOC_ALLOC, &req) < 0)
        return NULL;

    /* offset param encodes the handle, shifted into page units, matching
     * the driver's use of vma->vm_pgoff as the buffer id. */
    ptr = mmap(NULL, req.actual_size, PROT_READ | PROT_WRITE, MAP_SHARED,
               g_fd, (off_t)req.handle << 12);
    if (ptr == MAP_FAILED) {
        struct kmem_free_req f = { .handle = req.handle };
        ioctl(g_fd, KMEM_IOC_FREE, &f);
        return NULL;
    }

    if (g_nlive < MAX_LIVE) {
        g_live[g_nlive].ptr = ptr;
        g_live[g_nlive].len = req.actual_size;
        g_live[g_nlive].handle = req.handle;
        g_nlive++;
    }
    if (out_actual_size) *out_actual_size = req.actual_size;
    return ptr;
}

void kmem_free(void *ptr)
{
    for (int i = 0; i < g_nlive; i++) {
        if (g_live[i].ptr == ptr) {
            munmap(ptr, g_live[i].len);
            struct kmem_free_req f = { .handle = g_live[i].handle };
            ioctl(g_fd, KMEM_IOC_FREE, &f);
            g_live[i] = g_live[--g_nlive];
            return;
        }
    }
}
