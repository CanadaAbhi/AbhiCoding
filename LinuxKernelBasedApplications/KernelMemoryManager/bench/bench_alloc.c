// bench_alloc.c -- compares malloc() / mmap(MAP_ANONYMOUS) / kmem_alloc()
// (KMALLOC, PAGES cached, PAGES uncached, VMALLOC, DMA) across five axes:
// allocation latency, fragmentation, page faults, cache behavior, throughput.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <malloc.h>
#include "../lib/libkmem.h"

#define NITER      2000
#define BUF_SIZE   (64 * 1024)   /* 64KB per allocation */
#define TOUCH_SIZE BUF_SIZE

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/* ---------- 1. Allocation latency ---------- */

typedef void *(*alloc_fn_t)(size_t);
typedef void  (*free_fn_t)(void *);

static void bench_latency(const char *name, void *(*alloc_fn)(void), void (*free_fn)(void *))
{
    double *lat = malloc(NITER * sizeof(double));
    for (int i = 0; i < NITER; i++) {
        double t0 = now_us();
        void *p = alloc_fn();
        double t1 = now_us();
        lat[i] = t1 - t0;
        free_fn(p);
    }
    /* simple P50/P99 */
    for (int i = 0; i < NITER; i++)
        for (int j = i + 1; j < NITER; j++)
            if (lat[j] < lat[i]) { double t = lat[i]; lat[i] = lat[j]; lat[j] = t; }
    printf("%-18s alloc-latency: P50=%.2fus P99=%.2fus\n",
           name, lat[NITER/2], lat[(int)(NITER*0.99)]);
    free(lat);
}

static void *g_last;
static void *do_malloc(void)   { return (g_last = malloc(BUF_SIZE)); }
static void  do_free_malloc(void *p) { free(p); }
static void *do_mmap(void)     { return (g_last = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                                                         MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)); }
static void  do_free_mmap(void *p)   { munmap(p, BUF_SIZE); }

static kmem_type_t g_ktype; static unsigned g_kflags;
static void *do_kmem(void)     { return (g_last = kmem_alloc(g_ktype, BUF_SIZE, g_kflags, NULL)); }
static void  do_free_kmem(void *p)   { kmem_free(p); }

/* ---------- 2. Fragmentation ---------- */

static void bench_fragmentation_malloc(void)
{
    /* alternating alloc of varying sizes with every-other freed -- creates
     * classic external fragmentation holes in the glibc heap. */
    void *ptrs[512];
    size_t sizes[] = { 16, 512, 4096, 128, 65536, 32 };
    for (int i = 0; i < 512; i++)
        ptrs[i] = malloc(sizes[i % 6]);
    for (int i = 0; i < 512; i += 2) { free(ptrs[i]); ptrs[i] = NULL; }

    struct mallinfo2 mi = mallinfo2();
    size_t requested = 0;
    for (int i = 1; i < 512; i += 2) requested += sizes[i % 6];
    printf("malloc fragmentation: heap_arena=%zu bytes, live_requested=%zu bytes, "
           "overhead=%.1f%%\n", mi.arena, requested,
           100.0 * ((double)mi.arena - requested) / requested);
    for (int i = 0; i < 512; i++) if (ptrs[i]) free(ptrs[i]);
}

static void bench_fragmentation_kernel(kmem_type_t type, const char *name)
{
    /* kernel-side internal fragmentation: requested size vs actual backing
     * size (kmalloc slab rounding, alloc_pages order rounding, PAGE_ALIGN). */
    size_t sizes[] = { 100, 900, 3000, 5000, 20000, 60000 };
    size_t req_total = 0, act_total = 0;
    void *ptrs[6];
    for (int i = 0; i < 6; i++) {
        size_t actual;
        ptrs[i] = kmem_alloc(type, sizes[i], 0, &actual);
        req_total += sizes[i];
        act_total += actual;
    }
    printf("%-18s fragmentation: requested=%zu actual=%zu overhead=%.1f%%\n",
           name, req_total, act_total, 100.0 * (act_total - req_total) / req_total);
    for (int i = 0; i < 6; i++) kmem_free(ptrs[i]);
}

/* ---------- 3. Page faults ---------- */

static void bench_page_faults(const char *name, void *ptr, size_t len)
{
    struct rusage r0, r1;
    getrusage(RUSAGE_SELF, &r0);
    volatile char *p = ptr;
    for (size_t i = 0; i < len; i += 4096) p[i] = 1; /* first touch */
    getrusage(RUSAGE_SELF, &r1);
    printf("%-18s page-faults: minor=%ld major=%ld (touching %zu bytes)\n",
           name, r1.ru_minflt - r0.ru_minflt, r1.ru_majflt - r0.ru_majflt, len);
}

/* ---------- 4. Cache behavior ---------- */

static double bench_stride_bw(void *buf, size_t len, size_t stride)
{
    volatile char *p = buf;
    double t0 = now_us();
    long sum = 0;
    for (int rep = 0; rep < 20; rep++)
        for (size_t i = 0; i < len; i += stride)
            sum += p[i];
    double t1 = now_us();
    (void)sum;
    double bytes = (double)(len / stride) * 20;
    return bytes / ((t1 - t0) / 1e6) / 1e6; /* "M accesses/sec" */
}

static void bench_cache(const char *name, void *buf, size_t len)
{
    printf("%-18s cache: stride64=%.1fM/s stride256=%.1fM/s stride4096=%.1fM/s\n",
           name, bench_stride_bw(buf, len, 64),
           bench_stride_bw(buf, len, 256),
           bench_stride_bw(buf, len, 4096));
}

/* ---------- 5. Throughput ---------- */

static void bench_throughput(const char *name, void *buf, size_t len)
{
    void *tmp = malloc(len);
    double t0 = now_us();
    for (int i = 0; i < 200; i++) memcpy(tmp, buf, len);
    double t1 = now_us();
    double gbps = (200.0 * len) / ((t1 - t0) / 1e6) / 1e9;
    printf("%-18s throughput: %.2f GB/s\n", name, gbps);
    free(tmp);
}

/* ---------- driver ---------- */

static void run_all(const char *name, void *buf, size_t len)
{
    bench_page_faults(name, buf, len);
    bench_cache(name, buf, len);
    bench_throughput(name, buf, len);
}+
 

int main(void)
{
    if (kmem_lib_init() < 0) {
        fprintf(stderr, "failed to open /dev/kmem_lab (insmod kmem_lab_drv.ko?)\n");
        return 1;
    }

    printf("=== 1. Allocation latency ===\n");
    bench_latency("malloc",            do_malloc, do_free_malloc);
    bench_latency("mmap",              do_mmap,   do_free_mmap);
    g_ktype = KMEM_KMALLOC; g_kflags = 0;
    bench_latency("kmem_KMALLOC",      do_kmem,   do_free_kmem);
    g_ktype = KMEM_PAGES;   g_kflags = 0;
    bench_latency("kmem_PAGES",        do_kmem,   do_free_kmem);
    g_ktype = KMEM_VMALLOC; g_kflags = 0;
    bench_latency("kmem_VMALLOC",      do_kmem,   do_free_kmem);
    g_ktype = KMEM_DMA;     g_kflags = 0;
    bench_latency("kmem_DMA",          do_kmem,   do_free_kmem);

    printf("\n=== 2. Fragmentation ===\n");
    bench_fragmentation_malloc();
    bench_fragmentation_kernel(KMEM_KMALLOC, "kmem_KMALLOC");
    bench_fragmentation_kernel(KMEM_PAGES,   "kmem_PAGES");

    printf("\n=== 3/4/5. Page faults / cache / throughput ===\n");
    {
        void *m = malloc(BUF_SIZE);
        run_all("malloc", m, BUF_SIZE); free(m);

        void *mm = mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        run_all("mmap(anon)", mm, BUF_SIZE); munmap(mm, BUF_SIZE);

        void *k1 = kmem_alloc(KMEM_KMALLOC, BUF_SIZE, 0, NULL);
        run_all("kmem_KMALLOC", k1, BUF_SIZE); kmem_free(k1);

        void *k2 = kmem_alloc(KMEM_PAGES, BUF_SIZE, 0, NULL);
        run_all("kmem_PAGES(cached)", k2, BUF_SIZE); kmem_free(k2);

        void *k3 = kmem_alloc(KMEM_PAGES, BUF_SIZE, KMEM_FLAG_UNCACHED, NULL);
        run_all("kmem_PAGES(uncach)", k3, BUF_SIZE); kmem_free(k3);

        void *k4 = kmem_alloc(KMEM_VMALLOC, BUF_SIZE, 0, NULL);
        run_all("kmem_VMALLOC", k4, BUF_SIZE); kmem_free(k4);

        void *k5 = kmem_alloc(KMEM_DMA, BUF_SIZE, 0, NULL);
        run_all("kmem_DMA", k5, BUF_SIZE); kmem_free(k5);
    }

    kmem_lib_fini();
    return 0;
}
