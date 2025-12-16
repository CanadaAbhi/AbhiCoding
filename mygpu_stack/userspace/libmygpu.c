#include "libmygpu.h"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <errno.h>

/*
 * Educational userspace GPU library
 * Talks to kernel driver via:
 *  - open()   -> render node
 *  - mmap()   -> shared buffer
 *  - write()  -> command submission
 */

static int mygpu_fd = -1;

/* ------------------------------------------------------------ */
/* Open GPU device                                              */
/* ------------------------------------------------------------ */

int mygpu_open(void)
{
    if (mygpu_fd >= 0)
        return mygpu_fd;

    mygpu_fd = open("/dev/dri/renderD128", O_RDWR);
    if (mygpu_fd < 0) {
        perror("mygpu_open: open");
        return -1;
    }

    return mygpu_fd;
}

/* ------------------------------------------------------------ */
/* Allocate GPU buffer (educational mmap)                       */
/* ------------------------------------------------------------ */

void *mygpu_alloc(size_t size)
{
    void *addr;

    if (mygpu_fd < 0) {
        errno = EBADF;
        return MAP_FAILED;
    }

    addr = mmap(NULL,
                size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                mygpu_fd,
                0);

    if (addr == MAP_FAILED) {
        perror("mygpu_alloc: mmap");
        return MAP_FAILED;
    }

    memset(addr, 0, size);
    return addr;
}

/* ------------------------------------------------------------ */
/* Submit command to GPU                                        */
/* ------------------------------------------------------------ */

int mygpu_submit(const void *cmd, size_t size)
{
    ssize_t ret;

    if (mygpu_fd < 0)
        return -1;

    ret = write(mygpu_fd, cmd, size);
    if (ret < 0) {
        perror("mygpu_submit: write");
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------ */
/* Close GPU device                                             */
/* ------------------------------------------------------------ */

void mygpu_close(void)
{
    if (mygpu_fd >= 0) {
        close(mygpu_fd);
        mygpu_fd = -1;
    }
}
