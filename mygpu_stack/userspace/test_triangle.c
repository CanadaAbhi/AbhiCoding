#include "libmygpu.h"
#include <stdio.h>
#include <sys/mman.h>
int main(void)
{
    const size_t fb_size = 640 * 480 * 4;

    if (mygpu_open() < 0)
        return 1;

    int *fb = mygpu_alloc(fb_size);
    if (fb == MAP_FAILED)
        return 1;

    /* Fake “triangle” draw */
    for (int y = 100; y < 300; y++)
        for (int x = 100; x < 300; x++)
            fb[y * 640 + x] = 0x00FF0000;

    mygpu_submit("draw", 4);

    printf("Triangle rendered (software)\n");
    getchar();

    mygpu_close();
    return 0;
}
