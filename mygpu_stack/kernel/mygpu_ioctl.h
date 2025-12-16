#pragma once
#include <drm/drm.h>

#define MYGPU_IOCTL_BASE 'M'

struct mygpu_gem_create {
    __u64 size;
    __u32 handle;
};

#define MYGPU_IOCTL_GEM_CREATE \
    DRM_IOWR(MYGPU_IOCTL_BASE, 0x00, struct mygpu_gem_create)
