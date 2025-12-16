// SPDX-License-Identifier: GPL-2.0
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>

#include <drm/drm_gem.h>
#include <drm/drm_device.h>

#include "mygpu_gem.h"

/* ------------------------------------------------------------------ */
/* GEM object                                                          */
/* ------------------------------------------------------------------ */

struct mygpu_gem {
    struct drm_gem_object base;
    void *vaddr;
};

#define to_mygpu_gem(x) container_of(x, struct mygpu_gem, base)

/* ------------------------------------------------------------------ */
/* GEM object lifecycle                                                */
/* ------------------------------------------------------------------ */

static void mygpu_gem_free_object(struct drm_gem_object *obj)
{
    struct mygpu_gem *mgem = to_mygpu_gem(obj);

    if (mgem->vaddr)
        vfree(mgem->vaddr);

    drm_gem_object_release(obj);
    kfree(mgem);
}

/* ------------------------------------------------------------------ */
/* GEM object funcs                                                    */
/* ------------------------------------------------------------------ */

static const struct drm_gem_object_funcs mygpu_gem_object_funcs = {
    .free = mygpu_gem_free_object,
};

/* ------------------------------------------------------------------ */
/* Public GEM create API                                               */
/* ------------------------------------------------------------------ */

struct drm_gem_object *
mygpu_gem_create(struct drm_device *dev, size_t size)
{
    struct mygpu_gem *mgem;
    int ret;

    size = PAGE_ALIGN(size);

    mgem = kzalloc(sizeof(*mgem), GFP_KERNEL);
    if (!mgem)
        return ERR_PTR(-ENOMEM);

    ret = drm_gem_object_init(dev, &mgem->base, size);
    if (ret) {
        kfree(mgem);
        return ERR_PTR(ret);
    }

    mgem->base.funcs = &mygpu_gem_object_funcs;

    mgem->vaddr = vzalloc(size);
    if (!mgem->vaddr) {
        drm_gem_object_release(&mgem->base);
        kfree(mgem);
        return ERR_PTR(-ENOMEM);
    }

    return &mgem->base;
}
