// SPDX-License-Identifier: GPL-2.0
#include <drm/drm_ioctl.h>
#include <drm/drm_gem.h>

#include "mygpu_ioctl.h"
#include "mygpu_gem.h"

static int mygpu_ioctl_gem_create(struct drm_device *dev,
                                 void *data,
                                 struct drm_file *file)
{
    struct mygpu_gem_create *args = data;
    struct drm_gem_object *obj;
    int ret;

    obj = mygpu_gem_create(dev, args->size);
    if (IS_ERR(obj))
        return PTR_ERR(obj);

    ret = drm_gem_handle_create(file, obj, &args->handle);
    drm_gem_object_put(obj);

    return ret;
}

static const struct drm_ioctl_desc mygpu_ioctls[] = {
    DRM_IOCTL_DEF_DRV(MYGPU_IOCTL_GEM_CREATE,
                      mygpu_ioctl_gem_create,
                      DRM_RENDER_ALLOW),
};

int mygpu_ioctl_register(struct drm_device *dev)
{
    dev->driver->ioctls = mygpu_ioctls;
    dev->driver->num_ioctls = ARRAY_SIZE(mygpu_ioctls);
    return 0;
}
