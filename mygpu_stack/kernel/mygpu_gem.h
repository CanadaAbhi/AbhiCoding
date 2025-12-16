#pragma once
#include <drm/drm_gem.h>

struct drm_device;

struct drm_gem_object *
mygpu_gem_create(struct drm_device *dev, size_t size);

