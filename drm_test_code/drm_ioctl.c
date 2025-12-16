#include <drm/drm_drv.h>
#include <drm/drm_ioctl.h>
#include <drm/drm_file.h>
#include <drm/drm_mode.h>
#include <drm/drm_gem.h>
#include <drm/drm_prime.h>


static const struct drm_ioctl_desc my_driver_ioctls[] = {
    DRM_IOCTL_DEF_DRV(VERSION, drm_version, DRM_AUTH | DRM_RENDER_ALLOW),
    DRM_IOCTL_DEF_DRV(GET_UNIQUE, drm_getunique, DRM_AUTH | DRM_RENDER_ALLOW),
    DRM_IOCTL_DEF_DRV(MODE_GETRESOURCES, drm_mode_getresources, DRM_AUTH | DRM_RENDER_ALLOW),
    DRM_IOCTL_DEF_DRV(MODE_GETCONNECTOR, drm_mode_getconnector, DRM_AUTH | DRM_RENDER
