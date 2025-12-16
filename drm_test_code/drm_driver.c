/*
 * Simple DRM Test Driver - Complete and Corrected
 * Demonstrates basic DRM subsystem usage with atomic modesetting
 */

 #include <linux/module.h>
 #include <linux/platform_device.h>
 #include <linux/of.h>
 #include <drm/drm_drv.h>
 #include <drm/drm_device.h>
 #include <drm/drm_file.h>
 #include <drm/drm_ioctl.h>
 #include <drm/drm_mode_config.h>
 #include <drm/drm_crtc.h>
 #include <drm/drm_plane.h>
 #include <drm/drm_encoder.h>
 #include <drm/drm_connector.h>
 #include <drm/drm_atomic.h>
 #include <drm/drm_atomic_helper.h>
 #include <drm/drm_probe_helper.h>
 #include <drm/drm_gem_framebuffer_helper.h>
 #include <drm/drm_fb_helper.h>
 #include <drm/drm_gem.h>
 #include <drm/drm_gem_cma_helper.h>
 #include <drm/drm_vblank.h>
 #include <drm/drm_modes.h>
 
 /* Driver private data structure */
 struct simple_drm {
     struct drm_device *drm;
     struct drm_plane *primary;
     struct drm_crtc *crtc;
     struct drm_encoder *encoder;
     struct drm_connector *connector;
 };
 
 /* ============================================================================
  * Plane Functions
  * ============================================================================ */
 
 static const uint32_t simple_formats[] = {
     DRM_FORMAT_XRGB8888,
     DRM_FORMAT_ARGB8888,
     DRM_FORMAT_RGB565,
 };
 
 static int simple_plane_atomic_check(struct drm_plane *plane,
                       struct drm_atomic_state *state)
 {
     struct drm_plane_state *new_plane_state;
     struct drm_crtc_state *crtc_state;
 
     new_plane_state = drm_atomic_get_new_plane_state(state, plane);
 
     if (!new_plane_state->crtc)
         return 0;
 
     crtc_state = drm_atomic_get_new_crtc_state(state, new_plane_state->crtc);
     if (!crtc_state)
         return -EINVAL;
 
     return drm_atomic_helper_check_plane_state(new_plane_state, crtc_state,
                            DRM_PLANE_NO_SCALING,
                            DRM_PLANE_NO_SCALING,
                            false, false);
 }
 
 static void simple_plane_atomic_update(struct drm_plane *plane,
                     struct drm_atomic_state *state)
 {
     struct drm_plane_state *new_state;
 
     new_state = drm_atomic_get_new_plane_state(state, plane);
 
     if (!new_state->fb)
         return;
 
     /* Hardware would be programmed here */
     drm_dbg(plane->dev, "Plane update: fb=%p\n", new_state->fb);
 }
 
 static const struct drm_plane_helper_funcs simple_plane_helper_funcs = {
     .atomic_check = simple_plane_atomic_check,
     .atomic_update = simple_plane_atomic_update,
 };
 
 static const struct drm_plane_funcs simple_plane_funcs = {
     .update_plane = drm_atomic_helper_update_plane,
     .disable_plane = drm_atomic_helper_disable_plane,
     .destroy = drm_plane_cleanup,
     .reset = drm_atomic_helper_plane_reset,
     .atomic_duplicate_state = drm_atomic_helper_plane_duplicate_state,
     .atomic_destroy_state = drm_atomic_helper_plane_destroy_state,
 };
 
 static int simple_plane_init(struct drm_device *drm,
                   struct drm_plane **plane_out)
 {
     struct drm_plane *plane;
     int ret;
 
     plane = kzalloc(sizeof(*plane), GFP_KERNEL);
     if (!plane)
         return -ENOMEM;
 
     ret = drm_universal_plane_init(drm, plane, 0,
                        &simple_plane_funcs,
                        simple_formats,
                        ARRAY_SIZE(simple_formats),
                        NULL,
                        DRM_PLANE_TYPE_PRIMARY,
                        NULL);
     if (ret) {
         kfree(plane);
         return ret;
     }
 
     drm_plane_helper_add(plane, &simple_plane_helper_funcs);
     *plane_out = plane;
 
     return 0;
 }
 
 /* ============================================================================
  * CRTC Functions
  * ============================================================================ */
 
 static int simple_crtc_atomic_check(struct drm_crtc *crtc,
                      struct drm_atomic_state *state)
 {
     return 0;
 }
 
 static void simple_crtc_atomic_begin(struct drm_crtc *crtc,
                       struct drm_atomic_state *state)
 {
     drm_dbg(crtc->dev, "CRTC atomic begin\n");
 }
 
 static void simple_crtc_atomic_flush(struct drm_crtc *crtc,
                       struct drm_atomic_state *state)
 {
     drm_dbg(crtc->dev, "CRTC atomic flush\n");
 }
 
 static void simple_crtc_atomic_enable(struct drm_crtc *crtc,
                        struct drm_atomic_state *state)
 {
     drm_dbg(crtc->dev, "CRTC enable\n");
     drm_crtc_vblank_on(crtc);
 }
 
 static void simple_crtc_atomic_disable(struct drm_crtc *crtc,
                     struct drm_atomic_state *state)
 {
     drm_crtc_vblank_off(crtc);
     drm_dbg(crtc->dev, "CRTC disable\n");
 }
 
 static const struct drm_crtc_helper_funcs simple_crtc_helper_funcs = {
     .atomic_check = simple_crtc_atomic_check,
     .atomic_begin = simple_crtc_atomic_begin,
     .atomic_flush = simple_crtc_atomic_flush,
     .atomic_enable = simple_crtc_atomic_enable,
     .atomic_disable = simple_crtc_atomic_disable,
 };
 
 static const struct drm_crtc_funcs simple_crtc_funcs = {
     .reset = drm_atomic_helper_crtc_reset,
     .destroy = drm_crtc_cleanup,
     .set_config = drm_atomic_helper_set_config,
     .page_flip = drm_atomic_helper_page_flip,
     .atomic_duplicate_state = drm_atomic_helper_crtc_duplicate_state,
     .atomic_destroy_state = drm_atomic_helper_crtc_destroy_state,
 };
 
 static int simple_crtc_init(struct drm_device *drm,
                  struct drm_plane *primary,
                  struct drm_crtc **crtc_out)
 {
     struct drm_crtc *crtc;
     int ret;
 
     crtc = kzalloc(sizeof(*crtc), GFP_KERNEL);
     if (!crtc)
         return -ENOMEM;
 
     ret = drm_crtc_init_with_planes(drm, crtc, primary, NULL,
                     &simple_crtc_funcs, NULL);
     if (ret) {
         kfree(crtc);
         return ret;
     }
 
     drm_crtc_helper_add(crtc, &simple_crtc_helper_funcs);
     *crtc_out = crtc;
 
     return 0;
 }
 
 /* ============================================================================
  * Encoder Functions
  * ============================================================================ */
 
 static const struct drm_encoder_funcs simple_encoder_funcs = {
     .destroy = drm_encoder_cleanup,
 };
 
 static int simple_encoder_init(struct drm_device *drm,
                 struct drm_encoder **encoder_out)
 {
     struct drm_encoder *encoder;
     int ret;
 
     encoder = kzalloc(sizeof(*encoder), GFP_KERNEL);
     if (!encoder)
         return -ENOMEM;
 
     ret = drm_encoder_init(drm, encoder, &simple_encoder_funcs,
                    DRM_MODE_ENCODER_VIRTUAL, NULL);
     if (ret) {
         kfree(encoder);
         return ret;
     }
 
     encoder->possible_crtcs = 1;
     *encoder_out = encoder;
 
     return 0;
 }
 
 /* ============================================================================
  * Connector Functions
  * ============================================================================ */
 
 static int simple_connector_get_modes(struct drm_connector *connector)
 {
     struct drm_display_mode *mode;
     int count = 0;
 
     /* Add a default 1920x1080@60 mode */
     mode = drm_mode_create(connector->dev);
     if (!mode)
         return 0;
 
     mode->type = DRM_MODE_TYPE_DRIVER | DRM_MODE_TYPE_PREFERRED;
     mode->clock = 148500;
     mode->hdisplay = 1920;
     mode->hsync_start = 2008;
     mode->hsync_end = 2052;
     mode->htotal = 2200;
     mode->vdisplay = 1080;
     mode->vsync_start = 1084;
     mode->vsync_end = 1089;
     mode->vtotal = 1125;
     mode->flags = DRM_MODE_FLAG_PHSYNC | DRM_MODE_FLAG_PVSYNC;
 
     drm_mode_set_name(mode);
     drm_mode_probed_add(connector, mode);
     count++;
 
     return count;
 }
 
 static enum drm_mode_status
 simple_connector_mode_valid(struct drm_connector *connector,
                  struct drm_display_mode *mode)
 {
     /* Accept all modes for simplicity */
     return MODE_OK;
 }
 
 static const struct drm_connector_helper_funcs simple_connector_helper_funcs = {
     .get_modes = simple_connector_get_modes,
     .mode_valid = simple_connector_mode_valid,
 };
 
 static const struct drm_connector_funcs simple_connector_funcs = {
     .reset = drm_atomic_helper_connector_reset,
     .fill_modes = drm_helper_probe_single_connector_modes,
     .destroy = drm_connector_cleanup,
     .atomic_duplicate_state = drm_atomic_helper_connector_duplicate_state,
     .atomic_destroy_state = drm_atomic_helper_connector_destroy_state,
 };
 
 static int simple_connector_init(struct drm_device *drm,
                   struct drm_encoder *encoder,
                   struct drm_connector **connector_out)
 {
     struct drm_connector *connector;
     int ret;
 
     connector = kzalloc(sizeof(*connector), GFP_KERNEL);
     if (!connector)
         return -ENOMEM;
 
     ret = drm_connector_init(drm, connector, &simple_connector_funcs,
                  DRM_MODE_CONNECTOR_VIRTUAL);
     if (ret) {
         kfree(connector);
         return ret;
     }
 
     drm_connector_helper_add(connector, &simple_connector_helper_funcs);
     drm_connector_attach_encoder(connector, encoder);
 
     *connector_out = connector;
     return 0;
 }
 
 /* ============================================================================
  * Mode Config Functions
  * ============================================================================ */
 
 static const struct drm_mode_config_funcs simple_mode_config_funcs = {
     .fb_create = drm_gem_fb_create,
     .atomic_check = drm_atomic_helper_check,
     .atomic_commit = drm_atomic_helper_commit,
 };
 
 static int simple_modeset_init(struct drm_device *drm)
 {
     struct simple_drm *simple = drm->dev_private;
     int ret;
 
     ret = drmm_mode_config_init(drm);
     if (ret)
         return ret;
 
     drm->mode_config.min_width = 320;
     drm->mode_config.min_height = 200;
     drm->mode_config.max_width = 1920;
     drm->mode_config.max_height = 1200;
     drm->mode_config.funcs = &simple_mode_config_funcs;
 
     /* Create plane */
     ret = simple_plane_init(drm, &simple->primary);
     if (ret)
         return ret;
 
     /* Create CRTC */
     ret = simple_crtc_init(drm, simple->primary, &simple->crtc);
     if (ret)
         return ret;
 
     /* Create encoder */
     ret = simple_encoder_init(drm, &simple->encoder);
     if (ret)
         return ret;
 
     /* Create connector */
     ret = simple_connector_init(drm, simple->encoder, &simple->connector);
     if (ret)
         return ret;
 
     drm_mode_config_reset(drm);
 
     return 0;
 }
 
 /* ============================================================================
  * DRM Driver
  * ============================================================================ */
 
 DEFINE_DRM_GEM_CMA_FOPS(simple_drm_fops);
 
 static const struct drm_driver simple_drm_driver = {
     .driver_features = DRIVER_MODESET | DRIVER_GEM | DRIVER_ATOMIC,
     .fops = &simple_drm_fops,
     .name = "simple-drm",
     .desc = "Simple DRM Test Driver",
     .date = "20250207",
     .major = 1,
     .minor = 0,
     .patchlevel = 0,
 
     DRM_GEM_CMA_DRIVER_OPS,
 };
 
 /* ============================================================================
  * Platform Driver
  * ============================================================================ */
 
 static int simple_drm_probe(struct platform_device *pdev)
 {
     struct simple_drm *simple;
     struct drm_device *drm;
     int ret;
 
     simple = devm_kzalloc(&pdev->dev, sizeof(*simple), GFP_KERNEL);
     if (!simple)
         return -ENOMEM;
 
     drm = drm_dev_alloc(&simple_drm_driver, &pdev->dev);
     if (IS_ERR(drm))
         return PTR_ERR(drm);
 
     drm->dev_private = simple;
     simple->drm = drm;
     platform_set_drvdata(pdev, drm);
 
     ret = simple_modeset_init(drm);
     if (ret)
         goto err_put;
 
     ret = drm_dev_register(drm, 0);
     if (ret)
         goto err_put;
 
     drm_fbdev_generic_setup(drm, 32);
 
     dev_info(&pdev->dev, "Simple DRM driver loaded successfully\n");
     return 0;
 
 err_put:
     drm_dev_put(drm);
     return ret;
 }
 
 static int simple_drm_remove(struct platform_device *pdev)
 {
     struct drm_device *drm = platform_get_drvdata(pdev);
 
     drm_dev_unregister(drm);
     drm_atomic_helper_shutdown(drm);
     drm_dev_put(drm);
 
     return 0;
 }
 
 static const struct of_device_id simple_drm_of_match[] = {
     { .compatible = "simple,drm" },
     { /* sentinel */ }
 };
 MODULE_DEVICE_TABLE(of, simple_drm_of_match);
 
 static struct platform_driver simple_drm_platform_driver = {
     .probe = simple_drm_probe,
     .remove = simple_drm_remove,
     .driver = {
         .name = "simple-drm",
         .of_match_table = simple_drm_of_match,
     },
 };
 
 module_platform_driver(simple_drm_platform_driver);
 
 MODULE_LICENSE("GPL");
 MODULE_AUTHOR("DRM Example");
 MODULE_DESCRIPTION("Simple DRM Test Driver");
 MODULE_ALIAS("platform:simple-drm");