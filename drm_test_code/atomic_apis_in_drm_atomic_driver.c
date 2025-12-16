#include <drm/drm_atomic.h>
#include <drm/drm_atomic_helper.h>
#include <drm/drm_plane.h>
#include <drm/drm_crtc.h>
#include <drm/drm_connector.h>

// Sample code for learning , need to add makefile and invoke them calls from other function to prepare a buildable module.

struct drm_atomic_state *state;

state = drm_atomic_state_alloc(dev);
if (!state)
    return -ENOMEM;

/* refcount = 1 */

struct drm_plane_state *plane_state;

plane_state = drm_atomic_get_plane_state(state, plane);
if (IS_ERR(plane_state))
    goto err;

plane_state->crtc = crtc;
plane_state->fb   = fb;
plane_state->crtc_w = 1920;
plane_state->crtc_h = 1080;
plane_state->src_w  = 1920 << 16;
plane_state->src_h  = 1080 << 16;


//crtc state

struct drm_crtc_state *crtc_state;

crtc_state = drm_atomic_get_crtc_state(state, crtc);
if (IS_ERR(crtc_state))
    goto err;

crtc_state->active = true;
//connector state
struct drm_connector_state *conn_state;

conn_state = drm_atomic_get_connector_state(state, connector);
if (IS_ERR(conn_state))
    goto err;

conn_state->crtc = crtc;


ret = drm_atomic_helper_check(dev, state);
if (ret)
    goto err;

    ret = drm_atomic_commit(state);
if (ret)
    goto err;


    static const struct drm_mode_config_funcs my_mode_config_funcs = {
        .atomic_check  = drm_atomic_helper_check,
        .atomic_commit = drm_atomic_helper_commit,
    };

    drm_atomic_state_clear(state);

/* State can now be reused for another transaction */

drm_atomic_state_get(state);

/* later */
drm_atomic_state_put(state);


int my_atomic_commit_example(struct drm_device *dev,
    struct drm_crtc *crtc,
    struct drm_plane *plane,
    struct drm_connector *connector,
    struct drm_framebuffer *fb)
{
struct drm_atomic_state *state;
struct drm_plane_state *pstate;
struct drm_crtc_state *cstate;
struct drm_connector_state *conn_state;
int ret;

state = drm_atomic_state_alloc(dev);
if (!state)
return -ENOMEM;

pstate = drm_atomic_get_plane_state(state, plane);
cstate = drm_atomic_get_crtc_state(state, crtc);
conn_state = drm_atomic_get_connector_state(state, connector);

if (IS_ERR(pstate) || IS_ERR(cstate) || IS_ERR(conn_state)) {
ret = -EINVAL;
goto out;
}

pstate->crtc = crtc;
pstate->fb = fb;

cstate->active = true;
conn_state->crtc = crtc;

ret = drm_atomic_helper_check(dev, state);
if (ret)
goto out;

ret = drm_atomic_commit(state);

out:
drm_atomic_state_put(state);
return ret;
}

// drm_atomic_commit()
//  └─ driver atomic_commit()
//       ├─ drm_atomic_helper_commit_planes()
//       ├─ drm_atomic_helper_swap_state()
//       └─ events / vblank

#include <drm/drm_atomic_helper.h>
#include <drm/drm_plane.h>
#include <drm/drm_crtc.h>

static const struct drm_plane_funcs my_plane_funcs = {
    .reset = drm_atomic_helper_plane_reset,
    .destroy = drm_plane_cleanup,
    .atomic_duplicate_state =
        drm_atomic_helper_plane_duplicate_state,
    .atomic_destroy_state =
        drm_atomic_helper_plane_destroy_state,
};


static const struct drm_crtc_funcs my_crtc_funcs = {
    .reset = drm_atomic_helper_crtc_reset,
    .destroy = drm_crtc_cleanup,
    .atomic_duplicate_state =
        drm_atomic_helper_crtc_duplicate_state,
    .atomic_destroy_state =
        drm_atomic_helper_crtc_destroy_state,
};

drm_mode_config_reset(dev);

for each plane  → plane->funcs->reset()
for each crtc   → crtc->funcs->reset()
for each connector → connector->funcs->reset()

static void my_atomic_commit(struct drm_device *dev,
    struct drm_atomic_state *state,
    bool nonblock)
{
/* Program plane hardware */
drm_atomic_helper_commit_planes(dev, state, nonblock);

/* Optional: CRTC enable/disable, clocks, PHY */
}

static void my_atomic_commit_tail(struct drm_atomic_state *state)
{
    /* Swap old and new atomic state */
    drm_atomic_helper_swap_state(state);

    /* Send vblank / page-flip events */
    drm_atomic_helper_commit_hw_done(state);
    drm_atomic_helper_cleanup_planes(state);
}

static void my_atomic_commit_tail(struct drm_atomic_state *state)
{
    struct drm_device *dev = state->dev;

    /* Program plane registers */
    drm_atomic_helper_commit_planes(dev, state, false);

    /* Make state visible */
    drm_atomic_helper_swap_state(state);

    /* Finish commit */
    drm_atomic_helper_commit_hw_done(state);
    drm_atomic_helper_cleanup_planes(state);
}

static const struct drm_mode_config_funcs my_mode_config_funcs = {
    .atomic_check  = drm_atomic_helper_check,
    .atomic_commit = drm_atomic_helper_commit,
};

