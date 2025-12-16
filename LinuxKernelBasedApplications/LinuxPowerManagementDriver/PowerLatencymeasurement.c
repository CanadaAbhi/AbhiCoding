enum pm_lab_state { PM_STATE_ACTIVE, PM_STATE_IDLE, PM_STATE_SUSPENDED, PM_STATE_OFF };

static const u32 sim_power_mw[] = {
    [PM_STATE_ACTIVE]    = 45,
    [PM_STATE_IDLE]       = 12,
    [PM_STATE_SUSPENDED] = 3,
    [PM_STATE_OFF]       = 0,
};

struct pm_lab_stats {
    ktime_t   state_entered;
    u64       time_in_state_ns[4];
    u64       energy_uj_accum;          /* += mw * dt, simulated */
    struct latency_hist { u64 min, max, sum, count; u64 samples[128]; } suspend_lat, resume_lat;
};

void pm_lab_stats_transition(struct device *dev, enum pm_lab_state new_state)
{
    struct pm_lab_stats *s = dev_get_pm_stats(dev);
    ktime_t now = ktime_get();
    u64 dt_ns = ktime_to_ns(ktime_sub(now, s->state_entered));

    s->time_in_state_ns[s->cur_state] += dt_ns;
    s->energy_uj_accum += (sim_power_mw[s->cur_state] * dt_ns) / 1000000; /* mW*ns -> uJ */
    s->cur_state = new_state;
    s->state_entered = now;
}
