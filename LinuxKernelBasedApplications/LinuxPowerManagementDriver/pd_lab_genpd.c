struct pd_lab {
    struct generic_pm_domain genpd;
    struct regulator *rail;     /* -> qc_bringup RPMh-voting regulator */
    struct clk       *gcc_clk;  /* -> qc_bringup GCC branch clock */
    ktime_t           t_off_start;
};

static int pd_lab_power_off(struct generic_pm_domain *domain)
{
    struct pd_lab *pd = to_pd_lab(domain);
    ktime_t t0 = ktime_get();

    clk_disable_unprepare(pd->gcc_clk);        /* gate the branch clock first */
    regulator_disable(pd->rail);               /* -> retracts this subsystem's RPMh vote;
                                                    PMIC rail only actually powers down
                                                    once ALL votes are retracted */
    pd->t_off_start = ktime_get();
    pm_lab_stats_record_latency(PD_OFF, ktime_sub(ktime_get(), t0));
    return 0;
}

static int pd_lab_power_on(struct generic_pm_domain *domain)
{
    struct pd_lab *pd = to_pd_lab(domain);
    ktime_t t0 = ktime_get();
    int ret;

    ret = regulator_enable(pd->rail);           /* casts this subsystem's vote */
    if (ret) return ret;
    ret = clk_prepare_enable(pd->gcc_clk);
    if (pd->t_off_start)
        pm_lab_stats_record_off_duration(ktime_sub(t0, pd->t_off_start));
    pm_lab_stats_record_latency(PD_ON, ktime_sub(ktime_get(), t0));
    return ret;
}

static int __init pd_lab_init(void)
{
    struct pd_lab *pd = kzalloc(sizeof(*pd), GFP_KERNEL);
    pd->genpd.name = "pd_lab0";
    pd->genpd.power_off = pd_lab_power_off;
    pd->genpd.power_on  = pd_lab_power_on;
    pd->rail    = regulator_get(NULL, "rpmh_rail0");   /* qc_bringup's RPMh driver */
    pd->gcc_clk = clk_get(NULL, "gcc_branch0");         /* qc_bringup's GCC driver */
    pm_genpd_init(&pd->genpd, NULL, true /* start powered off */);
    of_genpd_add_provider_simple(pm_lab_of_node, &pd->genpd);
    return 0;
}
