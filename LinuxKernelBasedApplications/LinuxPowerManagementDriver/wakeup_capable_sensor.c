struct pm_lab_dev_a {
    struct clk *clk;
    struct regulator *supply;
    struct wakeup_source *wakeup;
};

static int dev_a_runtime_suspend(struct device *dev)
{
    struct pm_lab_dev_a *a = dev_get_drvdata(dev);
    ktime_t t0 = ktime_get();

    disable_irq(a->irq);
    clk_disable_unprepare(a->clk);
    regulator_disable(a->supply);

    pm_lab_stats_transition(dev, PM_STATE_SUSPENDED);
    pm_lab_stats_record_latency(dev, PM_OP_SUSPEND, ktime_sub(ktime_get(), t0));
    return 0;
}

static int dev_a_runtime_resume(struct device *dev)
{
    struct pm_lab_dev_a *a = dev_get_drvdata(dev);
    ktime_t t0 = ktime_get();

    regulator_enable(a->supply);
    clk_prepare_enable(a->clk);
    udelay(SIMULATED_SPMI_XACT_US);      /* models real PMIC I2C/SPMI transaction cost */
    enable_irq(a->irq);

    pm_lab_stats_transition(dev, PM_STATE_ACTIVE);
    pm_lab_stats_record_latency(dev, PM_OP_RESUME, ktime_sub(ktime_get(), t0));
    return 0;
}

static irqreturn_t dev_a_wake_irq(int irq, void *data)
{
    struct pm_lab_dev_a *a = data;
    pm_wakeup_event(a->dev, 0);   /* tells PM core: this is why we're resuming */
    return IRQ_HANDLED;
}

static int dev_a_probe(struct platform_device *pdev)
{
    struct pm_lab_dev_a *a = devm_kzalloc(&pdev->dev, sizeof(*a), GFP_KERNEL);
    a->clk    = devm_clk_get(&pdev->dev, "core");
    a->supply = devm_regulator_get(&pdev->dev, "vdd");

    device_init_wakeup(&pdev->dev, true);           /* opt this device into wakeup */
    dev_pm_set_wake_irq(&pdev->dev, a->irq);         /* IRQ can resume from suspend */

    pm_runtime_set_autosuspend_delay(&pdev->dev, 200 /* ms */);
    pm_runtime_use_autosuspend(&pdev->dev);
    pm_runtime_enable(&pdev->dev);
    return 0;
}

static const struct dev_pm_ops dev_a_pm_ops = {
    SET_RUNTIME_PM_OPS(dev_a_runtime_suspend, dev_a_runtime_resume, NULL)
    SET_SYSTEM_SLEEP_PM_OPS(dev_a_runtime_suspend, dev_a_runtime_resume)
};
