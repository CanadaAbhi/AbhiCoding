static int dev_b_probe(struct platform_device *pdev)
{
    struct pm_lab_dev_b *b = devm_kzalloc(&pdev->dev, sizeof(*b), GFP_KERNEL);
    struct device *dev_a = pm_lab_find_dev_a();     /* lookup by phandle in DT in the real version */

    b->link = device_link_add(&pdev->dev, dev_a,
                               DL_FLAG_PM_RUNTIME | DL_FLAG_RPM_ACTIVE);
    /* DL_FLAG_RPM_ACTIVE: whenever Device B is runtime-resumed,
     * the PM core automatically pm_runtime_get()s Device A FIRST
     * and won't let A suspend again while B is active. */

    b->clk    = devm_clk_get(&pdev->dev, "core");
    b->supply = devm_regulator_get(&pdev->dev, "vdd");

    pm_runtime_set_autosuspend_delay(&pdev->dev, 500);
    pm_runtime_use_autosuspend(&pdev->dev);
    pm_runtime_enable(&pdev->dev);
    return 0;
}
