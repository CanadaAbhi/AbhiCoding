                         Linux
                           |
                 sensor_fw_core.ko  (class "sensor_fw", common chardev+sysfs ABI)
                           |
              +------------+------------+
              |                         |
        fake_temp_i2c.ko          fake_imu_spi.ko
        (I2C, GPIO IRQ,           (SPI, GPIO IRQ,
         threaded IRQ,             threaded IRQ,
         workqueue, PM)            workqueue, PM)
              |                         |
        Temperature sensor         6-axis IMU
              |                         |
              +------------+------------+
                           |
                /dev/sensor_fw/temp0
                /dev/sensor_fw/imu_spi0
                           |
                    sensor_daemon
                     /          \
             monitoring_ui    MQTT broker -> cloud_subscriber
             (local UNIX               (acme/sensors/*, acme/alarms/*)
              socket dashboard)
