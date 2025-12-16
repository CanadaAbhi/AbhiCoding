# kernel side
cd kernel && make && sudo make load
ls /dev/sensor_fw/            # temp0  imu_spi0
cat /sys/class/sensor_fw/temp0/value_latest
echo 55000 > /sys/class/sensor_fw/temp0/threshold_mC
echo 200   > /sys/class/sensor_fw/imu_spi0/odr_hz

# broker (any local MQTT broker for the demo)
mosquitto -d

# userspace
cd app && make
sudo ./sensor_daemon &
./monitoring_ui              # local dashboard
./cloud_subscriber &         # cloud-side consumer
mosquitto_sub -t 'acme/#' -v # or just tail raw MQTT traffic
