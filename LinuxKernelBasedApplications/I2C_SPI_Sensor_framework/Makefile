obj-m += sensor_fw_core.o
obj-m += fake_temp_i2c.o
obj-m += fake_imu_spi.o

KDIR := /lib/modules/$(shell uname -r)/build
all:
	make -C $(KDIR) M=$(PWD) modules
clean:
	make -C $(KDIR) M=$(PWD) clean
load:
	sudo insmod sensor_fw_core.ko
	sudo insmod fake_temp_i2c.ko
	sudo insmod fake_imu_spi.ko
unload:
	sudo rmmod fake_imu_spi fake_temp_i2c sensor_fw_core
