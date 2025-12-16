KDIR := /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

obj-m += accel_drv.o
accel_drv-objs := drv/accel_core.o drv/accel_bufmgr.o drv/accel_cmdq.o \
                  drv/accel_sched.o drv/accel_fence.o drv/accel_irq.o

obj-m += accel_hw_sim.o
accel_hw_sim-objs := sim/accel_hw_sim.o

all: modules user/app

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

user/app: user/app.c user/accel_lib.c
	$(CC) -O2 -Wall -o $@ user/app.c user/accel_lib.c

load:
	insmod accel_hw_sim.ko
	insmod accel_drv.ko
	sleep 1
	chmod 666 /dev/accel0

unload:
	rmmod accel_drv || true
	rmmod accel_hw_sim || true

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f user/app
