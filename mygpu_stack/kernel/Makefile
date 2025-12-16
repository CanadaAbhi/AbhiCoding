# Kernel module name
obj-m := mygpu_drv.o

# Objects linked into the module
mygpu_drv-y := mygpu_drv.o mygpu_gem.o mygpu_scheduler.o mygpu_ioctl.o

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# Default target
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean build artifacts
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
