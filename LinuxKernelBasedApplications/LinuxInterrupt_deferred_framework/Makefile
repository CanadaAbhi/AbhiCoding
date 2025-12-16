obj-m += hw_event_gen.o
obj-m += irq_defer_lab.o

KDIR := /lib/modules/$(shell uname -r)/build
all:
	make -C $(KDIR) M=$(PWD) modules
clean:
	make -C $(KDIR) M=$(PWD) clean

# defer_mode: 0=workqueue 1=tasklet 2=threaded_irq
load:
	sudo insmod hw_event_gen.ko
	sudo insmod irq_defer_lab.ko defer_mode=$(MODE)
unload:
	sudo rmmod irq_defer_lab hw_event_gen
