obj-m += my_mmap.o

all: module testapp

module:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm a.out

testapp: test.c
	gcc test.c

	
