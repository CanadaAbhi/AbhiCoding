#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
	int fd = open("hello.txt", O_RDWR);
	char *address;

	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}

	address = mmap(NULL, 5, PROT_NONE, MAP_SHARED, fd, 0);
	if (address == MAP_FAILED) {
		perror("mmap");
		exit(EXIT_FAILURE);
	}

	mprotect(address, 5, PROT_WRITE);	
	address[0] = 'B';
	address[1] = 'Y';
	address[2] = 'E';
	munmap(address, 5);
	close(fd);

	return 0;
}	
