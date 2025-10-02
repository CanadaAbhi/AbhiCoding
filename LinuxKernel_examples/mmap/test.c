#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>

#define DEVNAME "/dev/mydev"

int main(int argc, char **argv)
{
	int fd, status, offset;
	char text[4096];
	void *ptr;

	if(argc < 2) {
		printf("Usage: %s [m,r,w,p] {data}\n", argv[0]);
		return 0;
	}

	fd = open(DEVNAME, O_RDWR);
	if(fd < 0) {
		perror("open");
		return 1;
	}

	switch(argv[1][0]) {
		case 'r':
			status = read(fd, text, 4096);
			printf("READ: I got %d bytes with '%s'\n", status, text);
			break;
		case 'w':
			if(argc < 3) {
				printf("Usage: %s w [data]\n", argv[0]);
				break;
			}

			memset(text, 0, 4096);
			strcpy(text, argv[2]);

			status = write(fd, text, 4096);
			printf("Wrote %d bytes\n", status);
			break;
		case 'm':
			ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if(ptr == MAP_FAILED) {
				perror("mmap");
				break;
			}

			if(argc > 2) {
				memset(text, 0, 4096);
				strcpy(text, argv[2]);
				memcpy(ptr, text, 4096);
			}

			memset(text, 0, 4096);
			memcpy(text, ptr, 4096);
			printf("MMAP: I got '%s'\n", text);
			munmap(ptr, 4096);
			break;
		case 'p':
			if(argc < 3) {
				printf("Usage: %s p [offset]\n", argv[0]);
				break;
			}
			offset = atoi(argv[2]);

			ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if(ptr == MAP_FAILED) {
				perror("mmap");
				break;
			}

			printf("MMAP: Byte on Offset %d: *(ptr + %d) = '%c'\n", offset, offset,
				       	*((char *) (ptr) + offset));

			munmap(ptr, 4096);
			break;
		default:
			printf("'%c' is invalid.\n", argv[1][0]);
			break;
	}

	close(fd);
}
