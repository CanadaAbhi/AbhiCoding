#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>

int main()
{
	int filedes[2];
	char buffer1[] = "hello pipe1";
	char buffer2[] = "hello pipe2";
	char readbuffer[30] = {};

	if (pipe(filedes) == 0) {
		printf("Pipe successful\n");
		printf("read from %d, write to %d\n", filedes[0], filedes[1]);
		while(1) {
			write(filedes[1], buffer1, sizeof(buffer1));
			perror("write");
			write(filedes[1], buffer2, sizeof(buffer2));
			perror("write");
		}
		read(filedes[0], readbuffer, sizeof(buffer1));
		printf("read:%s\n", readbuffer);
		read(filedes[0], readbuffer, sizeof(buffer2));
		printf("read:%s\n", readbuffer);
		close(filedes[1]);
		close(filedes[0]);
	} else {
		perror("pipe failed");
	}
	return 0;
}