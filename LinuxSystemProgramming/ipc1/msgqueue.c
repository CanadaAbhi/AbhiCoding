/*
A new message queue is created if key has the value IPC_PRIVATE or key isn't IPC_PRIVATE, no message queue with the given  key  key  exists,  and IPC_CREAT is specified in msgflg.

*/
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>

int main()
{
	key_t key;
	int msgid;

	key = ftok("5.c", 64);
	if (key == -1) {
		perror("ftok failed");
		exit(1);
	}
	printf("key:%x\n", key);
	msgid = msgget(key, IPC_CREAT| S_IRUSR | S_IWUSR);
	if (msgid == -1) {
		perror("msgget failed");
		printf("errno:%d\n", errno);
		//Run with different user
		if (errno == EACCES)
			printf(" A  message queue exists for key, but the calling process does not have permission to access the queue\n");
		exit(2);
	}
	printf("msgid:%u\n", msgid);


	return 0;
}