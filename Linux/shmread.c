/* This program illustrates how a data is read from a shared memory segment*/
#include<sys/ipc.h>
#include<sys/shm.h>
#include<stdio.h>

main()
{
	int shmid;
	char *msg;
	
	if((shmid=shmget(110,1024,0666|IPC_CREAT))==-1) {
		perror("shmget");
		exit(1);
	}
	
	msg=shmat(shmid,0,0);
	printf("Data written in the shared memory is:%s\n",msg);
	
	shmdt(msg);
}
	
	
