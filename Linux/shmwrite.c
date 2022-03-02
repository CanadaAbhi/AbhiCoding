/* This program illustrates how a shared memory segment attatched and how data is written to the shared memory segment*/

#include<sys/ipc.h>
#include<sys/shm.h>
#include<stdio.h>

main()
{
	int shmid,pos;
	char *msg;
	
	if((shmid=shmget(110,1024,IPC_CREAT|0666))==-1) {
		perror("shmget");
		exit(1);
	}
	
	msg=shmat(shmid,0,0);
	
	printf("Enter the data you want to write into shared memory\n");
	fgets(msg,1024,stdin);
	pos = strlen(msg);
	strcpy(msg+pos-1,"World");
	printf("Data successfully written\n");
	
	shmdt(msg);
}
