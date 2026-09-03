#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<semaphore.h>

void *thread_fun(void *arg);

sem_t bin_sem;
char work_area[1024];
int main()
{
	int res;
	pthread_t a_thread;
	void *thread_result;
	
	res=sem_init(&bin_sem,0,0);
	res=pthread_create(&a_thread,NULL,thread_fun,NULL);
	
	printf("input some text enter end to finish\n");
	while(strncmp("end",work_area,3) !=0)
	{
		fgets(work_area,1024,stdin);
		sem_post(&bin_sem);
	}
	printf("waiting for thread to finish\n");
	res=pthread_join(a_thread,&thread_result);
	printf("thread joined , it returned %s\n",(char *)thread_result);
	sem_destroy(&bin_sem);
	exit(0);
}

void *thread_fun(void *arg)
	{
		sem_wait(&bin_sem);
	while(strncmp("end",work_area,3) !=0)
	{
		printf("you entered %d characters \n",strlen(work_area) -1);	
		sem_wait(&bin_sem);
	}
	
	pthread_exit("thank you");
}
