#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<semaphore.h>

void *thread_fun(void *arg);

pthread_mutex_t work_mutex;

char work_area[1024];
int time_to_exit=0;
int main()
{
	int res;
	pthread_t a_thread;
	void *thread_result;
	
	res=pthread_mutex_init(&work_mutex,NULL);//initialize mutex  default attr
	res=pthread_create(&a_thread,NULL,thread_fun,NULL);
	pthread_mutex_lock(&work_mutex); //put a lock to the main thread, then enjoy
	
	printf("input some text enter end to finish\n");
	while(!time_to_exit)	{
		fgets(work_area,24,stdin);

		 //unlock the main thread,your subordinate is waiting 
		pthread_mutex_unlock(&work_mutex); 
		while(1){
			pthread_mutex_lock(&work_mutex);//lock it is your turn
			if(work_area[0] != '\0') {
				pthread_mutex_unlock(&work_mutex);
				sleep(1);
			}
			else
				break;

		}
	}
	
	pthread_mutex_unlock(&work_mutex);
	printf("waiting for thread to finish\n");
	res=pthread_join(a_thread,&thread_result);
	printf("thread joined , it returned %s\n",(char *)thread_result);
	pthread_mutex_destroy(&work_mutex);
	exit(0);
}

void *thread_fun(void *arg)
{
	sleep(1);//Sleep well Let main thread send some data
	pthread_mutex_lock(&work_mutex);//lock the curr thread
	while(strncmp("end",work_area,3) !=0)
	{
		printf("you entered %d characters \n",strlen(work_area) -1);	
		work_area[0]='\0';
		pthread_mutex_unlock(&work_mutex);//unlock the current thread
		sleep(1);//Sleep well , Let main thread do it's job
		pthread_mutex_lock(&work_mutex);
		while(work_area[0] == '\0')
		{
			pthread_mutex_unlock(&work_mutex);
			sleep(1);
			pthread_mutex_lock(&work_mutex);
		}
	}
	time_to_exit=1;
	work_area[0]='\0';	
	pthread_mutex_unlock(&work_mutex);

	pthread_exit("thank you");

}//End of the function
