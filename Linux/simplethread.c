/***************************************************************************
	Description : Program to create and join newly created thread to main thread.
		sending parameter to thread and receving data from thread.

	Compile and link this program using 'lpthread' library 
	Thread lib functions used :

	pthread_create(pthread_t *third_id,pthread_attr_t *attr,\
	 void *(*start_routine)(void  *),void  *arg);

	pthread_join(pthread_t thread_id, void **thread_return);

	pthread_exit(void *return_value);

***************************************************************************/
#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<pthread.h>
#include<string.h>

void *thread_fun(void *arg);

char message[]="hello world";

int main()
{
	int res;
	pthread_t a_thread;
	void *thread_result;

	res=pthread_create(&a_thread,NULL,thread_fun,(void *)message);
	if(res !=0){
		perror("unable to create thread\n");
		exit(1);
	}
	printf("waiting for thread to finish\n");
	//Thread joining, catch exit value from the thread	
	res=pthread_join(a_thread,&thread_result);
	
	if(res !=0){
		perror("unable to join thread\n");
		exit(1);
	}
	
	printf("thread joined , it returned %s\n",(char *)thread_result);
	printf("Message is now %s\n",message);
	exit(0);
}

void *thread_fun(void *arg)
{
	printf("thread fun ,arg is %s\n",(char *)arg);
	sleep(3);
	strcpy(message,"bye");
	//exit with return value
	pthread_exit("thank you");
}
