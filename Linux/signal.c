#include <stdio.h>
#include <stdlib.h>
# include <sys/types.h>
# include <pthread.h>
# include <signal.h>

void * th_fn1(void * p);
void * th_fn2(void * p);

void sigfn (int k);

int main()
{
        pthread_t t1,t2,t3,t4;
        int pid = getpid();

        system("clear");
        //signal(SIGUSR1, sigfn );

	pthread_create(&t1,NULL,th_fn1,"Thread One");
        pthread_create(&t2,NULL,th_fn2,"Thread Two");

	printf("main tid is :%u\n",pthread_self());
 //   	kill(getpid(), SIGUSR1);	//Sending signal to the main process

 //       pthread_kill(t1,SIGUSR1);
   //     pthread_kill(t2,SIGUSR2);

	pthread_join(t1,NULL);
	pthread_join(t2,NULL);

        //sleep(100);
}


void * th_fn2(void * p)
{
    	char * str;
   	int i, pid;
    
    	str = (char *)p;
    	pid = getpid();
   
    	signal(SIGUSR1, sigfn );
	//kill(pid, SIGUSR1);
    
	printf("tid is :%u    \n",pthread_self());
     	printf ("%s: \t Started Now: \t For Process %d \n",str,pid);
	

	i=0;
    	while(i++ < 3)
    	{
       		printf("%s: \t Hello : \t %d \n",str,pid);
        	sleep(1);

    	}
    	printf("%s: \t Finished: \t For Process %d \n\n",str,pid);
}
void * th_fn1(void * p)
{
    	char * str;
   	int i, pid;
    
    	str = (char *)p;
    	pid = getpid();
   
	kill(pid, SIGUSR1);
    	//signal(SIGUSR1, sigfn );
    
	printf("tid is :%u    \n",pthread_self());
     	printf ("%s: \t Started Now: \t For Process %d \n",str,pid);
	

	i=0;
    	while(i++ < 3)
    	{
       		printf("%s: \t Hello : \t %d \n",str,pid);
        	sleep(1);

    	}
    	printf("%s: \t Finished: \t For Process %d \n\n",str,pid);
}
void sigfn ( int k)
{
        printf("Yes I got the Signal: \t %u \n",pthread_self());
	printf("Signal number is %d\n",k);
}
