
#include <pthread.h>
#include <stdio.h>
#include <signal.h>

void *myThread( void *arg )
{
  struct sigaction act;

  act.sa_sigaction = sigaction;
  act.sa_flags = SA_SIGINFO;
  sigaction( SIGUSR1, &act, 0 );



  printf( "Thread %d started\n", (int)arg );



  pthread_exit( arg );
}

#define MAX_THREADS	5

int main()
{
  int ret, i, status;
  pthread_t threadId;

  ret = pthread_create( &threadId, NULL, myThread, NULL );
  if (ret != 0) {
    printf( "Error creating thread %d\n", (int)threadId );
  }

  ret = pthread_join( threadId, (void **)&status );
  if (ret != 0) {
    printf( "Error joining thread %d\n", (int)threadId );
  } else {
    printf( "Status = %d\n", status );
  }

  return 0;
}
