
#include <stdio.h>
#include <sys/sem.h>
#include "common.h"

int main()
{
  int semid;

  /* Create the semaphore with the id MY_SEM_ID */
  semid = semget( MY_SEMARRAY_ID, NUM_SEMAPHORES, 0666 | IPC_CREAT );

  if (semid != -1) {

    printf( "semacreate: Created a semaphore %d\n", semid );

  }

  return 0;
}

