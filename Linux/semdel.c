
#include <stdio.h>
#include <sys/sem.h>
#include "common.h"

int main()
{
  int semid, ret;

  /* Get the semaphore with the id MY_SEM_ID */
  semid = semget( MY_SEM_ID, 1, 0 );

  if (semid >= 0) {

    ret = semctl( semid, 0, IPC_RMID);

    if (ret != -1) {

      printf( "Semaphore %d removed.\n", semid );

    }

  }

  return 0;
}

