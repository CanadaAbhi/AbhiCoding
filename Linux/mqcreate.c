#include <stdio.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include "common.h"

int main()
{
  int msgid;

  /* Create the message queue with the id MY_MQ_ID */
  msgid = msgget( MY_MQ_ID, IPC_CREAT| 0666 );

  if (msgid >= 0) {

    printf( "Created a Message Queue %d\n", msgid );

  }
//	msgctl(msgid,IPC_RMID,NULL);

  return 0;
}

