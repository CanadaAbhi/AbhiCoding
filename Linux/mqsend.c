#include <sys/msg.h>
#include <stdio.h>
#include <string.h>
#include "common.h"

int main()
{
  MY_TYPE_T myObject;
  int qid, ret;

  /* Get the queue ID for the existing queue */
  qid = msgget( MY_MQ_ID, 0 );

  if (qid > 0) {

    /* Create our message with a message queue type of 1 */
    myObject.type = 1L;
    myObject.fval = 128.256;
    myObject.uival = 512;
    strncpy( myObject.strval, "This is a test.\n", MAX_LINE );

    /* Send the message to the queue defined by the queue ID */
    ret = msgsnd( qid, (struct msgbuf *)&myObject, 
                   sizeof(MY_TYPE_T), 0 );

    if (ret != -1) {

      printf( "Message successfully sent to queue %d\n", qid );

    }

  }

  return 0;
}

