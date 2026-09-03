#include <sys/msg.h>
#include <stdio.h>
#include "common.h"

int main()
{
  MY_TYPE_T myObject;
  int qid, ret;

  qid = msgget( MY_MQ_ID, 0 );

  if (qid >= 0) {

    ret = msgrcv( qid, (struct msgbuf *)&myObject, 
                   sizeof(MY_TYPE_T), 0, 0 );

    if (ret != -1) {

      printf( "Message Type: %ld\n", myObject.type );
      printf( "Float Value:  %f\n", myObject.fval );
      printf( "Uint Value:   %d\n", myObject.uival );
      printf( "String Value: %s\n", myObject.strval );

    }

  }

  return 0;
}

