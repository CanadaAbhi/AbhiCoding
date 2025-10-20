#include <unistd.h>
#include <sys/types.h>
#include <mqueue.h>
#include <fcntl.h>
#include <stdio.h>
#include <signal.h>

static void func(union sigval sv)
{
	char buffer[8192];
	int msg_priority = -1;
	mqd_t mqdes = (sv.sival_int);

	printf("%s\n", __func__);
	mq_receive(mqdes, buffer, sizeof(buffer), &msg_priority);
	perror("mq_receive");
	printf("Priority:%d\tBuffer:%s\n", msg_priority, buffer);
	struct sigevent sev;
	sev.sigev_notify = SIGEV_THREAD;
	sev.sigev_notify_function = func;
	sev.sigev_notify_attributes = NULL;
	sev.sigev_value.sival_int = mqdes;

	mq_notify(mqdes, &sev);
	perror("mq_notify");
}



int main()
{
	int status;
	mqd_t mqdes;
	struct mq_attr attr;

	mqdes = mq_open("/my_queue", O_RDONLY | O_NONBLOCK | O_CREAT, 0666, NULL);
	perror("mqdes");
	struct sigevent sev;
	sev.sigev_notify = SIGEV_THREAD;
	sev.sigev_notify_function = func;
	sev.sigev_notify_attributes = NULL;
	sev.sigev_value.sival_int = mqdes;

	mq_notify(mqdes, &sev);
	perror("mq_notify");
	while(1) {}
	status = mq_close(mqdes);          /* Close message queue  */
	return 0;
}
