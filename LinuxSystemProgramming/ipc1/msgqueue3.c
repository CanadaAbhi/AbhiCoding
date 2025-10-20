#include <unistd.h>
#include <sys/types.h>
#include <mqueue.h>
#include <fcntl.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

mqd_t mqdes;

void* message_callback(int signum, siginfo_t* info, void* context)
{
	char buffer[8192];
	int msg_priority = -1;

	printf("%s\n", __func__);
	mq_receive(mqdes, buffer, sizeof(buffer), &msg_priority);
	perror("mq_receive");
	printf("Priority:%d\tBuffer:%s\n", msg_priority, buffer);
	struct sigevent sev;
	sev.sigev_notify = SIGEV_SIGNAL;
	sev.sigev_signo = SIGUSR1;
	mq_notify(mqdes, &sev);
	perror("mq_notify");

}



int main()
{
	int status;
	struct mq_attr attr;

	mqdes = mq_open("/my_queue", O_RDONLY | O_NONBLOCK | O_CREAT, 0666, NULL);
	perror("mqdes");
	struct sigevent sev;
	sev.sigev_notify = SIGEV_SIGNAL;
	sev.sigev_signo = SIGUSR1;

	struct sigaction sa;
	sa.sa_flags = SA_SIGINFO;
	sa.sa_sigaction = message_callback;

	sigemptyset(&(sa.sa_mask));
	if(sigaction(SIGUSR1, &sa, NULL) == -1){
		perror("sigaction");
		exit(1);
	}

	mq_notify(mqdes, &sev);
	perror("mq_notify");
	while(1) {}
	status = mq_close(mqdes);          /* Close message queue  */
	return 0;
}