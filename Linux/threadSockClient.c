# include <unistd.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <arpa/inet.h>
# include <sys/types.h>
# include <stdio.h>
# include <string.h>
# include <pthread.h>

int cfd;

void * th_fn(void * p)   // The thread Function called by both threads
{
	recv_data(cfd);
}
main (int argc, char * argv[])
{
	int res;
	struct sockaddr_in saddr,caddr;
	pthread_t t1,t2;

	saddr.sin_family = AF_INET;
	saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	saddr.sin_port = htons(atoi(argv[1]));
	memset(&(saddr.sin_zero), 0, 8);

	cfd = socket(AF_INET, SOCK_STREAM, 0);

	if(cfd > 0)
		printf("Client: socket Successful %d \n",cfd);
	else if (cfd == -1)
		printf("Client: socket Failed \n");

	res = connect(cfd,(struct sockaddr *)&saddr,sizeof(saddr));

	if (res == 0)
		printf("Client: connected to the server \n");
	else if (res == -1)
		printf("Client: connect Failed \n");

	pthread_create(&t1,NULL,th_fn,NULL);
	pthread_create(&t2,NULL,th_fn,NULL);
	pthread_join(t2,NULL);
	close(cfd);
} // End Of Client with two reader threads

