/* This is a simple tcpclient which connects to the server and sends a message*/
#include<sys/socket.h>
#include<sys/types.h>
#include<netinet/in.h>
#include <stdlib.h>
#include<arpa/inet.h>
#include<stdio.h>

#define PORT 1034

struct sockaddr_in server;
main()
{
	int n,sd,length;
	char msg[40];
	//length=sizeof(client);
	if((sd=socket(PF_INET,SOCK_STREAM,0))==-1) {
		perror("socket");
		exit(1);
	}

	server.sin_family=PF_INET;
	server.sin_port=htons(PORT);
	server.sin_addr.s_addr=inet_addr("10.0.0.1");
	

	if(connect(sd,(struct sockaddr *)&server,sizeof(server))==-1) {
		perror("connect");
		exit(1);
	}
            
	printf("Enter the message you want to send to server\n");
        fgets(msg,40,stdin);
 	send(sd,msg,40,0);
	printf("Waiting for message from server..............\n");
	n=recv(sd,msg,40,0);
	msg[n]='\0';
	
	printf("Message received from server is:%s\n",msg);
	close(sd);	
}
