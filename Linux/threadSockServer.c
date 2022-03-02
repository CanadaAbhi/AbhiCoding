# include <unistd.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <arpa/inet.h>
# include <stdio.h>
# include <errno.h>

main (int argc, char * argv[])
{
	int size;
	int sfd,cfd,res;
	struct sockaddr_in saddr, caddr;

	saddr.sin_family = AF_INET;
	saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	saddr.sin_port = htons(atoi(argv[1]));
	memset(&(saddr.sin_zero), 0, 8);

	sfd = socket(AF_INET, SOCK_STREAM, 0);

	if(sfd > 0)
	    printf("Server: socket Successful ID = %d \n",sfd);
	else if (sfd == -1)
	    printf("Server: socket Failed \n");

	res = bind(sfd,(struct sockaddr *) &saddr,sizeof(struct sockaddr));

	if (res == 0)
	    printf("Server: bind Successful \n");
	else if (res == -1)
	    printf("Server: bind Failed \n");


	res = listen(sfd,5);
	if (res == 0)
	    printf("Server: listen Successful \n");
	else if (res == -1)
	    printf("Server: listen Failed \n");
	size = sizeof(struct sockaddr);
	//while(1){
	   	cfd=accept(sfd,(struct sockaddr *) &caddr,&size);

		if (cfd > 0)   {
	 	       printf("Server: connection received from client \n");
            		close(sfd);
 	           	send_data(cfd);
        	    	close(cfd);
 		/*if (fork  () == 0) Uncomment this to have two writer-server processes
        	{
            		close(sfd);
            		send_data(cfd);
            		close(cfd);
            		exit(0);
        	}*/
    		}
	    	else if (cfd == -1)    {
		        printf("Server: accept failed \n", cfd);
        		close(sfd);
    		}
//	} // End of While (1) Loop
} // End of Main Server Program

