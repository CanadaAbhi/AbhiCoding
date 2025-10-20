#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

int main()
{
	const char *SOCK_PATH = "srv_socket" ;
	int sock_fd, client_fd;
	struct sockaddr_un addr, client_addr;
	socklen_t addrlen;
	char buffer[] = "Hello from Server";

	sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		perror("socket");
		exit(-1);
	}

	unlink(SOCK_PATH);
	printf("Socket File Descriptor:%d\n", sock_fd);

	memset(&addr, 0, sizeof(struct sockaddr_un));
	addr.sun_family = AF_UNIX;
	//copying one byte less for trailing zero
	strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
	
	if (bind(sock_fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) < 0) {
		perror("bind failed");
	}
	printf("Bind successful\n");

	listen(sock_fd, 5);
	perror("listen");
	while(1)
	{
		client_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &addrlen);
		perror("accept");	
		printf("accept client filedescriptor:%d\n", client_fd);
		write(client_fd, buffer, sizeof(buffer));
		perror("write");
		printf("buffer to client:%s\n", buffer);
		close(client_fd);
	}
	close(sock_fd);

	return 0;
}
