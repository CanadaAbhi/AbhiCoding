#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

int main(int argc, char *argv[])
{
	int sock_fd;
	struct sockaddr_un addr;
	char buffer[] = "Hello from Client";
	char *socket_name;

	if (argc < 2) {
		fprintf(stderr, "usage: %s <name> (omit leading nul)\n", argv[0]);
		exit(1);
	}

	socket_name = argv[1];

	sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		perror("socket");
		exit(-1);
	}

	printf("Socket File Descriptor:%d\n", sock_fd);

	memset(&addr, 0, sizeof(struct sockaddr_un));
	addr.sun_family = AF_UNIX;
	*addr.sun_path = '\0';
	//copying one byte less for trailing zero
	strncpy(addr.sun_path+1, socket_name, sizeof(addr.sun_path)-1);
	
	/* the length when connecting to an abstract socket is the
	 * initial sa_family_t (2 bytes) plus leading null byte + name 
	 * (no trailing null in the name, for an autobind socket) */
	int slen = sizeof(sa_family_t)+1+strlen(socket_name);
	if (connect(sock_fd, (struct sockaddr *)&addr, slen) < 0) {
		perror("connect failed");
		exit(-2);
	}
	printf("connect successful\n");
	write(sock_fd, buffer, sizeof(buffer));
	perror("write");
	close(sock_fd);

	return 0;
}
