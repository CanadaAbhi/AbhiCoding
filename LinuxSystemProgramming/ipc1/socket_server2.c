#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>

static void hexdump(char *buf, size_t len) {
	size_t i,n=0;
	unsigned char c;
	while(n < len) {
		for(i=0; i < 16; i++) {
			c = (n+i < len) ? buf[n+i] : 0;
			if (n+i < len) fprintf(stderr,"%.2x ", c);
			else fprintf(stderr, "   ");
		}
		for(i=0; i < 16; i++) {
			c = (n+i < len) ? buf[n+i] : ' ';
			if (c < 0x20 || c > 0x7e) c = '.';
			fprintf(stderr,"%c",c);
		}
		fprintf(stderr,"\n");
		n += 16;
	}
}

int main()
{
	int sock_fd, client_fd;
	struct sockaddr_un addr, client_addr;
	char buffer[1000];

	sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		perror("socket");
		exit(-1);
	}

	printf("Socket File Descriptor:%d\n", sock_fd);

	memset(&addr, 0, sizeof(struct sockaddr_un));
	addr.sun_family = AF_UNIX;
	//copying one byte less for trailing zero

	if (bind(sock_fd, (struct sockaddr *)&addr, sizeof(sa_family_t)) < 0) {
		perror("bind failed");
	}
	printf("Bind successful\n");

	/* display the abstract socket name. the socket name isn't null terminated! */
	struct sockaddr_un tmp;
	int addrlen = sizeof(struct sockaddr_un);
	int sc;
	sc = getsockname(sock_fd, (struct sockaddr *)&tmp, &addrlen);
	if (sc < 0) {
		fprintf(stderr,"getsockname: %s\n", strerror(errno));
	}

	fprintf(stderr, "socket name in hex and ASCII (. for null byte):\n");
	hexdump(tmp.sun_path, addrlen-sizeof(sa_family_t));



	listen(sock_fd, 5);
	perror("listen");

	client_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &addrlen);
	perror("accept");	
	printf("accept client filedescriptor:%d\n", client_fd);
	read(client_fd, buffer, sizeof(buffer));
	perror("read");
	printf("buffer from client:%s\n", buffer);
	close(client_fd);
	close(sock_fd);

	return 0;
}
