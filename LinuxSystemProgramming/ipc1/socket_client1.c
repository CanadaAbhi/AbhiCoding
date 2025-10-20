#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>

int main()
{
	const char *SOCK_PATH = "srv_socket" ;
	int sock_fd;
	struct sockaddr_un addr;
	char buffer[2];

	sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		perror("socket");
		exit(-1);
	}

	printf("Socket File Descriptor:%d\n", sock_fd);

	memset(&addr, 0, sizeof(struct sockaddr_un));
	addr.sun_family = AF_UNIX;
	//copying one byte less for trailing zero
	strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
	
	if (connect(sock_fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) < 0) {
		perror("connect failed");
		exit(-2);
	}
	printf("connect successful\n");
	while(1) {
		int unread_data;
		int read_data;
		ioctl(sock_fd, SIOCINQ, &unread_data);
		perror("ioctl");
		printf("Unread data:%d\n", unread_data);
		read_data = read(sock_fd, buffer, sizeof(buffer));
		perror("read");
		if (read_data == 0)
			break;
		printf("Data Read:%s\n", buffer);
	}
	close(sock_fd);

	return 0;
}