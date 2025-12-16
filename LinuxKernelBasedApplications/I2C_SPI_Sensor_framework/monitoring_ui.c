// monitoring_ui.c -- connects to sensor_daemon's UNIX socket, renders a
// live terminal dashboard from the broadcast JSON lines.
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#define UI_SOCK_PATH "/tmp/sensor_daemon.sock"

int main(void)
{
	struct sockaddr_un addr = { .sun_family = AF_UNIX };
	char buf[512];
	int fd;

	strncpy(addr.sun_path, UI_SOCK_PATH, sizeof(addr.sun_path) - 1);
	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("connect (is sensor_daemon running?)");
		return 1;
	}

	printf("\033[2J"); /* clear screen once */
	for (;;) {
		ssize_t n = read(fd, buf, sizeof(buf) - 1);
		if (n <= 0)
			break;
		buf[n] = '\0';

		int temp_mC = 0, alarm = 0, ax = 0, ay = 0, az = 0;
		sscanf(buf, "{\"temp_mC\":%d,\"alarm\":%d,\"ax\":%d,\"ay\":%d,\"az\":%d}",
		       &temp_mC, &alarm, &ax, &ay, &az);

		printf("\033[H"); /* cursor home, redraw in place */
		printf("=== Sensor Monitoring UI ===================\n");
		printf(" Temperature : %6.2f C   %s\n", temp_mC / 1000.0,
		       alarm ? "*** ALARM ***" : "OK");
		printf(" IMU accel   : ax=%6d ay=%6d az=%6d (milli-g)\n", ax, ay, az);
		printf("=============================================\n");
		fflush(stdout);
	}
	close(fd);
	return 0;
}
