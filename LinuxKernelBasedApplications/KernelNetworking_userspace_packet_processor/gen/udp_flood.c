// udp_flood.c -- sends UDP datagrams at a target rate; first 8 bytes of the
// payload are a CLOCK_MONOTONIC ns timestamp, consumed by the eBPF program's
// latency_events and by socket_app.c/xsk_app.c for end-to-end latency.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>

static unsigned long long now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s <dst_ip> <dst_port> <pps> [run_secs] [payload_bytes]\n", argv[0]);
		return 1;
	}
	const char *dst_ip = argv[1];
	int dst_port = atoi(argv[2]);
	long pps = atol(argv[3]);
	int run_secs = argc > 4 ? atoi(argv[4]) : 10;
	int paylen = argc > 5 ? atoi(argv[5]) : 64;
	if (paylen < 12) paylen = 12;

	int fd = socket(AF_INET, SOCK_DGRAM, 0);
	struct sockaddr_in dst = { .sin_family = AF_INET, .sin_port = htons(dst_port) };
	inet_pton(AF_INET, dst_ip, &dst.sin_addr);

	char *buf = calloc(1, paylen);
	unsigned int seq = 0;
	long interval_ns = 1000000000L / pps;
	unsigned long long next = now_ns();
	unsigned long long sent = 0, t_start = now_ns();
	time_t deadline = time(NULL) + run_secs;

	while (time(NULL) < deadline) {
		unsigned long long ts = now_ns();
		memcpy(buf, &ts, 8);
		memcpy(buf + 8, &seq, 4);
		seq++;
		if (sendto(fd, buf, paylen, 0, (struct sockaddr *)&dst, sizeof(dst)) > 0)
			sent++;
		next += interval_ns;
		long sleep_ns = next - now_ns();
		if (sleep_ns > 0) {
			struct timespec req = { .tv_sec = sleep_ns / 1000000000L,
						 .tv_nsec = sleep_ns % 1000000000L };
			nanosleep(&req, NULL);
		}
	}
	double secs = (now_ns() - t_start) / 1e9;
	printf("sent=%llu pps_actual=%.0f\n", sent, sent / secs);
	return 0;
}
