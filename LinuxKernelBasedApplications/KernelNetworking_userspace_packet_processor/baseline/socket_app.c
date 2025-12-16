// socket_app.c -- ordinary AF_INET/SOCK_DGRAM receiver: full netstack path
// (NIC -> driver -> sk_buff alloc -> netif_receive_skb -> IP -> UDP -> socket
// recv queue -> recvmmsg). This is the "normal socket" comparison point;
// run this while xdp_loader is attached in MODE_PASS_ALL on the same iface.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>

#define PORT 9999
#define BATCH 32
#define BUFLEN 2048

static volatile int stop;
static void on_sig(int s) { (void)s; stop = 1; }

static __u64_t_unused; /* placeholder avoided */

static unsigned long long now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv)
{
	int run_secs = argc > 1 ? atoi(argv[1]) : 10;
	int fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (fd < 0) { perror("socket"); return 1; }

	struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(PORT),
				     .sin_addr.s_addr = INADDR_ANY };
	if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

	struct mmsghdr msgs[BATCH];
	struct iovec iovecs[BATCH];
	char bufs[BATCH][BUFLEN];
	memset(msgs, 0, sizeof(msgs));
	for (int i = 0; i < BATCH; i++) {
		iovecs[i].iov_base = bufs[i];
		iovecs[i].iov_len = BUFLEN;
		msgs[i].msg_hdr.msg_iov = &iovecs[i];
		msgs[i].msg_hdr.msg_iovlen = 1;
	}

	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);

	unsigned long long rx_pkts = 0, rx_bytes = 0, lat_sum = 0, lat_count = 0;
	unsigned long long t_start = now_ns();
	time_t deadline = time(NULL) + run_secs;

	while (!stop && time(NULL) < deadline) {
		int n = recvmmsg(fd, msgs, BATCH, MSG_DONTWAIT, NULL);
		if (n <= 0) { usleep(200); continue; }
		unsigned long long recv_ts = now_ns();
		for (int i = 0; i < n; i++) {
			rx_pkts++;
			rx_bytes += msgs[i].msg_len;
			if (msgs[i].msg_len >= 8) {
				unsigned long long send_ns;
				memcpy(&send_ns, bufs[i], 8);
				lat_sum += (recv_ts - send_ns);
				lat_count++;
			}
		}
	}

	double secs = (now_ns() - t_start) / 1e9;
	printf("=== plain UDP socket results ===\n");
	printf("duration_s : %.2f\n", secs);
	printf("rx_packets : %llu\n", rx_pkts);
	printf("pps        : %.0f\n", rx_pkts / secs);
	printf("throughput : %.2f Mbps\n", (rx_bytes * 8.0 / secs) / 1e6);
	if (lat_count)
		printf("latency_ns : avg=%llu (n=%llu)\n", lat_sum / lat_count, lat_count);
	close(fd);
	return 0;
}
