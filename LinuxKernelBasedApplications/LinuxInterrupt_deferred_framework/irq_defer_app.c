// irq_defer_app.c -- polls irq_defer_lab0, collects latency records for a
// fixed duration, prints P50/P95/P99 for dispatch/proc/total latency.
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <time.h>
#include <string.h>
#include "irq_defer_uapi.h"

#define DEV "/dev/irq_defer_lab0"
#define MAX_SAMPLES 200000

static int cmp_u64(const void *a, const void *b)
{
	__u64 x = *(const __u64 *)a, y = *(const __u64 *)b;
	return (x > y) - (x < y);
}

static __u64 percentile(__u64 *sorted, int n, double p)
{
	int idx = (int)(p * (n - 1));
	return n ? sorted[idx] : 0;
}

int main(int argc, char **argv)
{
	int fd, run_secs = argc > 1 ? atoi(argv[1]) : 10;
	struct pollfd pfd;
	struct timespec start, now;
	static __u64 dispatch[MAX_SAMPLES], proc[MAX_SAMPLES], total[MAX_SAMPLES];
	int n = 0;
	__u32 mode;

	fd = open(DEV, O_RDONLY);
	if (fd < 0) { perror("open"); return 1; }
	ioctl(fd, IRQ_DEFER_IOC_GET_MODE, &mode);
	static const char *names[] = { "workqueue", "tasklet", "threaded_irq" };
	printf("mode=%s, collecting for %ds...\n", names[mode], run_secs);

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (;;) {
		struct irq_defer_record rec;
		pfd.fd = fd; pfd.events = POLLIN;
		if (poll(&pfd, 1, 200) > 0 && (pfd.revents & POLLIN)) {
			if (read(fd, &rec, sizeof(rec)) == sizeof(rec) && n < MAX_SAMPLES) {
				dispatch[n] = rec.dispatch_ns;
				proc[n] = rec.proc_ns;
				total[n] = rec.total_ns;
				n++;
			}
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (now.tv_sec - start.tv_sec >= run_secs)
			break;
	}
	close(fd);

	qsort(dispatch, n, sizeof(__u64), cmp_u64);
	qsort(proc, n, sizeof(__u64), cmp_u64);
	qsort(total, n, sizeof(__u64), cmp_u64);

	printf("samples=%d\n", n);
	printf("%-10s %10s %10s %10s\n", "stage", "P50(ns)", "P95(ns)", "P99(ns)");
	printf("%-10s %10llu %10llu %10llu\n", "dispatch",
	       (unsigned long long)percentile(dispatch, n, 0.50),
	       (unsigned long long)percentile(dispatch, n, 0.95),
	       (unsigned long long)percentile(dispatch, n, 0.99));
	printf("%-10s %10llu %10llu %10llu\n", "proc",
	       (unsigned long long)percentile(proc, n, 0.50),
	       (unsigned long long)percentile(proc, n, 0.95),
	       (unsigned long long)percentile(proc, n, 0.99));
	printf("%-10s %10llu %10llu %10llu\n", "total",
	       (unsigned long long)percentile(total, n, 0.50),
	       (unsigned long long)percentile(total, n, 0.95),
	       (unsigned long long)percentile(total, n, 0.99));
	return 0;
}
