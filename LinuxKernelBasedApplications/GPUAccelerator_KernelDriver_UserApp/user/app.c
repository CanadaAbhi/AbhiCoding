// app.c -- Application:
//   Buffer input; Buffer output; submit_job(input, output); wait_for_completion();
// plus a benchmark loop measuring pps/latency/throughput and result correctness.
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "accel_lib.h"
#include "../include/accel_uapi.h"

static uint64_t now_ns(void)
{
	struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void)
{
	int fd = accel_open_device();
	if (fd < 0) return 1;

	const size_t N = 1 << 20; /* 1 MiB */
	Buffer input  = accel_buffer_alloc(fd, N);
	Buffer output = accel_buffer_alloc(fd, N);
	if (!input.vaddr || !output.vaddr) { fprintf(stderr, "buffer alloc failed\n"); return 1; }

	memset(input.vaddr, 7, N); /* test pattern: every byte = 7 */

	/* ---- single job, exactly the requested shape ---- */
	uint64_t seqno = submit_job(fd, &input, &output, ACCEL_OP_VEC_ADD_SCAL, 42);
	int status = wait_for_completion(fd, seqno, 1000000000ULL /* 1s */);
	printf("single job: status=%d output[0]=%d (expect %d)\n",
	       status, ((unsigned char *)output.vaddr)[0], (7 + 42) & 0xff);

	/* ---- benchmark loop ---- */
	const int ITERS = 5000;
	uint64_t lat_sum = 0, lat_min = ~0ULL, lat_max = 0, ok = 0, fail = 0;
	uint64_t t0 = now_ns();

	for (int i = 0; i < ITERS; i++) {
		uint64_t t_submit = now_ns();
		uint64_t sq = submit_job(fd, &input, &output, ACCEL_OP_VEC_ADD_SCAL, i & 0xff);
		int st = wait_for_completion(fd, sq, 1000000000ULL);
		uint64_t d = now_ns() - t_submit;
		lat_sum += d;
		if (d < lat_min) lat_min = d;
		if (d > lat_max) lat_max = d;
		if (st == 0) ok++; else fail++;
	}
	double secs = (now_ns() - t0) / 1e9;

	struct accel_stats st;
	ioctl(fd, ACCEL_IOC_GET_STATS, &st);

	printf("\n=== accel_drv benchmark (%d jobs, %zu B each) ===\n", ITERS, N);
	printf("jobs/sec       : %.0f\n", ITERS / secs);
	printf("throughput     : %.2f MB/s\n", (ITERS * (double)N / secs) / 1e6);
	printf("latency_ns     : min=%llu avg=%llu max=%llu\n",
	       (unsigned long long)lat_min, (unsigned long long)(lat_sum / ITERS),
	       (unsigned long long)lat_max);
	printf("ok=%llu fail=%llu\n", (unsigned long long)ok, (unsigned long long)fail);
	printf("cmdq depth_max : %llu\n", (unsigned long long)st.cmdq_depth_max);
	printf("driver stats   : submitted=%llu completed=%llu failed=%llu\n",
	       (unsigned long long)st.jobs_submitted, (unsigned long long)st.jobs_completed,
	       (unsigned long long)st.jobs_failed);

	accel_buffer_free(fd, &input);
	accel_buffer_free(fd, &output);
	close(fd);
	return 0;
}
