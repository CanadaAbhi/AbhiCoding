// xsk_app.c -- loads xdp_kern.o, sets MODE_AF_XDP_REDIR, creates a UMEM +
// AF_XDP socket bound to the target queue, populates xsk_map, then runs a
// tight RX loop pulling frames straight out of the NIC's DMA rings (bypassing
// the normal sk_buff path entirely) and classifying/measuring in userspace.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <time.h>
#include <net/if.h>
#include <sys/mman.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <xdp/xsk.h>
#include "../bpf/xdp_common.h"

#define NUM_FRAMES   4096
#define FRAME_SIZE   XSK_UMEM__DEFAULT_FRAME_SIZE
#define RX_BATCH     64

struct xsk_ctx {
	struct xsk_umem *umem;
	void *buffer;
	struct xsk_ring_prod fq, cq, tx;
	struct xsk_ring_cons rx;
	struct xsk_socket *xsk;
	int fd;
};

static volatile int stop;
static void on_sig(int s) { (void)s; stop = 1; }

static __u64 now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* latency samples arrive via perf buffer from the eBPF program */
static __u64 lat_sum, lat_count, lat_min = ~0ULL, lat_max;

static void on_latency_sample(void *ctx, int cpu, void *data, __u32 size)
{
	(void)ctx; (void)cpu;
	if (size < sizeof(struct latency_sample)) return;
	struct latency_sample *ls = data;
	__u64 d = ls->xdp_ns - ls->send_ns;
	lat_sum += d; lat_count++;
	if (d < lat_min) lat_min = d;
	if (d > lat_max) lat_max = d;
}

static int setup_umem(struct xsk_ctx *x)
{
	size_t sz = NUM_FRAMES * FRAME_SIZE;
	x->buffer = mmap(NULL, sz, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	if (x->buffer == MAP_FAILED) { perror("mmap"); return -1; }

	struct xsk_umem_config cfg = {
		.fill_size = XSK_RING_PROD__DEFAULT_NUM_DESCS,
		.comp_size = XSK_RING_CONS__DEFAULT_NUM_DESCS,
		.frame_size = FRAME_SIZE,
		.frame_headroom = XSK_UMEM__DEFAULT_FRAME_HEADROOM,
	};
	return xsk_umem__create(&x->umem, x->buffer, sz, &x->fq, &x->cq, &cfg);
}

static void refill_fill_queue(struct xsk_ctx *x, __u32 count, __u32 *frame_off)
{
	__u32 idx;
	if (xsk_ring_prod__reserve(&x->fq, count, &idx) < count)
		return;
	for (__u32 i = 0; i < count; i++) {
		*xsk_ring_prod__fill_addr(&x->fq, idx + i) = *frame_off;
		*frame_off = (*frame_off + FRAME_SIZE) % (NUM_FRAMES * FRAME_SIZE);
	}
	xsk_ring_prod__submit(&x->fq, count);
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s <ifname> <queue_id> [run_secs]\n", argv[0]); return 1; }
	const char *ifname = argv[1];
	int queue_id = atoi(argv[2]);
	int run_secs = argc > 3 ? atoi(argv[3]) : 10;
	int ifindex = if_nametoindex(ifname);
	if (!ifindex) { perror("if_nametoindex"); return 1; }

	struct bpf_object *obj = bpf_object__open_file("xdp_kern.o", NULL);
	if (!obj || bpf_object__load(obj)) { fprintf(stderr, "load failed\n"); return 1; }
	struct bpf_program *prog = bpf_object__find_program_by_name(obj, "xdp_pkt_classify");
	int prog_fd = bpf_program__fd(prog);
	if (bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL) < 0) {
		fprintf(stderr, "bpf_xdp_attach failed\n"); return 1;
	}

	__u32 mode = MODE_AF_XDP_REDIR, key0 = 0;
	struct bpf_map *mode_map = bpf_object__find_map_by_name(obj, "mode_map");
	bpf_map__update_elem(mode_map, &key0, sizeof(key0), &mode, sizeof(mode), BPF_ANY);

	struct xsk_ctx x = {0};
	if (setup_umem(&x)) { fprintf(stderr, "umem create failed\n"); return 1; }

	struct xsk_socket_config sxdp = {
		.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS,
		.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS,
		.libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD, /* we already attached our own prog */
		.xdp_flags = XDP_FLAGS_UPDATE_IF_NOEXIST,
		.bind_flags = XDP_ZEROCOPY | XDP_USE_NEED_WAKEUP,
	};
	if (xsk_socket__create(&x.xsk, ifname, queue_id, x.umem, &x.rx, &x.tx, &sxdp)) {
		fprintf(stderr, "zero-copy bind failed, retrying in copy mode\n");
		sxdp.bind_flags = XDP_COPY;
		if (xsk_socket__create(&x.xsk, ifname, queue_id, x.umem, &x.rx, &x.tx, &sxdp)) {
			perror("xsk_socket__create"); return 1;
		}
	}
	x.fd = xsk_socket__fd(x.xsk);

	/* wire this AF_XDP socket into the xsk_map[queue_id] used by bpf_redirect_map() */
	struct bpf_map *xsk_map = bpf_object__find_map_by_name(obj, "xsk_map");
	__u32 qid = queue_id;
	bpf_map__update_elem(xsk_map, &qid, sizeof(qid), &x.fd, sizeof(x.fd), BPF_ANY);

	__u32 frame_off = 0;
	refill_fill_queue(&x, XSK_RING_PROD__DEFAULT_NUM_DESCS, &frame_off);

	/* latency perf buffer consumer */
	struct bpf_map *lat_map = bpf_object__find_map_by_name(obj, "latency_events");
	struct perf_buffer *pb = perf_buffer__new(bpf_map__fd(lat_map), 64, on_latency_sample, NULL, NULL, NULL);

	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);

	struct pollfd pfd = { .fd = x.fd, .events = POLLIN };
	__u64 rx_pkts = 0, rx_bytes = 0;
	__u64 t_start = now_ns();
	time_t deadline = time(NULL) + run_secs;

	while (!stop && time(NULL) < deadline) {
		int n = poll(&pfd, 1, 100);
		perf_buffer__poll(pb, 0);
		if (n <= 0) continue;

		__u32 idx, received = xsk_ring_cons__peek(&x.rx, RX_BATCH, &idx);
		if (!received) continue;

		for (__u32 i = 0; i < received; i++) {
			const struct xdp_desc *desc = xsk_ring_cons__rx_desc(&x.rx, idx + i);
			/* userspace "classification/processing" stage: the packet's raw
			 * bytes live at x.buffer + desc->addr, len = desc->len. A real
			 * pipeline would parse further here; we just count bytes. */
			rx_pkts++;
			rx_bytes += desc->len;
		}
		xsk_ring_cons__release(&x.rx, received);
		refill_fill_queue(&x, received, &frame_off);
	}

	double secs = (now_ns() - t_start) / 1e9;
	printf("\n=== AF_XDP results (%s, queue %d) ===\n", ifname, queue_id);
	printf("duration_s   : %.2f\n", secs);
	printf("rx_packets   : %llu\n", (unsigned long long)rx_pkts);
	printf("rx_bytes     : %llu\n", (unsigned long long)rx_bytes);
	printf("pps          : %.0f\n", rx_pkts / secs);
	printf("throughput   : %.2f Mbps\n", (rx_bytes * 8.0 / secs) / 1e6);
	if (lat_count) {
		printf("latency_ns   : min=%llu avg=%llu max=%llu (n=%llu)\n",
		       (unsigned long long)lat_min, (unsigned long long)(lat_sum / lat_count),
		       (unsigned long long)lat_max, (unsigned long long)lat_count);
	}

	xsk_socket__delete(x.xsk);
	xsk_umem__delete(x.umem);
	bpf_xdp_detach(ifindex, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);
	bpf_object__close(obj);
	return 0;
}
