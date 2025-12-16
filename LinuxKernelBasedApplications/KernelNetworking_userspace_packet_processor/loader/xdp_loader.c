// xdp_loader.c -- generic loader used for MODE_PASS_ALL (baseline classify+pass,
// feeds normal socket) and MODE_XDP_DROP (pure in-kernel drop ceiling benchmark).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <net/if.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "../bpf/xdp_common.h"

static volatile int stop;
static void on_sig(int s) { (void)s; stop = 1; }

static const char *stat_names[__STAT_MAX] = {
	"rx_total", "rx_bytes", "pass", "drop", "redirect_xsk",
	"tcp", "udp", "other", "malformed",
};

static void print_stats(struct bpf_map *stats_map, int ncpu)
{
	__u64 vals[ncpu];
	for (__u32 i = 0; i < __STAT_MAX; i++) {
		__u64 sum = 0;
		if (bpf_map__lookup_elem(stats_map, &i, sizeof(i), vals, sizeof(vals[0]) * ncpu, 0) == 0)
			for (int c = 0; c < ncpu; c++) sum += vals[c];
		printf("%-14s %llu\n", stat_names[i], (unsigned long long)sum);
	}
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: %s <ifname> <mode: pass|drop> [interval_s]\n", argv[0]);
		return 1;
	}
	const char *ifname = argv[1];
	int ifindex = if_nametoindex(ifname);
	if (!ifindex) { perror("if_nametoindex"); return 1; }

	__u32 mode = strcmp(argv[2], "drop") == 0 ? MODE_XDP_DROP : MODE_PASS_ALL;
	int interval = argc > 3 ? atoi(argv[3]) : 1;

	struct bpf_object *obj = bpf_object__open_file("xdp_kern.o", NULL);
	if (!obj || bpf_object__load(obj)) { fprintf(stderr, "load failed\n"); return 1; }

	struct bpf_program *prog = bpf_object__find_program_by_name(obj, "xdp_pkt_classify");
	int prog_fd = bpf_program__fd(prog);

	if (bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL) < 0) {
		fprintf(stderr, "bpf_xdp_attach failed (try XDP_FLAGS_SKB_MODE fallback)\n");
		return 1;
	}

	struct bpf_map *mode_map = bpf_object__find_map_by_name(obj, "mode_map");
	__u32 key0 = 0;
	bpf_map__update_elem(mode_map, &key0, sizeof(key0), &mode, sizeof(mode), BPF_ANY);

	struct bpf_map *stats_map = bpf_object__find_map_by_name(obj, "stats_map");
	int ncpu = libbpf_num_possible_cpus();

	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);
	printf("attached xdp_pkt_classify to %s, mode=%s\n", ifname, mode == MODE_XDP_DROP ? "drop" : "pass");

	while (!stop) {
		sleep(interval);
		printf("---- stats (ifindex=%d) ----\n", ifindex);
		print_stats(stats_map, ncpu);
	}

	bpf_xdp_detach(ifindex, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);
	bpf_object__close(obj);
	return 0;
}
