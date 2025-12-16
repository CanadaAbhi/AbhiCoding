// xdp_kern.bpf.c -- NIC -> XDP -> eBPF classification pipeline.
// Parses eth/ip/udp|tcp, maintains per-CPU counters + a 5-tuple flow_map,
// emits a latency sample for the benchmark target port, then dispatches
// XDP_PASS / XDP_DROP / bpf_redirect_map() depending on mode_map[0].
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include "xdp_common.h"

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, __STAT_MAX);
	__type(key, __u32);
	__type(value, __u64);
} stats_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, struct flow_key);
	__type(value, struct flow_stats);
} flow_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__uint(max_entries, 64);
	__type(key, __u32);
	__type(value, __u32);
} xsk_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u32);
} mode_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
	__uint(max_entries, 128);
	__type(key, __u32);
	__type(value, __u32);
} latency_events SEC(".maps");

static __always_inline void bump(__u32 idx, __u64 add)
{
	__u64 *v = bpf_map_lookup_elem(&stats_map, &idx);
	if (v)
		__sync_fetch_and_add(v, add);
}

static __always_inline __u32 get_mode(void)
{
	__u32 key = 0;
	__u32 *m = bpf_map_lookup_elem(&mode_map, &key);
	return m ? *m : MODE_PASS_ALL;
}

SEC("xdp")
int xdp_pkt_classify(struct xdp_md *ctx)
{
	void *data = (void *)(long)ctx->data;
	void *data_end = (void *)(long)ctx->data_end;
	__u32 mode = get_mode();
	__u64 len = data_end - data;

	bump(STAT_RX_TOTAL, 1);
	bump(STAT_RX_BYTES, len);

	struct ethhdr *eth = data;
	if ((void *)(eth + 1) > data_end) {
		bump(STAT_MALFORMED, 1);
		return XDP_PASS;
	}
	if (eth->h_proto != bpf_htons(ETH_P_IP)) {
		bump(STAT_OTHER, 1);
		return XDP_PASS;
	}

	struct iphdr *ip = (void *)(eth + 1);
	if ((void *)(ip + 1) > data_end) {
		bump(STAT_MALFORMED, 1);
		return XDP_PASS;
	}

	struct flow_key key = { .saddr = ip->saddr, .daddr = ip->daddr, .proto = ip->protocol };
	__u16 sport = 0, dport = 0;
	void *l4 = (void *)ip + (ip->ihl * 4);

	if (ip->protocol == IPPROTO_UDP) {
		struct udphdr *udp = l4;
		if ((void *)(udp + 1) > data_end) { bump(STAT_MALFORMED, 1); return XDP_PASS; }
		sport = bpf_ntohs(udp->source);
		dport = bpf_ntohs(udp->dest);
		bump(STAT_UDP, 1);
	} else if (ip->protocol == IPPROTO_TCP) {
		struct tcphdr *tcp = l4;
		if ((void *)(tcp + 1) > data_end) { bump(STAT_MALFORMED, 1); return XDP_PASS; }
		sport = bpf_ntohs(tcp->source);
		dport = bpf_ntohs(tcp->dest);
		bump(STAT_TCP, 1);
	} else {
		bump(STAT_OTHER, 1);
		return XDP_PASS;
	}
	key.sport = sport;
	key.dport = dport;

	struct flow_stats *fs = bpf_map_lookup_elem(&flow_map, &key);
	if (fs) {
		__sync_fetch_and_add(&fs->packets, 1);
		__sync_fetch_and_add(&fs->bytes, len);
		fs->last_seen_ns = bpf_ktime_get_ns();
	} else {
		struct flow_stats init = { .packets = 1, .bytes = len, .last_seen_ns = bpf_ktime_get_ns() };
		bpf_map_update_elem(&flow_map, &key, &init, BPF_ANY);
	}

	if (ip->protocol == IPPROTO_UDP && dport == TARGET_UDP_PORT) {
		struct udphdr *udp = l4;
		__u64 *payload_ts = (void *)(udp + 1);
		if ((void *)(payload_ts + 1) <= data_end) {
			struct latency_sample ls = { .send_ns = *payload_ts, .xdp_ns = bpf_ktime_get_ns() };
			bpf_perf_event_output(ctx, &latency_events, BPF_F_CURRENT_CPU, &ls, sizeof(ls));
		}
	}

	switch (mode) {
	case MODE_XDP_DROP:
		bump(STAT_DROP, 1);
		return XDP_DROP;
	case MODE_AF_XDP_REDIR:
		if (ip->protocol == IPPROTO_UDP && dport == TARGET_UDP_PORT) {
			int ret = bpf_redirect_map(&xsk_map, ctx->rx_queue_index, 0);
			if (ret == XDP_REDIRECT) { bump(STAT_REDIRECT_XSK, 1); return ret; }
		}
		bump(STAT_PASS, 1);
		return XDP_PASS;
	case MODE_PASS_ALL:
	default:
		bump(STAT_PASS, 1);
		return XDP_PASS;
	}
}
char _license[] SEC("license") = "GPL";
