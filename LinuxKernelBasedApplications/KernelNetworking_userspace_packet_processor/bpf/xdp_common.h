#ifndef XDP_COMMON_H
#define XDP_COMMON_H
#include <linux/types.h>

enum stat_idx {
	STAT_RX_TOTAL = 0,
	STAT_RX_BYTES,
	STAT_PASS,
	STAT_DROP,
	STAT_REDIRECT_XSK,
	STAT_TCP,
	STAT_UDP,
	STAT_OTHER,
	STAT_MALFORMED,
	__STAT_MAX,
};

struct flow_key {
	__u32 saddr;
	__u32 daddr;
	__u16 sport;
	__u16 dport;
	__u8  proto;
	__u8  pad[3];
};

struct flow_stats {
	__u64 packets;
	__u64 bytes;
	__u64 last_seen_ns;
};

/* pushed to userspace via BPF_MAP_TYPE_PERF_EVENT_ARRAY for latency histograms */
struct latency_sample {
	__u64 send_ns; /* embedded by gen/udp_flood.c */
	__u64 xdp_ns;  /* bpf_ktime_get_ns() at classify time */
	__u32 seq;
};

enum xdp_mode {
	MODE_PASS_ALL     = 0, /* classify + count, XDP_PASS everything -> normal socket */
	MODE_XDP_DROP     = 1, /* pure in-kernel drop, never leaves XDP -> ceiling pps */
	MODE_AF_XDP_REDIR = 2, /* redirect target-port UDP into AF_XDP socket */
};

#define TARGET_UDP_PORT 9999

#endif
