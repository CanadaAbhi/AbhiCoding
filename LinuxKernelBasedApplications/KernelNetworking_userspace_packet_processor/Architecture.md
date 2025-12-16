   traffic generator (gen/udp_flood.c)
              |  UDP, 8-byte send-timestamp payload
              v
  +-------------------------- NIC --------------------------+
  |                            |                             |
  |                     XDP hook (early, driver RX)          |
  |                            v                             |
  |            eBPF: xdp_kern.bpf.c (xdp_pkt_classify)       |
  |   parse eth/ip/udp/tcp -> percpu stats -> flow_map       |
  |   -> emit latency_sample (perf buffer) -> mode dispatch  |
  |                            |                             |
  |     MODE_PASS_ALL    MODE_XDP_DROP    MODE_AF_XDP_REDIR  |
  |     XDP_PASS          XDP_DROP        bpf_redirect_map   |
  |          |                 |               (xsk_map)     |
  +----------|-----------------|-------------------|----------+
             v                 v                   v
      normal netstack     (never leaves       AF_XDP socket
      -> socket_app.c      kernel;             -> xsk_app.c
      (baseline compare)   pure in-kernel      (zero-copy RX ring,
                            drop benchmark)     userspace classify+stats)




xdp_pkt_lab/
  bpf/xdp_common.h
  bpf/xdp_kern.bpf.c
  loader/xdp_loader.c        # attach + run MODE_PASS_ALL or MODE_XDP_DROP, print stats
  af_xdp/xsk_app.c            # attach + MODE_AF_XDP_REDIR + AF_XDP consumer
  baseline/socket_app.c       # plain UDP socket receiver (comparison baseline)
  gen/udp_flood.c             # traffic generator w/ embedded send timestamp
  bench/bench_harness.sh
  Makefile