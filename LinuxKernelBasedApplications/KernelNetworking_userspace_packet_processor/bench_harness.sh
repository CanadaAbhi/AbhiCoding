#!/bin/bash
# bench_harness.sh -- run normal-socket, XDP-drop, and AF_XDP-redirect modes
# back-to-back against the same udp_flood generator, capturing pps/latency/
# throughput plus CPU utilization (mpstat) and NIC drop counters (ethtool -S).
set -e
IFACE=${1:-veth0}
DST_IP=${2:-10.0.0.2}
PPS=${3:-200000}
DURATION=${4:-10}

run_case () {
	local name=$1; shift
	echo "=== $name ==="
	mpstat -P ALL 1 $DURATION > /tmp/mpstat_$name.log &
	MPID=$!
	"$@" &
	APPID=$!
	./gen/udp_flood $DST_IP 9999 $PPS $DURATION
	wait $APPID 2>/dev/null || true
	wait $MPID 2>/dev/null || true
	echo "--- cpu (avg all) ---"
	tail -n 5 /tmp/mpstat_$name.log
	echo
}

echo "## normal socket (XDP MODE_PASS_ALL feeds netstack) ##"
./loader/xdp_loader "$IFACE" pass 1000 &
LOADER=$!
run_case socket ./baseline/socket_app $DURATION
kill $LOADER 2>/dev/null || true

echo "## pure XDP_DROP ceiling (no userspace at all) ##"
run_case xdp_drop ./loader/xdp_loader "$IFACE" drop $DURATION

echo "## AF_XDP zero-copy redirect ##"
run_case af_xdp ./af_xdp/xsk_app "$IFACE" 0 $DURATION

echo "## drop counters ##"
ethtool -S "$IFACE" 2>/dev/null | grep -i drop || echo "(ethtool stats unavailable on this iface)"
