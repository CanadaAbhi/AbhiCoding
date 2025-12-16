#!/bin/bash
# Quantifies the ioeventfd/irqfd win: vmexits-per-request before/after
# correct notification wiring.
GUEST_PID=$(pgrep -f "launch_guest1_net")
echo "== baseline (vhost=off, no ioeventfd bypass) =="
perf kvm --host --guest -p $GUEST_PID stat live -e vmexit -- sleep 10

echo "== with vhost=on + ioeventfd/irqfd (current launch config) =="
perf kvm --host --guest -p $GUEST_PID stat live -e vmexit,mmio,exit_reason -- sleep 10
