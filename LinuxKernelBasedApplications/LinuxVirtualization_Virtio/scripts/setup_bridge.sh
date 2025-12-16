#!/bin/bash
# Host-side network back-end for Guest 1's virtio-net (the "Host Driver" box).
set -e
sudo ip link add br0 type bridge
sudo ip link set br0 up
sudo ip tuntap add dev tap0 mode tap
sudo ip link set tap0 master br0
sudo ip link set tap0 up
sudo ip addr add 192.168.100.1/24 dev br0
# vhost-net: kernel-side backend that services virtio-net without QEMU
# userspace mediating each packet -- load it explicitly for the demo:
sudo modprobe vhost_net
