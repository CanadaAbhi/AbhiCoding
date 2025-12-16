#!/bin/bash
# Guest 1: virtio-net (vhost=on -> kernel vhost-net backend handles the
# datapath; QEMU userspace only sets up control-plane + ioeventfd/irqfd).
qemu-system-x86_64 \
  -enable-kvm -m 2G -smp 2 \
  -M q35 \
  -kernel bzImage -initrd initrd.img -append "console=ttyS0 root=/dev/vda" \
  -netdev tap,id=net0,ifname=tap0,script=no,downscript=no,vhost=on \
  -device virtio-net-pci,netdev=net0,mq=on,vectors=4 \
  -device virtio-toy-pci \
  -nographic
