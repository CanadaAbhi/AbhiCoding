#!/bin/bash
# Guest 2: virtio-blk backed by a raw/qcow2 image, iothread offloads the
# blk request processing off the main QEMU thread (closer to real vhost-blk).
qemu-img create -f qcow2 disk2.qcow2 4G
qemu-system-x86_64 \
  -enable-kvm -m 2G -smp 2 \
  -M q35 \
  -kernel bzImage -initrd initrd.img -append "console=ttyS0 root=/dev/vda" \
  -object iothread,id=io1 \
  -drive file=disk2.qcow2,if=none,id=d0,format=qcow2,cache=none,aio=native \
  -device virtio-blk-pci,drive=d0,iothread=io1 \
  -device virtio-toy-pci \
  -nographic
