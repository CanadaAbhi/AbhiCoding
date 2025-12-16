#!/bin/bash
# scripts/run_bench.sh
set -e
sudo insmod driver/kmem_lab_drv.ko
sudo chmod 666 /dev/kmem_lab
./bench_alloc
sudo rmmod kmem_lab_drv
