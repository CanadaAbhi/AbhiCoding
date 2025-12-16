#!/bin/bash
# scripts/build.sh
set -e
make -C driver           # builds kmem_lab_drv.ko via standard kbuild Makefile
gcc -O2 -o bench_alloc bench/bench_alloc.c lib/libkmem.c
echo "Built kmem_lab_drv.ko and bench_alloc"
