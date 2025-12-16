#!/bin/bash
# bench_compare.sh -- reload irq_defer_lab in each defer_mode and capture
# comparable P50/P95/P99 numbers for workqueue vs tasklet vs threaded_irq.
set -e
cd "$(dirname "$0")/kernel"
sudo insmod hw_event_gen.ko 2>/dev/null || true

for MODE in 0 1 2; do
	NAME=("workqueue" "tasklet" "threaded_irq")
	echo "=== mode=$MODE (${NAME[$MODE]}) ==="
	sudo insmod irq_defer_lab.ko defer_mode=$MODE
	../app/irq_defer_app 10
	sudo rmmod irq_defer_lab
	echo
done

sudo rmmod hw_event_gen
