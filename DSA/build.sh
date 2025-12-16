#!/bin/bash
set -e
SINGLE="reverse_list cycle_detect my_memcpy my_memmove my_strlen my_strcpy \
        circular_buffer ring_buffer_spsc stack queue hash_table lru_cache \
        find_duplicate binary_search merge_sorted fixed_allocator"
THREADED="producer_consumer thread_safe_queue thread_pool atomic_counter"

for f in $SINGLE; do
    gcc -Wall -Wextra -std=c11 -O2 "$f.c" -o "$f"
    echo "Built $f"
done

for f in $THREADED; do
    gcc -Wall -Wextra -std=c11 -O2 -pthread "$f.c" -o "$f"
    echo "Built $f"
done
