#pragma once
#include <linux/types.h>

int  mygpu_sched_start(void);
void mygpu_sched_stop(void);
void mygpu_sched_submit(size_t size);
