#pragma once
#include <stddef.h>

int   mygpu_open(void);
void *mygpu_alloc(size_t size);
int   mygpu_submit(const void *cmd, size_t size);
void  mygpu_close(void);
