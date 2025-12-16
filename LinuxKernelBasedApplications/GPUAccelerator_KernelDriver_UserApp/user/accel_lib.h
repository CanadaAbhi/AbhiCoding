// accel_lib.h
#ifndef ACCEL_LIB_H
#define ACCEL_LIB_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
	uint32_t handle;
	size_t size;
	void *vaddr;
} Buffer;

int accel_open_device(void);
Buffer accel_buffer_alloc(int fd, size_t size);
void accel_buffer_free(int fd, Buffer *buf);
uint64_t submit_job(int fd, Buffer *input, Buffer *output, uint32_t opcode, int32_t scalar);
int wait_for_completion(int fd, uint64_t seqno, uint64_t timeout_ns);

#endif
