// accel_lib.c
#include "accel_lib.h"
#include "../include/accel_uapi.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int accel_open_device(void)
{
	int fd = open("/dev/accel0", O_RDWR);
	if (fd < 0) perror("open /dev/accel0");
	return fd;
}

Buffer accel_buffer_alloc(int fd, size_t size)
{
	struct accel_buffer_alloc a = { .size = size };
	Buffer b = {0};

	if (ioctl(fd, ACCEL_IOC_BUFFER_ALLOC, &a) < 0) { perror("ALLOC"); return b; }
	b.handle = a.handle;
	b.size = size;
	b.vaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, a.mmap_offset);
	if (b.vaddr == MAP_FAILED) { perror("mmap"); b.vaddr = NULL; }
	return b;
}

void accel_buffer_free(int fd, Buffer *buf)
{
	struct accel_buffer_free f = { .handle = buf->handle };
	if (buf->vaddr) munmap(buf->vaddr, buf->size);
	ioctl(fd, ACCEL_IOC_BUFFER_FREE, &f);
}

uint64_t submit_job(int fd, Buffer *input, Buffer *output, uint32_t opcode, int32_t scalar)
{
	struct accel_submit s = {
		.input_handle = input->handle, .input_len = input->size,
		.output_handle = output->handle, .output_len = output->size,
		.opcode = opcode, .scalar = scalar, .priority = ACCEL_PRIO_NORMAL,
	};
	if (ioctl(fd, ACCEL_IOC_SUBMIT, &s) < 0) { perror("SUBMIT"); return (uint64_t)-1; }
	return s.out_seqno;
}

int wait_for_completion(int fd, uint64_t seqno, uint64_t timeout_ns)
{
	struct accel_wait w = { .seqno = seqno, .timeout_ns = timeout_ns };
	if (ioctl(fd, ACCEL_IOC_WAIT, &w) < 0) { perror("WAIT"); return -1; }
	return w.out_status; /* 0=ok 1=timeout 2=error */
}
