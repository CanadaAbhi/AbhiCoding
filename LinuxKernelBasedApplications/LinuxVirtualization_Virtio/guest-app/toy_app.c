// toy_app.c -- "the application that communicates through a VirtIO device."
// Deliberately mirrors accel_lab's Buffer/submit_job()/wait_for_completion()
// API shape, but every byte here actually crosses: userspace -> ioctl ->
// virtio_toy.ko -> virtqueue split-ring -> KVM ioeventfd -> QEMU device
// model -> virtio_notify() -> KVM irqfd -> guest MSI-X ISR -> completion().
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>
#include "../guest-driver/virtio_toy.h"

#define NELEM 256

int main(void)
{
    int fd = open("/dev/virtio_toy0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    uint32_t input[NELEM], output[NELEM];
    for (int i = 0; i < NELEM; i++) input[i] = i;

    struct toy_ioc_submit args = {
        .op = TOY_OP_SQUARE,
        .scalar = 0,
        .nelem = NELEM,
        .in_uaddr = (uint64_t)(uintptr_t)input,
        .out_uaddr = (uint64_t)(uintptr_t)output,
    };

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (ioctl(fd, TOY_IOC_SUBMIT, &args) < 0) { perror("TOY_IOC_SUBMIT"); return 1; }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;

    int ok = 1;
    for (int i = 0; i < NELEM; i++)
        if (output[i] != (uint32_t)i * (uint32_t)i) { ok = 0; break; }

    printf("virtio-toy round-trip latency=%.1f us  result=%s\n", us, ok ? "CORRECT" : "MISMATCH");
    close(fd);
    return !ok;
}
