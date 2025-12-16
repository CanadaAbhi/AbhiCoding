CLANG := clang
CC    := gcc
ARCH  := $(shell uname -m | sed 's/x86_64/x86/')

CFLAGS_BPF := -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -I/usr/include/$(shell uname -m)-linux-gnu
CFLAGS     := -O2 -Wall -g
LIBS       := -lbpf -lelf -lz -lxdp -pthread

all: bpf/xdp_kern.o loader/xdp_loader af_xdp/xsk_app baseline/socket_app gen/udp_flood

bpf/xdp_kern.o: bpf/xdp_kern.bpf.c bpf/xdp_common.h
	$(CLANG) $(CFLAGS_BPF) -c $< -o $@

loader/xdp_loader: loader/xdp_loader.c bpf/xdp_common.h
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

af_xdp/xsk_app: af_xdp/xsk_app.c bpf/xdp_common.h
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

baseline/socket_app: baseline/socket_app.c
	$(CC) $(CFLAGS) $< -o $@

gen/udp_flood: gen/udp_flood.c
	$(CC) $(CFLAGS) $< -o $@

# both binaries look for xdp_kern.o in CWD; symlink it in for convenience
xdp_kern.o: bpf/xdp_kern.o
	ln -sf bpf/xdp_kern.o xdp_kern.o

clean:
	rm -f bpf/*.o xdp_kern.o loader/xdp_loader af_xdp/xsk_app baseline/socket_app gen/udp_flood
