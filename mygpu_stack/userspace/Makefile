CC      := gcc
CFLAGS  := -Wall -Wextra -O2
LDFLAGS :=

TARGET  := test_triangle
OBJS    := libmygpu.o test_triangle.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

libmygpu.o: libmygpu.c libmygpu.h
	$(CC) $(CFLAGS) -c libmygpu.c

test_triangle.o: test_triangle.c libmygpu.h
	$(CC) $(CFLAGS) -c test_triangle.c

clean:
	rm -f $(TARGET) *.o
