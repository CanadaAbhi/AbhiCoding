CC = gcc
CFLAGS = -Wall -Wextra -std=c11
LIBS = -lGL -lGLEW -lglfw -lpthread -lm

TARGET = thread_renderer
SRC = thread.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run