CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LIBS = -lGL -lGLEW -lglfw -lm

TARGET = shader_task_scheduler
SRC = shader_task_scheduler.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS)
	@echo "Build complete!"
	@echo ""
	@echo "Run with: ./$(TARGET)"

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

info:
	@echo "Shader Task Scheduler"
	@echo "  Particles: 10000"
	@echo "  Work Group Size: 128"
	@echo "  Requires: OpenGL 4.3+ (compute shader support)"

.PHONY: all debug clean run info