# Makefile for all RTOS Projects
# Compiles all 12 RTOS demonstration programs

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pthread
LIBS = -pthread -lrt -lm

# All RTOS projects
PROJECTS = \
	rtos_01_led_blink \
	rtos_02_producer_consumer \
	rtos_03_button_interrupt \
	rtos_04_timer_vs_task \
	rtos_05_mutex_vs_semaphore \
	rtos_06_logger \
	rtos_07_sensor_pipeline

all: $(PROJECTS)
	@echo ""
	@echo "✓ All RTOS projects built successfully!"
	@echo ""
	@echo "Available programs:"
	@for prog in $(PROJECTS); do \
		echo "  ./$$prog"; \
	done
	@echo ""

# Project 1: LED Blinking
rtos_01_led_blink: rtos_01_led_blink.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 2: Producer-Consumer
rtos_02_producer_consumer: rtos_02_producer_consumer.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 3: Button Interrupt
rtos_03_button_interrupt: rtos_03_button_interrupt.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 4: Timer vs Task
rtos_04_timer_vs_task: rtos_04_timer_vs_task.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 5: Mutex vs Semaphore
rtos_05_mutex_vs_semaphore: rtos_05_mutex_vs_semaphore.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 6: Logger
rtos_06_logger: rtos_06_logger.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Project 7: Sensor Pipeline
rtos_07_sensor_pipeline: rtos_07_sensor_pipeline.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
	@echo "✓ Built: $@"

# Individual run targets
run1: rtos_01_led_blink
	./rtos_01_led_blink

run2: rtos_02_producer_consumer
	./rtos_02_producer_consumer

run3: rtos_03_button_interrupt
	./rtos_03_button_interrupt

run4: rtos_04_timer_vs_task
	./rtos_04_timer_vs_task

run5: rtos_05_mutex_vs_semaphore
	./rtos_05_mutex_vs_semaphore

run6: rtos_06_logger
	./rtos_06_logger

run7: rtos_07_sensor_pipeline
	./rtos_07_sensor_pipeline

# Clean all
clean:
	rm -f $(PROJECTS) rtos_log.txt
	@echo "✓ Cleaned all binaries"

# Show help
help:
	@echo "RTOS Projects Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all projects"
	@echo "  run1     - Run Project 1 (LED Blinking)"
	@echo "  run2     - Run Project 2 (Producer-Consumer)"
	@echo "  run3     - Run Project 3 (Button Interrupt)"
	@echo "  run4     - Run Project 4 (Timer vs Task)"
	@echo "  run5     - Run Project 5 (Mutex vs Semaphore)"
	@echo "  run6     - Run Project 6 (Logger)"
	@echo "  run7     - Run Project 7 (Sensor Pipeline)"
	@echo "  clean    - Remove all binaries"
	@echo "  help     - Show this help"

.PHONY: all clean help run1 run2 run3 run4 run5 run6 run7