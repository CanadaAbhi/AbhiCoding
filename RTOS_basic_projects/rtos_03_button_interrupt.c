/*
 * RTOS Project 3: Button Interrupt → Task Notification
 * 
 * Demonstrates:
 * - ISR to task communication
 * - Task notifications (lightweight signaling)
 * - Interrupt latency measurement
 * - Event-driven programming
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <signal.h>
#include <string.h>

#define MAX_EVENTS 100

volatile bool running = true;
volatile bool button_pressed = false;
volatile int button_press_count = 0;

pthread_mutex_t notification_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t notification_cond = PTHREAD_COND_INITIALIZER;

// Timing statistics
struct {
    long long isr_times[MAX_EVENTS];
    long long task_times[MAX_EVENTS];
    int event_count;
} timing_stats = {0};

// Get current time in nanoseconds for precise measurement
long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Get current time in milliseconds
long long get_time_ms() {
    return get_time_ns() / 1000000;
}

// Simulated ISR (Interrupt Service Routine)
void button_isr() {
    long long isr_time = get_time_ns();
    
    button_press_count++;
    button_pressed = true;
    
    // Record ISR timing
    if (timing_stats.event_count < MAX_EVENTS) {
        timing_stats.isr_times[timing_stats.event_count] = isr_time;
    }
    
    printf("[%lld ms] ⚡ [ISR] Button pressed! Count: %d\n", 
           get_time_ms(), button_press_count);
    
    // Notify waiting task (minimal work in ISR)
    pthread_mutex_lock(&notification_mutex);
    pthread_cond_signal(&notification_cond);
    pthread_mutex_unlock(&notification_mutex);
}

// Button simulation thread (triggers interrupts)
void* button_simulator(void* param) {
    printf("[%lld ms] [SIMULATOR] Button simulator started\n", get_time_ms());
    printf("             Simulating random button presses...\n\n");
    
    srand(time(NULL));
    
    while (running) {
        // Random delay between button presses (1-5 seconds)
        int delay = (rand() % 4000) + 1000;
        usleep(delay * 1000);
        
        if (running) {
            button_isr();  // Trigger interrupt
        }
    }
    
    return NULL;
}

// Task that handles button press (deferred interrupt handling)
void* task_button_handler(void* param) {
    printf("[%lld ms] [HANDLER] Button handler task started\n", get_time_ms());
    
    while (running) {
        // Wait for notification from ISR
        pthread_mutex_lock(&notification_mutex);
        while (!button_pressed && running) {
            pthread_cond_wait(&notification_cond, &notification_mutex);
        }
        
        if (!running) {
            pthread_mutex_unlock(&notification_mutex);
            break;
        }
        
        long long task_time = get_time_ns();
        button_pressed = false;
        pthread_mutex_unlock(&notification_mutex);
        
        // Record task wake time
        int event_idx = timing_stats.event_count;
        if (event_idx < MAX_EVENTS) {
            timing_stats.task_times[event_idx] = task_time;
            
            // Calculate latency (ISR to task wake time)
            long long latency_ns = task_time - timing_stats.isr_times[event_idx];
            double latency_us = latency_ns / 1000.0;
            
            printf("[%lld ms] 🔧 [HANDLER] Processing button event\n", 
                   get_time_ms());
            printf("             Interrupt latency: %.2f µs\n", latency_us);
            
            timing_stats.event_count++;
        }
        
        // Simulate button debouncing and processing
        printf("             Debouncing...\n");
        usleep(50000);  // 50ms debounce
        
        // Perform button action
        printf("             Action: Toggle system state\n");
        printf("             Event processed successfully\n\n");
    }
    
    printf("[%lld ms] [HANDLER] Stopped\n", get_time_ms());
    return NULL;
}

// Statistics monitoring task
void* task_statistics(void* param) {
    printf("[%lld ms] [STATS] Statistics monitor started\n\n", get_time_ms());
    
    while (running) {
        sleep(10);
        
        if (timing_stats.event_count > 0) {
            printf("\n[%lld ms] ═══ INTERRUPT STATISTICS ═══\n", get_time_ms());
            printf("  Total events: %d\n", timing_stats.event_count);
            printf("  Button presses: %d\n", button_press_count);
            
            // Calculate average latency
            double total_latency = 0;
            double min_latency = 1e9;
            double max_latency = 0;
            
            for (int i = 0; i < timing_stats.event_count; i++) {
                double latency = (timing_stats.task_times[i] - 
                                 timing_stats.isr_times[i]) / 1000.0;  // µs
                total_latency += latency;
                if (latency < min_latency) min_latency = latency;
                if (latency > max_latency) max_latency = latency;
            }
            
            double avg_latency = total_latency / timing_stats.event_count;
            
            printf("  Interrupt Latency:\n");
            printf("    Average: %.2f µs\n", avg_latency);
            printf("    Min: %.2f µs\n", min_latency);
            printf("    Max: %.2f µs\n", max_latency);
            printf("═══════════════════════════════════\n\n");
        }
    }
    
    return NULL;
}

// Signal handler
void signal_handler(int sig) {
    printf("\n\n[SHUTDOWN] Stopping all tasks...\n");
    running = false;
    pthread_cond_broadcast(&notification_cond);
}

int main() {
    pthread_t thread_simulator, thread_handler, thread_stats;
    
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  RTOS Project 3: Button Interrupt → Task Notification ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    printf("Concepts demonstrated:\n");
    printf("  - ISR to task communication\n");
    printf("  - Task notifications (lightweight event signaling)\n");
    printf("  - Interrupt latency measurement\n");
    printf("  - Deferred interrupt handling\n");
    printf("  - Event-driven architecture\n\n");
    
    printf("Operation:\n");
    printf("  1. Button simulator triggers random interrupts\n");
    printf("  2. ISR notifies handler task (minimal work)\n");
    printf("  3. Handler task processes event (heavy work)\n");
    printf("  4. Latency is measured and reported\n\n");
    
    printf("Press Ctrl+C to stop\n");
    printf("════════════════════════════════════════════════════════\n\n");
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    
    // Create tasks
    pthread_create(&thread_simulator, NULL, button_simulator, NULL);
    pthread_create(&thread_handler, NULL, task_button_handler, NULL);
    pthread_create(&thread_stats, NULL, task_statistics, NULL);
    
    // Wait for completion
    pthread_join(thread_simulator, NULL);
    pthread_join(thread_handler, NULL);
    pthread_join(thread_stats, NULL);
    
    // Final statistics
    printf("\n═══ FINAL STATISTICS ═══\n");
    printf("Total events processed: %d\n", timing_stats.event_count);
    printf("Total button presses: %d\n", button_press_count);
    
    if (timing_stats.event_count > 0) {
        double total_latency = 0;
        for (int i = 0; i < timing_stats.event_count; i++) {
            total_latency += (timing_stats.task_times[i] - 
                             timing_stats.isr_times[i]) / 1000.0;
        }
        printf("Average latency: %.2f µs\n", 
               total_latency / timing_stats.event_count);
    }
    
    printf("\n[SHUTDOWN] Program terminated successfully\n");
    
    return 0;
}