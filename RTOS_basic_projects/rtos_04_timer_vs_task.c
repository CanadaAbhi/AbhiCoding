/*
 * RTOS Project 4: Software Timers vs Tasks Comparison
 * 
 * Demonstrates:
 * - Software timers for periodic jobs
 * - Comparison with task-based periodic execution
 * - Jitter measurement and timing accuracy
 * - Timer callback vs task polling
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <signal.h>
#include <sys/time.h>

#define MAX_MEASUREMENTS 100

volatile bool running = true;

// Jitter measurement structures
struct TimingData {
    long long timestamps[MAX_MEASUREMENTS];
    long long expected_times[MAX_MEASUREMENTS];
    int count;
    long long start_time;
} timer_data = {0}, task_data = {0};

long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

long long get_time_ms() {
    return get_time_ns() / 1000000;
}

// Calculate jitter statistics
void calculate_jitter(struct TimingData* data, const char* name) {
    if (data->count < 2) return;
    
    double total_jitter = 0;
    double max_jitter = 0;
    double min_jitter = 1e9;
    
    printf("\n═══ %s JITTER ANALYSIS ═══\n", name);
    printf("Measurements: %d\n", data->count);
    
    for (int i = 1; i < data->count; i++) {
        long long expected = data->expected_times[i];
        long long actual = data->timestamps[i];
        long long jitter_ns = actual - expected;
        double jitter_us = jitter_ns / 1000.0;
        
        total_jitter += fabs(jitter_us);
        if (fabs(jitter_us) > max_jitter) max_jitter = fabs(jitter_us);
        if (fabs(jitter_us) < min_jitter) min_jitter = fabs(jitter_us);
        
        if (i < 10) {  // Show first 10
            printf("  [%d] Jitter: %+.2f µs\n", i, jitter_us);
        }
    }
    
    printf("  Average jitter: %.2f µs\n", total_jitter / (data->count - 1));
    printf("  Max jitter: %.2f µs\n", max_jitter);
    printf("  Min jitter: %.2f µs\n", min_jitter);
    printf("═══════════════════════════════════\n\n");
}

// Software timer callback
void timer_callback(union sigval arg) {
    long long now = get_time_ns();
    
    if (timer_data.count == 0) {
        timer_data.start_time = now;
    }
    
    if (timer_data.count < MAX_MEASUREMENTS) {
        timer_data.timestamps[timer_data.count] = now;
        
        // Calculate expected time (100ms intervals)
        long long expected = timer_data.start_time + 
                            (timer_data.count * 100000000LL);
        timer_data.expected_times[timer_data.count] = expected;
        
        long long jitter_ns = now - expected;
        
        printf("[%lld ms] ⏰ [TIMER] Callback #%d | Jitter: %+.2f µs\n",
               get_time_ms(), timer_data.count, jitter_ns / 1000.0);
        
        timer_data.count++;
    }
}

// Setup software timer
timer_t setup_timer(int interval_ms) {
    timer_t timerid;
    struct sigevent sev;
    struct itimerspec its;
    
    // Create timer
    sev.sigev_notify = SIGEV_THREAD;
    sev.sigev_notify_function = timer_callback;
    sev.sigev_notify_attributes = NULL;
    sev.sigev_value.sival_ptr = &timerid;
    
    timer_create(CLOCK_MONOTONIC, &sev, &timerid);
    
    // Start timer
    its.it_value.tv_sec = interval_ms / 1000;
    its.it_value.tv_nsec = (interval_ms % 1000) * 1000000;
    its.it_interval.tv_sec = its.it_value.tv_sec;
    its.it_interval.tv_nsec = its.it_value.tv_nsec;
    
    timer_settime(timerid, 0, &its, NULL);
    
    return timerid;
}

// Task-based periodic execution
void* task_periodic(void* param) {
    int period_ms = 100;
    long long now;
    
    printf("[%lld ms] [TASK] Periodic task started (100ms period)\n", 
           get_time_ms());
    
    task_data.start_time = get_time_ns();
    
    while (running && task_data.count < MAX_MEASUREMENTS) {
        usleep(period_ms * 1000);
        
        now = get_time_ns();
        task_data.timestamps[task_data.count] = now;
        
        // Calculate expected time
        long long expected = task_data.start_time + 
                            (task_data.count * 100000000LL);
        task_data.expected_times[task_data.count] = expected;
        
        long long jitter_ns = now - expected;
        
        printf("[%lld ms] 📋 [TASK] Execution #%d | Jitter: %+.2f µs\n",
               get_time_ms(), task_data.count, jitter_ns / 1000.0);
        
        task_data.count++;
    }
    
    printf("[%lld ms] [TASK] Stopped\n", get_time_ms());
    return NULL;
}

// Monitor task
void* task_monitor(void* param) {
    while (running) {
        sleep(5);
        
        printf("\n[%lld ms] ═══ STATUS ═══\n", get_time_ms());
        printf("  Timer callbacks: %d/%d\n", timer_data.count, MAX_MEASUREMENTS);
        printf("  Task executions: %d/%d\n", task_data.count, MAX_MEASUREMENTS);
        printf("═══════════════════\n\n");
        
        if (timer_data.count >= MAX_MEASUREMENTS && 
            task_data.count >= MAX_MEASUREMENTS) {
            running = false;
        }
    }
    return NULL;
}

void signal_handler(int sig) {
    printf("\n\n[SHUTDOWN] Stopping...\n");
    running = false;
}

int main() {
    pthread_t thread_task, thread_monitor;
    timer_t timer;
    
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  RTOS Project 4: Software Timers vs Tasks Comparison  ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    printf("Concepts demonstrated:\n");
    printf("  - Software timers with callbacks\n");
    printf("  - Task-based periodic execution\n");
    printf("  - Jitter measurement and analysis\n");
    printf("  - Timing accuracy comparison\n\n");
    
    printf("Test configuration:\n");
    printf("  Period: 100ms (both methods)\n");
    printf("  Measurements: %d samples each\n", MAX_MEASUREMENTS);
    printf("  Metrics: Timing jitter in microseconds\n\n");
    
    printf("Starting test...\n");
    printf("════════════════════════════════════════════════════════\n\n");
    
    signal(SIGINT, signal_handler);
    
    // Create software timer
    timer = setup_timer(100);  // 100ms period
    
    // Create periodic task
    pthread_create(&thread_task, NULL, task_periodic, NULL);
    
    // Create monitor
    pthread_create(&thread_monitor, NULL, task_monitor, NULL);
    
    // Wait for completion
    pthread_join(thread_task, NULL);
    pthread_join(thread_monitor, NULL);
    
    // Stop timer
    timer_delete(timer);
    
    // Calculate and display jitter statistics
    calculate_jitter(&timer_data, "SOFTWARE TIMER");
    calculate_jitter(&task_data, "PERIODIC TASK");
    
    // Comparison
    printf("═══ COMPARISON ═══\n");
    if (timer_data.count > 0 && task_data.count > 0) {
        double timer_avg = 0, task_avg = 0;
        
        for (int i = 1; i < timer_data.count; i++) {
            timer_avg += fabs((timer_data.timestamps[i] - 
                              timer_data.expected_times[i]) / 1000.0);
        }
        timer_avg /= (timer_data.count - 1);
        
        for (int i = 1; i < task_data.count; i++) {
            task_avg += fabs((task_data.timestamps[i] - 
                             task_data.expected_times[i]) / 1000.0);
        }
        task_avg /= (task_data.count - 1);
        
        printf("Average jitter:\n");
        printf("  Timer: %.2f µs\n", timer_avg);
        printf("  Task:  %.2f µs\n", task_avg);
        printf("\nWinner: %s (%.1f%% better)\n",
               timer_avg < task_avg ? "SOFTWARE TIMER" : "PERIODIC TASK",
               fabs((task_avg - timer_avg) / fmax(timer_avg, task_avg)) * 100);
    }
    printf("═══════════════════\n\n");
    
    printf("[SHUTDOWN] Test completed successfully\n");
    return 0;
}