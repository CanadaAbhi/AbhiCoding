/* RTOS Project 7: Sensor Data Pipeline */
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

int sensor_data = 0;
int filtered_data = 0;
pthread_mutex_t sensor_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t filter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* task_sensor(void* param) {
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&sensor_mutex);
        sensor_data = rand() % 100;
        printf("[SENSOR] Read: %d\n", sensor_data);
        pthread_mutex_unlock(&sensor_mutex);
        usleep(200000);
    }
    return NULL;
}

void* task_filter(void* param) {
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&sensor_mutex);
        int data = sensor_data;
        pthread_mutex_unlock(&sensor_mutex);
        
        pthread_mutex_lock(&filter_mutex);
        filtered_data = (data + filtered_data) / 2;
        printf("[FILTER] Filtered: %d -> %d\n", data, filtered_data);
        pthread_mutex_unlock(&filter_mutex);
        usleep(200000);
    }
    return NULL;
}

void* task_transmit(void* param) {
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&filter_mutex);
        printf("[TRANSMIT] Sending: %d\n", filtered_data);
        pthread_mutex_unlock(&filter_mutex);
        usleep(300000);
    }
    return NULL;
}

int main() {
    printf("RTOS Project 7: Sensor Data Pipeline\n\n");
    pthread_t t1, t2, t3;
    srand(time(NULL));
    
    pthread_create(&t1, NULL, task_sensor, NULL);
    pthread_create(&t2, NULL, task_filter, NULL);
    pthread_create(&t3, NULL, task_transmit, NULL);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    return 0;
}