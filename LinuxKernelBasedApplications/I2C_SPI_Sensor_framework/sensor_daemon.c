// sensor_daemon.c -- poll both sensor_fw char devices, threshold alarm,
// broadcast JSON to local monitoring_ui clients over a UNIX socket, and
// publish to an MQTT broker for cloud integration.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <pthread.h>
#include <syslog.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <mosquitto.h>
#include "sensor_fw_uapi.h"

#define TEMP_DEV "/dev/sensor_fw/temp0"
#define IMU_DEV  "/dev/sensor_fw/imu_spi0"
#define TEMP_THRESHOLD_MC 60000   /* 60.000 C, in milli-C */
#define UI_SOCK_PATH "/tmp/sensor_daemon.sock"
#define MQTT_HOST "localhost"
#define MQTT_PORT 1883

struct sensor_daemon {
	int temp_fd, imu_fd;
	int ui_listen_fd;
	int ui_clients[8];
	pthread_mutex_t ui_lock;

	struct mosquitto *mqtt;

	int32_t last_temp_mC;
	int32_t last_imu[6];
	int alarm_active;
};

static struct sensor_daemon g;

/* ---- local monitoring-UI broadcast server ---- */
static void *ui_server_thread(void *arg)
{
	struct sockaddr_un addr = { .sun_family = AF_UNIX };
	strncpy(addr.sun_path, UI_SOCK_PATH, sizeof(addr.sun_path) - 1);
	unlink(UI_SOCK_PATH);

	g.ui_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	bind(g.ui_listen_fd, (struct sockaddr *)&addr, sizeof(addr));
	listen(g.ui_listen_fd, 4);

	for (;;) {
		int cfd = accept(g.ui_listen_fd, NULL, NULL);
		if (cfd < 0)
			continue;
		pthread_mutex_lock(&g.ui_lock);
		for (int i = 0; i < 8; i++) {
			if (g.ui_clients[i] == 0) {
				g.ui_clients[i] = cfd;
				break;
			}
		}
		pthread_mutex_unlock(&g.ui_lock);
	}
	return NULL;
}

static void ui_broadcast(const char *json_line)
{
	pthread_mutex_lock(&g.ui_lock);
	for (int i = 0; i < 8; i++) {
		if (g.ui_clients[i] > 0) {
			if (write(g.ui_clients[i], json_line, strlen(json_line)) < 0) {
				close(g.ui_clients[i]);
				g.ui_clients[i] = 0;
			}
		}
	}
	pthread_mutex_unlock(&g.ui_lock);
}

/* ---- alarm + MQTT ---- */
static void trigger_alarm(int32_t temp_mC)
{
	char payload[128];

	syslog(LOG_WARNING, "ALARM: temperature %d mC exceeds threshold %d mC",
	       temp_mC, TEMP_THRESHOLD_MC);
	g.alarm_active = 1;

	snprintf(payload, sizeof(payload),
		 "{\"event\":\"temp_high\",\"temp_mC\":%d,\"threshold_mC\":%d,\"ts\":%ld}",
		 temp_mC, TEMP_THRESHOLD_MC, (long)time(NULL));

	if (g.mqtt)
		mosquitto_publish(g.mqtt, NULL, "acme/alarms/temp",
				   strlen(payload), payload, 1 /*qos*/, true /*retain*/);
}

static void publish_snapshot(void)
{
	char payload[256];
	int n;

	if (!g.mqtt)
		return;

	n = snprintf(payload, sizeof(payload), "{\"temp_mC\":%d,\"alarm\":%d,\"ts\":%ld}",
		     g.last_temp_mC, g.alarm_active, (long)time(NULL));
	mosquitto_publish(g.mqtt, NULL, "acme/sensors/temp", n, payload, 0, false);

	n = snprintf(payload, sizeof(payload),
		     "{\"ax\":%d,\"ay\":%d,\"az\":%d,\"gx\":%d,\"gy\":%d,\"gz\":%d,\"ts\":%ld}",
		     g.last_imu[0], g.last_imu[1], g.last_imu[2],
		     g.last_imu[3], g.last_imu[4], g.last_imu[5], (long)time(NULL));
	mosquitto_publish(g.mqtt, NULL, "acme/sensors/imu", n, payload, 0, false);
}

static void handle_temp_sample(const struct sensor_fw_sample *s)
{
	g.last_temp_mC = s->chan[0] * 10; /* centi-C -> milli-C, per driver scale_milli */
	if (g.last_temp_mC > TEMP_THRESHOLD_MC)
		trigger_alarm(g.last_temp_mC);
	else
		g.alarm_active = 0;
}

static void handle_imu_sample(const struct sensor_fw_sample *s)
{
	memcpy(g.last_imu, s->chan, sizeof(g.last_imu));
}

int main(void)
{
	struct pollfd pfds[2];
	pthread_t ui_thread;
	time_t last_publish = 0;

	openlog("sensor_daemon", LOG_PID | LOG_CONS, LOG_DAEMON);
	pthread_mutex_init(&g.ui_lock, NULL);

	g.temp_fd = open(TEMP_DEV, O_RDONLY);
	g.imu_fd  = open(IMU_DEV, O_RDONLY);
	if (g.temp_fd < 0 || g.imu_fd < 0) {
		syslog(LOG_ERR, "failed to open sensor devices");
		return 1;
	}

	mosquitto_lib_init();
	g.mqtt = mosquitto_new("sensor_daemon", true, NULL);
	if (g.mqtt) {
		if (mosquitto_connect(g.mqtt, MQTT_HOST, MQTT_PORT, 60) == MOSQ_ERR_SUCCESS)
			mosquitto_loop_start(g.mqtt);
		else
			syslog(LOG_WARNING, "MQTT broker unreachable, continuing without cloud publish");
	}

	pthread_create(&ui_thread, NULL, ui_server_thread, NULL);

	syslog(LOG_INFO, "sensor_daemon running, threshold=%d mC", TEMP_THRESHOLD_MC);

	/* ============ the requested core loop, event-driven via poll() ============ */
	int running = 1;
	while (running) {
		struct sensor_fw_sample sample;

		pfds[0] = (struct pollfd){ .fd = g.temp_fd, .events = POLLIN };
		pfds[1] = (struct pollfd){ .fd = g.imu_fd,  .events = POLLIN };

		int ret = poll(pfds, 2, 1000 /* ms: also drives periodic MQTT publish */);
		if (ret < 0) {
			if (errno == EINTR) continue;
			break;
		}

		if (pfds[0].revents & POLLIN) {
			if (read(g.temp_fd, &sample, sizeof(sample)) == sizeof(sample))
				handle_temp_sample(&sample);   /* data.temperature > threshold -> trigger_alarm() */
		}
		if (pfds[1].revents & POLLIN) {
			if (read(g.imu_fd, &sample, sizeof(sample)) == sizeof(sample))
				handle_imu_sample(&sample);
		}

		time_t now = time(NULL);
		if (now != last_publish) {
			char line[256];
			int n = snprintf(line, sizeof(line),
				"{\"temp_mC\":%d,\"alarm\":%d,\"ax\":%d,\"ay\":%d,\"az\":%d}\n",
				g.last_temp_mC, g.alarm_active,
				g.last_imu[0], g.last_imu[1], g.last_imu[2]);
			(void)n;
			ui_broadcast(line);
			publish_snapshot();
			last_publish = now;
		}
	}

	mosquitto_loop_stop(g.mqtt, true);
	mosquitto_destroy(g.mqtt);
	mosquitto_lib_cleanup();
	close(g.temp_fd);
	close(g.imu_fd);
	closelog();
	return 0;
}
