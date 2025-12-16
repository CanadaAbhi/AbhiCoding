// cloud_subscriber.c -- represents the "cloud" side: subscribes to the same
// MQTT topics sensor_daemon publishes, independent of the local UI.
#include <stdio.h>
#include <string.h>
#include <mosquitto.h>

static void on_message(struct mosquitto *m, void *ud, const struct mosquitto_message *msg)
{
	printf("[cloud] %s => %.*s\n", msg->topic, msg->payloadlen, (char *)msg->payload);
	if (strcmp(msg->topic, "acme/alarms/temp") == 0)
		printf("[cloud] ** ALARM notification received, would trigger cloud action (e.g. SNS/webhook) **\n");
}

int main(void)
{
	struct mosquitto *m;

	mosquitto_lib_init();
	m = mosquitto_new("cloud_subscriber", true, NULL);
	mosquitto_message_callback_set(m, on_message);

	if (mosquitto_connect(m, "localhost", 1883, 60) != MOSQ_ERR_SUCCESS) {
		fprintf(stderr, "cannot connect to broker\n");
		return 1;
	}
	mosquitto_subscribe(m, NULL, "acme/sensors/#", 0);
	mosquitto_subscribe(m, NULL, "acme/alarms/#", 1);

	mosquitto_loop_forever(m, -1, 1);

	mosquitto_destroy(m);
	mosquitto_lib_cleanup();
	return 0;
}
