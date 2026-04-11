import json
import os

import paho.mqtt.client as mqtt


TOPIC = os.environ.get("SAFEROAD_TOPIC", "saferoadai/alerts")
BROKER = os.environ.get("MQTT_BROKER", "localhost")
PORT = int(os.environ.get("MQTT_PORT", 1883))
USERNAME = os.environ.get("MQTT_USERNAME") or None
PASSWORD = os.environ.get("MQTT_PASSWORD") or None


def on_connect(client, userdata, flags, reason_code, properties=None):  # type: ignore[override]
    if reason_code == 0:
        print(f"✅ Connected to MQTT broker at {BROKER}:{PORT}, subscribing to '{TOPIC}'")
        client.subscribe(TOPIC, qos=1)
    else:
        print(f"❌ Failed to connect, reason_code={reason_code}")


def on_message(client, userdata, msg):  # type: ignore[override]
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
    except Exception:
        print(f"[{msg.topic}] raw -> {msg.payload!r}")
        return

    print("\n🚗 Received VANET alert:")
    print(json.dumps(data, indent=2))


def main() -> None:
    client = mqtt.Client()
    if USERNAME or PASSWORD:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()

