import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import requests

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mqtt = None  # graceful fallback if MQTT is not installed


@dataclass
class AccidentAlert:
    """
    Canonical VANET / V2X accident alert, matching the project document.

    {
      "event_id": "uuid",
      "type": "accident",
      "timestamp": "UTC time",
      "location": {"lat": xx.x, "lon": yy.y},
      "confidence": 0.92,
      "camera_id": "cam42"
    }
    """

    event_id: str
    type: str
    timestamp: str
    location: Dict[str, float]
    confidence: float
    camera_id: str


def parse_location_string(loc: str) -> Optional[Dict[str, float]]:
    """Convert \"lat,lon\" string to a location dict."""
    try:
        lat_str, lon_str = (loc or "").split(",", 1)
        return {"lat": float(lat_str.strip()), "lon": float(lon_str.strip())}
    except Exception:
        return None


def build_accident_alert(
    location_str: Optional[str],
    confidence: float,
    camera_id: Optional[str] = None,
) -> AccidentAlert:
    if not camera_id:
        camera_id = os.environ.get("CAMERA_ID", socket.gethostname())

    loc = parse_location_string(location_str or "") or {"lat": 0.0, "lon": 0.0}

    return AccidentAlert(
        event_id=str(uuid.uuid4()),
        type="accident",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        location=loc,
        confidence=float(confidence),
        camera_id=camera_id,
    )


def alert_to_json(alert: AccidentAlert) -> str:
    return json.dumps(asdict(alert), separators=(",", ":"), sort_keys=False)


def publish_mqtt_alert(
    alert: AccidentAlert,
    topic: str = "saferoadai/alerts",
    broker_host: Optional[str] = None,
    broker_port: int = 1883,
    username: Optional[str] = None,
    password: Optional[str] = None,
    qos: int = 1,
) -> bool:
    """
    Publish alert over MQTT to emulate VANET/V2X broadcast.

    Uses PAHO MQTT if available. If MQTT is not installed or broker settings
    are missing, the function becomes a no-op and returns False.
    """
    if mqtt is None:
        return False

    broker_host = broker_host or os.environ.get("MQTT_BROKER", "localhost")
    broker_port = int(os.environ.get("MQTT_PORT", broker_port))
    username = username or os.environ.get("MQTT_USERNAME") or None
    password = password or os.environ.get("MQTT_PASSWORD") or None

    client = mqtt.Client()
    if username or password:
        client.username_pw_set(username, password)

    try:
        client.connect(broker_host, broker_port, keepalive=30)
    except Exception:
        return False

    payload = alert_to_json(alert)
    result = client.publish(topic, payload, qos=qos)
    client.disconnect()
    return result.rc == 0


def demo_http_broadcast(alert: AccidentAlert) -> Optional[requests.Response]:
    """
    Optional helper: send alert to an HTTP endpoint to integrate with
    external dashboards / RSUs / simulations.

    If SAFEROAD_V2X_ENDPOINT is not set, this function is a no-op.
    """
    endpoint = os.environ.get("SAFEROAD_V2X_ENDPOINT")
    if not endpoint:
        return None

    try:
        return requests.post(
            endpoint,
            json=asdict(alert),
            timeout=3,
        )
    except Exception:
        return None

