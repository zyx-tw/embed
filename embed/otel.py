# OTLP/HTTP JSON logs for sensorium. Off unless OTEL_EXPORTER_OTLP_ENDPOINT is
# set. Records carry routes, counts and timings, never images, text or tokens.
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

ENDPOINT = (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").rstrip("/")
SERVICE = os.environ.get("OTEL_SERVICE_NAME", "embed")
HEADERS = {"content-type": "application/json"}
for pair in (os.environ.get("OTEL_EXPORTER_OTLP_HEADERS") or "").split(","):
    key, _, value = pair.partition("=")
    if key.strip() and value:
        HEADERS[urllib.parse.unquote(key.strip())] = urllib.parse.unquote(value.strip())

_queue = []
_lock = threading.Lock()


def _attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": value}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def log(event, attributes=None, severity="INFO"):
    fields = {"event": event, **(attributes or {})}
    print(json.dumps(fields), file=sys.stdout if severity == "INFO" else sys.stderr, flush=True)
    if not ENDPOINT:
        return
    with _lock:
        _queue.append(
            {
                "timeUnixNano": str(time.time_ns()),
                "severityText": severity,
                "body": {"stringValue": event},
                "attributes": [_attr(k, v) for k, v in fields.items() if v is not None],
            }
        )


def flush():
    global _queue
    with _lock:
        records, _queue = _queue, []
    if not ENDPOINT or not records:
        return
    body = {
        "resourceLogs": [
            {
                "resource": {"attributes": [_attr("service.name", SERVICE)]},
                "scopeLogs": [{"scope": {"name": "embed"}, "logRecords": records}],
            }
        ]
    }
    request = urllib.request.Request(f"{ENDPOINT}/v1/logs", data=json.dumps(body).encode(), headers=HEADERS, method="POST")
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except Exception as error:  # noqa: BLE001 - telemetry never takes the service down
        print(json.dumps({"event": "otel.export_failed", "error": str(error)[:200]}), file=sys.stderr, flush=True)


def _loop():
    while True:
        time.sleep(5)
        flush()


if ENDPOINT:
    threading.Thread(target=_loop, daemon=True).start()
