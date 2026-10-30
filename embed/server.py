# embed: one HTTP service that turns text and photos into EmbeddingGemma 2
# vectors, finds faces and reads text in images, for nas and roll. It keeps
# nothing: every request brings its input and takes its answer back.
#
# GET  /        -> {"ok": true, "vision": "openvino-gpu" | "torch-cpu", "faces": bool, "ocr": bool}
# POST /embed   {"kind": "query" | "document", "items": [{"text": ...} | {"image_b64": ...}]}
#               -> {"vectors": [[768 floats], ...]}
# POST /faces   {"image_b64": ...}
#               -> {"width", "height", "faces": [{"box": [x1, y1, x2, y2], "score", "vector": [512 floats]}]}
# POST /ocr     {"image_b64": ...}
#               -> {"text": "...", "lines": [{"text", "score", "box": [[x, y] x 4]}]}
#
# Every POST needs "Authorization: Bearer <EMBED_TOKEN>". The work runs one
# job per kind at a time; text queries have their own lane, so a search never
# waits behind a photo.
#
# A job that runs longer than EMBED_JOB_LIMIT_S (default 300) is stuck: the Intel
# GPU can hang inside the vision tower, and then every later image waits for a
# lock that never opens while GET / still says ok. The watchdog ends the process
# so systemd starts a fresh one.
import contextlib
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embed import otel

MAX_BODY = 48 * 1024 * 1024
JOB_LIMIT_S = float(os.environ.get("EMBED_JOB_LIMIT_S", "300"))
WATCH_EVERY_S = 5
EXIT_STUCK = 70


class Lanes:
    """One lock per kind of work. Text queries, images, faces and OCR may run
    at the same time; two jobs of one kind queue."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.locks = {name: threading.Lock() for name in ("text", "image", "faces", "ocr")}
        # When the job running now in each lane began.
        self.started = {}

    @contextlib.contextmanager
    def run(self, name):
        with self.locks[name]:
            self.started[name] = self.clock()
            try:
                yield
            finally:
                self.started.pop(name, None)

    def stuck(self, limit):
        """The lanes whose running job is older than limit seconds."""
        now = self.clock()
        return [name for name, began in list(self.started.items()) if now - began > limit]


def watch(lanes, limit, stop, every=WATCH_EVERY_S, die=lambda: os._exit(EXIT_STUCK)):
    """Ends the process when a job runs past the limit. It cannot interrupt a
    hung GPU call, so ending the process is the only way out."""
    while not stop.wait(every):
        stuck = lanes.stuck(limit)
        if stuck:
            otel.log("watchdog", {"lanes": ",".join(stuck), "limit_s": limit}, "ERROR")
            otel.flush()
            die()


def make_handler(backends, token, lanes=None):
    lanes = lanes or Lanes()
    expected = f"Bearer {token}".encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path != "/":
                return self.reply(404, {"error": "not found"})
            self.reply(200, {"ok": True, **backends.status()})

        def do_POST(self):
            started = time.monotonic()
            route = self.path
            if route not in ("/embed", "/faces", "/ocr"):
                return self.reply(404, {"error": "not found"})
            given = (self.headers.get("Authorization") or "").encode()
            if not token or not hmac.compare_digest(given, expected):
                return self.reply(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self.reply(413, {"error": "body must be 1 byte to 48 MB"})
            try:
                body = json.loads(self.rfile.read(length))
                if route == "/embed":
                    items = body["items"]
                    kind = body.get("kind", "document")
                    if kind not in ("query", "document"):
                        raise ValueError("kind must be query or document")
                    images = any("image_b64" in item for item in items)
                    with lanes.run("image" if images else "text"):
                        result = {"vectors": backends.embed(kind, items)}
                    count = len(items)
                elif route == "/faces":
                    with lanes.run("faces"):
                        result = backends.faces(body["image_b64"])
                    count = len(result["faces"])
                else:
                    with lanes.run("ocr"):
                        result = backends.ocr(body["image_b64"])
                    count = len(result["lines"])
            except (KeyError, ValueError, TypeError) as error:
                otel.log("request", {"route": route, "status": 400, "ms": ms(started)}, "WARN")
                return self.reply(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - every failure goes back as JSON
                otel.log("request", {"route": route, "status": 500, "ms": ms(started), "error": type(error).__name__}, "ERROR")
                return self.reply(500, {"error": str(error)})
            otel.log("request", {"route": route, "status": 200, "ms": ms(started), "count": count})
            self.reply(200, result)

        def reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def ms(started):
    return round((time.monotonic() - started) * 1000)


def main():
    from embed.backends import Backends

    token = os.environ.get("EMBED_TOKEN", "")
    if not token:
        raise SystemExit("EMBED_TOKEN is required")
    backends = Backends(os.environ.get("EMBED_MODELS", "/var/lib/embed"), int(os.environ.get("EMBED_THREADS", "4")))
    port = int(os.environ.get("EMBED_PORT", "8000"))
    otel.log("ready", {"port": port, "job_limit_s": JOB_LIMIT_S, **backends.status()})
    lanes = Lanes()
    threading.Thread(target=watch, args=(lanes, JOB_LIMIT_S, threading.Event()), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(backends, token, lanes)).serve_forever()


if __name__ == "__main__":
    main()
