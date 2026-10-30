# The HTTP layer with fake backends: auth, routing, limits and error shapes.
# The models themselves are checked on the host with scripts/smoke.py.
import json
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from embed.server import MAX_BODY, Lanes, make_handler, watch

TOKEN = "test-token"


class FakeBackends:
    def status(self):
        return {"model": "fake", "vision": "torch-cpu", "faces": True, "ocr": True}

    def embed(self, kind, items):
        if any("text" not in item and "image_b64" not in item for item in items):
            raise ValueError("each item needs text or image_b64")
        return [[1.0 if kind == "query" else 0.0] for _ in items]

    def faces(self, image_b64):
        return {"width": 1, "height": 1, "faces": []}

    def ocr(self, image_b64):
        if image_b64 == "boom":
            raise RuntimeError("engine failed")
        return {"text": "hi", "lines": [{"text": "hi", "score": 1.0, "box": []}]}


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeBackends(), TOKEN))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def call(self, path, body=None, token=TOKEN, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {"content-type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health_needs_no_token(self):
        status, body = self.call("/", token=None)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["vision"], "torch-cpu")

    def test_post_needs_the_token(self):
        self.assertEqual(self.call("/embed", {"items": [{"text": "x"}]}, token=None)[0], 401)
        self.assertEqual(self.call("/embed", {"items": [{"text": "x"}]}, token="wrong")[0], 401)

    def test_embed_text_query(self):
        status, body = self.call("/embed", {"kind": "query", "items": [{"text": "海邊"}, {"text": "貓"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["vectors"], [[1.0], [1.0]])

    def test_bad_input_is_400(self):
        self.assertEqual(self.call("/embed", {"kind": "other", "items": []})[0], 400)
        self.assertEqual(self.call("/embed", {"items": [{"nothing": 1}]})[0], 400)
        self.assertEqual(self.call("/faces", {})[0], 400)
        self.assertEqual(self.call("/embed", raw=b"not json")[0], 400)

    def test_backend_failure_is_500_json(self):
        status, body = self.call("/ocr", {"image_b64": "boom"})
        self.assertEqual(status, 500)
        self.assertIn("engine failed", body["error"])

    def test_faces_and_ocr(self):
        self.assertEqual(self.call("/faces", {"image_b64": "x"}), (200, {"width": 1, "height": 1, "faces": []}))
        self.assertEqual(self.call("/ocr", {"image_b64": "x"})[1]["text"], "hi")

    def test_unknown_route_is_404(self):
        self.assertEqual(self.call("/nope", {"x": 1})[0], 404)

    def test_body_limit(self):
        request = urllib.request.Request(
            self.base + "/embed",
            data=b"{}",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Length": str(MAX_BODY + 1)},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 413)


if __name__ == "__main__":
    unittest.main()


class WatchdogTest(unittest.TestCase):
    def test_a_running_job_is_stuck_only_past_the_limit(self):
        now = [100.0]
        lanes = Lanes(clock=lambda: now[0])
        with lanes.run("image"):
            now[0] += 299
            self.assertEqual(lanes.stuck(300), [])
            now[0] += 2
            self.assertEqual(lanes.stuck(300), ["image"])
        self.assertEqual(lanes.stuck(300), [])

    def test_a_job_that_raises_leaves_no_stuck_lane(self):
        lanes = Lanes()
        with self.assertRaises(RuntimeError), lanes.run("ocr"):
            raise RuntimeError("engine failed")
        self.assertEqual(lanes.started, {})
        with lanes.run("ocr"):
            pass

    def test_the_watchdog_ends_the_process_for_a_stuck_lane(self):
        now = [0.0]
        lanes = Lanes(clock=lambda: now[0])
        stop = threading.Event()
        died = threading.Event()

        def die():
            died.set()
            stop.set()

        with lanes.run("image"):
            now[0] = 1000.0
            thread = threading.Thread(target=watch, args=(lanes, 300, stop, 0.01, die))
            thread.start()
            self.assertTrue(died.wait(2))
            thread.join(2)

    def test_the_watchdog_leaves_a_healthy_service_alone(self):
        lanes = Lanes()
        stop = threading.Event()
        died = threading.Event()
        thread = threading.Thread(target=watch, args=(lanes, 300, stop, 0.01, died.set))
        thread.start()
        with lanes.run("image"):
            self.assertFalse(died.wait(0.3))
        stop.set()
        thread.join(2)
        self.assertFalse(died.is_set())


HANG = """
import json, sys, threading, time, urllib.request
from http.server import ThreadingHTTPServer
from embed import server

class Hung:
    def status(self):
        return {}
    def embed(self, kind, items):
        time.sleep(60)

lanes = server.Lanes()
threading.Thread(target=server.watch, args=(lanes, 1, threading.Event(), 0.1), daemon=True).start()
httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(Hung(), "t", lanes))
threading.Thread(target=httpd.serve_forever, daemon=True).start()
request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/embed", data=json.dumps({"items": [{"image_b64": "x"}]}).encode(), headers={"Authorization": "Bearer t"})
urllib.request.urlopen(request, timeout=30)
"""


class HungProcessTest(unittest.TestCase):
    def test_a_hung_job_ends_the_real_process_with_code_70(self):
        done = subprocess.run([sys.executable, "-c", HANG], capture_output=True, timeout=30, check=False)
        self.assertEqual(done.returncode, 70)
        self.assertIn(b'"event": "watchdog"', done.stderr)
