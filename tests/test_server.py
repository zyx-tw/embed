# The HTTP layer with fake backends: auth, routing, limits and error shapes.
# The models themselves are checked on the host with scripts/smoke.py.
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from embed.server import MAX_BODY, make_handler

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
