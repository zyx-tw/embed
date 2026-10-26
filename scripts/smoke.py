# Check a running embed service with real inputs: python scripts/smoke.py <url> <photo.jpg> [more.jpg ...]
# Reads EMBED_TOKEN. Prints timings (p50 / p95 over the photos) and fails on a wrong shape.
import base64
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request

url, photos = sys.argv[1].rstrip("/"), sys.argv[2:]
headers = {"content-type": "application/json", "Authorization": f"Bearer {os.environ['EMBED_TOKEN']}"}


def post(path, body):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read()), time.monotonic() - started


def summary(name, seconds):
    seconds = sorted(seconds)
    print(f"{name}: n={len(seconds)} p50={statistics.median(seconds):.2f}s p95={seconds[max(0, int(len(seconds) * 0.95) - 1)]:.2f}s")


with urllib.request.urlopen(url + "/") as response:
    print("health", json.loads(response.read()))
query, _ = post("/embed", {"kind": "query", "items": [{"text": "海邊的夕陽"}]})
assert len(query["vectors"][0]) == 768, "query vector is not 768 numbers"
times = {"image": [], "faces": [], "ocr": []}
for photo in photos:
    data = base64.b64encode(pathlib.Path(photo).read_bytes()).decode()
    result, seconds = post("/embed", {"kind": "document", "items": [{"image_b64": data}]})
    assert len(result["vectors"][0]) == 768
    times["image"].append(seconds)
    result, seconds = post("/faces", {"image_b64": data})
    assert all(len(face["vector"]) == 512 for face in result["faces"])
    times["faces"].append(seconds)
    result, seconds = post("/ocr", {"image_b64": data})
    assert isinstance(result["text"], str)
    times["ocr"].append(seconds)
for name, seconds in times.items():
    summary(name, seconds)
