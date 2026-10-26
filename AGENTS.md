# AGENTS.md

## Layout

- `embed/server.py`: the HTTP layer. Routes, the bearer token, body limits and the work lanes.
- `embed/backends.py`: the models. EmbeddingGemma 2 in sentence-transformers, its vision tower on the Intel GPU through OpenVINO, faces, and OCR.
- `embed/faces.py`: SCRFD and ArcFace from InsightFace's buffalo_l, run with onnxruntime. Copied from NYCU-WinLab/gallery; keep the two alike.
- `embed/otel.py`: OTLP/HTTP JSON logs for sensorium.
- `deploy/`: the systemd unit, the env file shape and the install script.

## Rules

- The service keeps nothing. No database, no files from requests, no request content in logs.
- Log with `otel.log()`: routes, counts, timings and error types. Never text, images, vectors or tokens.
- Every POST checks the token with `hmac.compare_digest`. `GET /` is the only open route.
- A text query never waits behind an image. Keep text and image work on separate lanes.
- A change to the model, its revision or `MAX_SIDE` changes every vector: the callers must re-embed. Say so in the PR.
- The GPU path must match the CPU path. Check cosine against `EMBED_VISION=cpu` before changing `vision_on_gpu`.
- Python is the exception here: only sentence-transformers runs EmbeddingGemma 2's vision encoder.

## Verify

```sh
python3 -m unittest discover -s tests -t .
EMBED_TOKEN=... python scripts/smoke.py http://10.10.12.210:8000 photo1.jpg photo2.jpg
```
