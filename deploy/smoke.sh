#!/usr/bin/env bash
# The release bundle as deployed: copy /release out of the image, check it
# holds the package and compiles with the CT's Python. The models need the GPU
# host, so the server itself is checked by the deploy's health check.
set -euo pipefail
image=${1:-ghcr.io/zyx-tw/embed:ci}
dir=$(mktemp -d)
cid=$(docker create "$image" /none)
docker cp "$cid:/release/." "$dir/"
docker rm "$cid" >/dev/null
test -s "$dir/requirements.txt"
test -s "$dir/embed/server.py"
docker run --rm -v "$dir:/app:ro" python:3.12-slim python -c "import ast, pathlib; [ast.parse(p.read_text(), str(p)) for p in pathlib.Path('/app/embed').rglob('*.py')]"
echo "smoke passed"
