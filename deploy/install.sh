#!/bin/sh
# Build or update embed's Python environment on a Debian host, as root, from a
# checkout in the current folder: sh deploy/install.sh. Needs uv, and Intel's
# compute runtime for the GPU (see README). The code itself deploys through
# land on every merge; run this once, and again whenever requirements.txt
# changes (deploy/carrel-start.sh refuses to start until it matches).
set -e
# The service user reads everything here, whatever umask the caller had.
umask 022
# Keep uv's Python under /opt so the service user can read it (not in /root).
export UV_PYTHON_INSTALL_DIR=/opt/embed/python
id embed >/dev/null 2>&1 || useradd --system --home /var/lib/embed --shell /usr/sbin/nologin embed
install -d -o embed -g embed /var/lib/embed
install -d -m 755 /opt/embed
[ -d /opt/embed/.venv ] || uv venv -q -p 3.12 /opt/embed/.venv
VIRTUAL_ENV=/opt/embed/.venv uv pip install -q --index-strategy unsafe-best-match -r requirements.txt
# RapidOCR fetches its models into its own package folder on first use, which
# the service user cannot write: fetch them now, as root.
/opt/embed/.venv/bin/python -c "from rapidocr import RapidOCR; RapidOCR()" >/dev/null
sha256sum requirements.txt | cut -d' ' -f1 > /opt/embed/requirements.sha256
echo "venv ready for requirements $(cut -c1-12 /opt/embed/requirements.sha256)"
