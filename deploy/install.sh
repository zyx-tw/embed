#!/bin/sh
# Install or update embed on a Debian host, as root, from a checkout in the
# current folder: sh deploy/install.sh. Needs uv, and Intel's compute runtime
# for the GPU (see README). Keeps /etc/embed/env and /var/lib/embed.
set -e
# The service user reads everything here, whatever umask the caller had.
umask 022
# Keep uv's Python under /opt so the service user can read it (not in /root).
export UV_PYTHON_INSTALL_DIR=/opt/embed/python
id embed >/dev/null 2>&1 || useradd --system --home /var/lib/embed --shell /usr/sbin/nologin embed
install -d -o embed -g embed /var/lib/embed
install -d -m 755 /opt/embed
rm -rf /opt/embed/embed
cp -r embed requirements.txt /opt/embed/
[ -d /opt/embed/.venv ] || uv venv -q -p 3.12 /opt/embed/.venv
VIRTUAL_ENV=/opt/embed/.venv uv pip install -q --index-strategy unsafe-best-match -r /opt/embed/requirements.txt
# RapidOCR fetches its models into its own package folder on first use, which
# the service user cannot write: fetch them now, as root.
/opt/embed/.venv/bin/python -c "from rapidocr import RapidOCR; RapidOCR()" >/dev/null
install -m 644 deploy/embed.service /etc/systemd/system/embed.service
[ -f /etc/embed/env ] || { echo "write /etc/embed/env first (deploy/env.example)"; exit 1; }
systemctl daemon-reload
systemctl enable embed >/dev/null 2>&1
systemctl restart embed
echo "embed restarted; the first start downloads models and converts the vision tower (a few minutes)"
