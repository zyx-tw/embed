#!/bin/sh
# carrel-deployd (from land) runs this from the release directory on every
# deploy and rollback, as the embed user, with /srv/apps/embed/env loaded.
# The code comes from the release; the Python environment is the shared venv
# deploy/install.sh builds as root, because installing it also fetches models.
set -eu
want=$(sha256sum requirements.txt | cut -d' ' -f1)
have=$(cat /opt/embed/requirements.sha256 2>/dev/null || true)
if [ "$want" != "$have" ]; then
  echo "requirements.txt changed: run 'sh deploy/install.sh' as root on the CT, then redeploy" >&2
  exit 1
fi
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export EMBED_MODELS=/var/lib/embed HF_HOME=/var/lib/embed
exec /opt/embed/.venv/bin/python -m embed.server
