# The release bundle: the Python package and its pinned requirements under
# /release, nothing else. The embed CT has no Docker; ship copies /release into
# the land release and deploy/carrel-start.sh runs it with the venv that
# deploy/install.sh built.
FROM scratch
COPY embed /release/embed
COPY requirements.txt /release/requirements.txt
