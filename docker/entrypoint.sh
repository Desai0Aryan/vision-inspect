#!/bin/sh
# Container entrypoint: make sure the demo clip exists, then hand over to the server.
# The server MUST run a single worker: the job queue, trackers and live sessions live in process memory.
set -e
if [ ! -f "${VI_DEMO_VIDEO}" ]; then
  echo "[entrypoint] demo clip missing, generating ${VI_DEMO_VIDEO}"
  mkdir -p "$(dirname "${VI_DEMO_VIDEO}")"
  python scripts/make_demo_video.py --out "${VI_DEMO_VIDEO}"
fi
mkdir -p "${VI_JOBS_DIR}"
exec "$@"
