#!/usr/bin/env bash
# One-command start for a fresh clone:  ./docker/start.sh
#
# Needs: Docker + Compose v2, an NVIDIA GPU (driver >= 580) and the NVIDIA
# Container Toolkit. Safe to re-run; later runs are fast (cached build).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: cannot talk to Docker (permission denied or Docker not running)." >&2
    echo "  Not in the docker group yet:  sudo usermod -aG docker \$USER" >&2
    echo "  Just added? This session doesn't know yet. Run  newgrp docker  or log out and back in" >&2
    echo "  (VS Code Remote: 'Remote-SSH: Kill VS Code Server on Host', then reconnect)." >&2
    exit 1
fi

# Git LFS stores the trained plate detector. Without `git lfs pull` the file
# is a ~130-byte text pointer and the pipeline fails on its first frame.
if head -c 200 models/plate_detector/best.pt | grep -q "git-lfs"; then
    echo "ERROR: model weights are Git LFS pointers. Run:  git lfs install && git lfs pull" >&2
    exit 1
fi

# Git-ignored, so absent after a clone. If Docker creates them for the bind
# mounts they belong to root and the container (running as you) cannot write.
mkdir -p data logs

# Run the containers as the current user so everything they write stays yours.
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" > .env

docker compose up -d --build

ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "Started. First start can take a minute while models load."
echo "  Dashboard : http://${ip:-localhost}:8502"
echo "  API docs  : http://${ip:-localhost}:8000/docs"
echo "  Logs      : docker compose logs -f alpr"
echo "  Stop      : docker compose down"
