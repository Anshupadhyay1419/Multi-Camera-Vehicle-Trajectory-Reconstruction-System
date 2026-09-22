#!/usr/bin/env bash
# Kept so that anything (a bookmark, a note, an old README, muscle memory)
# pointing at docker/start.sh still works.
#
# Everything it used to do is now in ./run, which does the same setup and
# rather more besides -- it also evicts containers left behind by a copy of
# this project at another path, rebuilds only when the dependencies actually
# changed, and waits until the dashboard really answers before telling you
# it started. See ./run --help.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "docker/start.sh now delegates to ./run up  (try ./run --help)"
exec ./run up "$@"
