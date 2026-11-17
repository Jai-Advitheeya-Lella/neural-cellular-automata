#!/usr/bin/env bash
# Train every growing-NCA model used in the README, two at a time on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p runs
lane() { for job in "$@"; do t=${job%%:*}; m=${job##*:};
  [ -f "runs/${t}_${m}/summary.json" ] || $PY -m nca.train --target "$t" --mode "$m" > "runs/${t}_${m}.log" 2>&1; done; }
lane lizard:regenerating lizard:growing butterfly:regenerating mushroom:regenerating &
lane lizard:persistent octopus:regenerating ladybug:regenerating &
wait
