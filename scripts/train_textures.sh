#!/usr/bin/env bash
# Train the three texture-synthesis NCAs used in the README (sequentially, one GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p runs
for t in coral spots cells; do
  [ -f "runs/texture_${t}/summary.json" ] && continue
  for attempt in 1 2 3 4 5; do   # the GPU is shared; retry if it is momentarily full
    $PY -m nca.train_texture --texture "$t" --iters 4000 > "runs/texture_${t}.log" 2>&1 && break
    sleep 30
  done
done
