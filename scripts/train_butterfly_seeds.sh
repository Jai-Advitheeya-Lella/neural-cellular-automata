#!/usr/bin/env bash
# The seed-0 butterfly model drifts over long rollouts (see README); train two more seeds and keep the stablest.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for s in 1 2; do
  [ -f "runs_seeds/seed${s}/butterfly_regenerating/summary.json" ] && continue
  for attempt in 1 2 3 4 5; do
    $PY -m nca.train --target butterfly --mode regenerating --seed $s --out-dir runs_seeds/seed${s} \
      > runs_seeds/butterfly_seed${s}.log 2>&1 && break
    sleep 30
  done
done
