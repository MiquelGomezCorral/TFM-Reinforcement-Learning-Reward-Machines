#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

nohup jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/MultiTaxi-5x5-Benchmark-dqn.ipynb \
  > logs/MultiTaxi-5x5-Benchmark-dqn.log 2>&1 < /dev/null &

echo "Notebook started in background (PID $!)."
