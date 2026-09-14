#!/usr/bin/env bash
set -euo pipefail

echo "=== Autonomous Vehicle Parking (PPO-1) – Headline Reproduction ==="

# Install if needed
if ! python -c "import highway_env, stable_baselines3" 2>/dev/null; then
  pip install -r requirements.txt
fi

# Short training for CI / quick check (full training uses 300k steps)
TIMESTEPS=${TIMESTEPS:-50000}
SEEDS=${SEEDS:-"42 123 456"}

echo "Training with ${TIMESTEPS} timesteps and seeds: ${SEEDS}"
python -m src.train \
  --config configs/ppo_parking.yaml \
  --seeds ${SEEDS} \
  --timesteps ${TIMESTEPS} \
  --n-envs 2

# Discover the latest run directory
LATEST_LOG=$(ls -td logs/run_* | head -1)
LATEST_MODEL=$(ls -td models/run_* | head -1)

echo "Evaluating..."
python -m src.evaluate \
  --models-dir "${LATEST_MODEL}" \
  --n-episodes 10 \
  --out results/evaluation.json

echo "Plotting..."
python -m src.plotting \
  --log-dir "${LATEST_LOG}" \
  --eval-json results/evaluation.json \
  --out-dir figures

echo "Done. Results in results/evaluation.json and figures/"
