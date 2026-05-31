#!/usr/bin/env bash
# End-to-end smoke test: encoder shape checks -> few training steps -> rollout + CEM.
set -e
cd "$(dirname "$0")/.."

export XFORMERS_DISABLED=1
PY=".venv/bin/python"
# Restrict to a single GPU for the smoke test.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=================== [1/3] encoder shape checks ==================="
$PY -m vtwm.check_encoders --config configs/default.yaml

echo "=================== [2/3] train (few steps) ====================="
$PY -m vtwm.train --config configs/default.yaml

echo "=================== [3/3] rollout + CEM demo ===================="
$PY -m vtwm.infer --config configs/default.yaml --ckpt ./runs/smoke/predictor.pt
