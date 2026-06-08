#!/usr/bin/env bash
# Open-loop CEM evaluation of a trained VT-WM against the UniVTAC lift_bottle dataset.
# For each demonstration window it runs the train/deploy cold-start CEM planner and dumps
# the GT vs CEM-sampled actions, plus current / goal / imagined images (see vtwm/openloop_eval.py).
set -euo pipefail

cd "$(dirname "$0")/.."

export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/univtac_lift_bottle_ada.yaml}"
CKPT="${CKPT:-runs/univtac_lift_bottle/predictor.pt}"
OUT_DIR="${OUT_DIR:-eval_out_openloop_univtac_lift_bottle}"
SPLIT="${SPLIT:-train}"
NUM_WINDOWS="${NUM_WINDOWS:-6}"

exec "${PYTHON_BIN}" -m vtwm.openloop_eval \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --num_windows "${NUM_WINDOWS}" \
  "$@"
