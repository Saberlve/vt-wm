#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/univtac_lift_bottle_ada.yaml}"
CKPT="${CKPT:-runs/univtac_lift_bottle/predictor.pt}"
OUT_DIR="${OUT_DIR:-eval_out_univtac_lift_bottle}"

exec "${PYTHON_BIN}" -m vtwm.eval \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  "$@"
