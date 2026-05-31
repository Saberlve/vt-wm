#!/bin/bash
# Determined entrypoint: 6-GPU single-node DDP training of VT-WM on ManiFeel (ada-24g).
# Uses the project's uv venv (self-contained torch); tees all output to a local log.
set -euo pipefail

cd /home/wangshuxun/VLA/vt-wm
export PYTHONPATH=.
export XFORMERS_DISABLED=1
# wandb server (api.bandw.top) is reachable directly from cluster nodes -> online auto-sync.
# Bounded init so a node without egress falls back gracefully instead of hanging.
export WANDB_INIT_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
# RTX 4090 lacks P2P/IB; keep NCCL on shared memory for a stable single-node ring.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NPROC="${VTWM_NPROC:-6}"
PORT="${VTWM_MASTER_PORT:-29501}"

LOG_DIR=/home/wangshuxun/VLA/vt-wm/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ada24g_$(date +%Y%m%d_%H%M%S).log"
echo "[entrypoint] host=$(hostname) nproc=${NPROC} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "${LOG_FILE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | tee -a "${LOG_FILE}" || true

.venv/bin/torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${PORT}" \
  -m vtwm.train --config configs/manifeel_ada.yaml --resume auto 2>&1 | tee -a "${LOG_FILE}"
