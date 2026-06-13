#!/usr/bin/env bash
set -euo pipefail

# Batch collection wrapper for lift_bottle_neg.
#
# Defaults collect 50 save_all negative episodes across GPUs 0,1 from seed 101,
# split by disjoint seed ranges.
# Override via env vars:
#   TOTAL=200 GPUS=0,1,2,3 START_SEED=1 PER_GPU_WORKERS=1 ./collect_lift_bottle_neg_batch.sh
#
# Positional args are also accepted:
#   ./collect_lift_bottle_neg_batch.sh [TOTAL] [GPUS] [START_SEED] [PER_GPU_WORKERS]
#
# RESUME / explicit ranges: set RANGES to override the even auto-split with exact
# per-worker seed ranges (one worker per token). Format: space-separated
# "gpu:start-max" tokens; episode budget per worker = max-start+1. Example to fill
# in the seeds left after an interrupted run:
#   RANGES="0:122-125 1:145-150" ./collect_lift_bottle_neg_batch.sh
# When RANGES is set, TOTAL/START_SEED/PER_GPU_WORKERS are ignored.

TASK_NAME="${TASK_NAME:-lift_bottle_neg}"
CONFIG_NAME="${CONFIG_NAME:-neg}"
TOTAL="${1:-${TOTAL:-50}}"
GPUS="${2:-${GPUS:-0,1}}"
START_SEED="${3:-${START_SEED:-101}}"
PER_GPU_WORKERS="${4:-${PER_GPU_WORKERS:-1}}"
RANGES="${RANGES:-}"
CONDA_ENV="${CONDA_ENV:-UniVTAC}"
LOG_ROOT="${LOG_ROOT:-batch_logs/${TASK_NAME}_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_ROOT}"

echo "[batch] task=${TASK_NAME} config=${CONFIG_NAME}"
echo "[batch] output_dir=./data/${TASK_NAME}/${CONFIG_NAME}"

# Build the worker list as parallel arrays (gpu / start / max / count).
W_GPU=(); W_START=(); W_MAX=(); W_COUNT=()

if [[ -n "${RANGES}" ]]; then
  # Explicit per-worker ranges: "gpu:start-max gpu:start-max ..."
  echo "[batch] explicit RANGES=${RANGES}"
  for tok in ${RANGES}; do
    gpu="${tok%%:*}"; span="${tok#*:}"
    start="${span%%-*}"; max="${span##*-}"
    if [[ -z "${gpu}" || -z "${start}" || -z "${max}" || "${max}" -lt "${start}" ]]; then
      echo "bad RANGES token: ${tok} (want gpu:start-max with max>=start)" >&2
      exit 2
    fi
    W_GPU+=("${gpu}"); W_START+=("${start}"); W_MAX+=("${max}")
    W_COUNT+=("$(( max - start + 1 ))")
  done
else
  # Even auto-split of TOTAL across GPUS x PER_GPU_WORKERS contiguous seed ranges.
  if [[ "${TOTAL}" -le 0 ]]; then
    echo "TOTAL must be positive, got ${TOTAL}" >&2
    exit 2
  fi
  if [[ "${PER_GPU_WORKERS}" -le 0 ]]; then
    echo "PER_GPU_WORKERS must be positive, got ${PER_GPU_WORKERS}" >&2
    exit 2
  fi
  IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
  WORKERS=$(( ${#GPU_LIST[@]} * PER_GPU_WORKERS ))
  if [[ "${WORKERS}" -le 0 ]]; then
    echo "No GPUs specified in GPUS=${GPUS}" >&2
    exit 2
  fi
  echo "[batch] total=${TOTAL} start_seed=${START_SEED} gpus=${GPUS} per_gpu_workers=${PER_GPU_WORKERS} workers=${WORKERS}"
  base=$(( TOTAL / WORKERS ))
  rem=$(( TOTAL % WORKERS ))
  next_seed="${START_SEED}"
  worker_idx=0
  for gpu in "${GPU_LIST[@]}"; do
    for local_idx in $(seq 1 "${PER_GPU_WORKERS}"); do
      count="${base}"
      if [[ "${worker_idx}" -lt "${rem}" ]]; then
        count=$(( count + 1 ))
      fi
      worker_idx=$(( worker_idx + 1 ))
      if [[ "${count}" -le 0 ]]; then
        continue
      fi
      start="${next_seed}"
      max=$(( start + count - 1 ))
      W_GPU+=("${gpu}"); W_START+=("${start}"); W_MAX+=("${max}"); W_COUNT+=("${count}")
      next_seed=$(( max + 1 ))
    done
  done
fi

echo "[batch] log_root=${LOG_ROOT}"

pids=()
for i in "${!W_GPU[@]}"; do
  gpu="${W_GPU[$i]}"; start="${W_START[$i]}"; max="${W_MAX[$i]}"; count="${W_COUNT[$i]}"
  log_file="${LOG_ROOT}/w${i}_gpu${gpu}_seed${start}-${max}.log"
  echo "[batch] worker=${i} gpu=${gpu} seeds=${start}-${max} episodes=${count} log=${log_file}"
  conda run -n "${CONDA_ENV}" bash collect_data.sh \
    "${TASK_NAME}" "${CONFIG_NAME}" "${gpu}" "${start}" "${max}" "${count}" \
    > "${log_file}" 2>&1 &
  pids+=("$!")
done

cleanup() {
  echo "[batch] stopping workers: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[batch] one or more workers failed; check ${LOG_ROOT}" >&2
  exit 1
fi

echo "[batch] complete. Logs: ${LOG_ROOT}"
