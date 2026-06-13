#!/bin/bash
# Determined entrypoint: 2-GPU UniVTAC lift_bottle_neg counterexample data collection on ada-24g.
#
# Runs two parallel Isaac Sim collection workers (one per allocated GPU) via
# collect_lift_bottle_neg_batch.sh. Unlike the closed-loop eval there is no in-process model
# inference, so no render/model GPU split is needed: each worker pins itself to its own card
# (collect_data.py sets CUDA_VISIBLE_DEVICES=<gpu> -> that card becomes the process cuda:0,
# satisfying Isaac's single-cuda:0 scenegraph constraint).
set -euo pipefail

cd /home/wangshuxun/VLA/vt-wm/third_party/UniVTAC

# conda is needed because collect_lift_bottle_neg_batch.sh dispatches workers via
# `conda run -n UniVTAC ...`; make the conda launcher resolvable inside the container.
export PATH=/home/wangshuxun/miniconda3/bin:${PATH}

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=all

# Batch knobs (overridable from the Determined experiment env). Defaults match the experiment:
# 50 episodes, GPUs 0,1, start seed 101, one worker per GPU.
# RANGES (optional) overrides the even auto-split with explicit per-worker seed ranges,
# e.g. RANGES="0:122-125 1:145-150" to resume the seeds left after an interrupted run.
export TOTAL="${TOTAL:-50}"
export GPUS="${GPUS:-0,1}"
export START_SEED="${START_SEED:-101}"
export PER_GPU_WORKERS="${PER_GPU_WORKERS:-1}"
export RANGES="${RANGES:-}"
export CONDA_ENV="${CONDA_ENV:-UniVTAC}"

LOG_DIR=/home/wangshuxun/VLA/vt-wm/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ada24g_univtac_lift_bottle_neg_collect_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[entrypoint] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[entrypoint] total=${TOTAL} gpus=${GPUS} start_seed=${START_SEED} per_gpu_workers=${PER_GPU_WORKERS} ranges='${RANGES}'"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

  echo "[entrypoint] --- NVIDIA graphics/RTX lib presence ---"
  for n in libGLX_nvidia.so.0 libnvidia-glcore.so libnvidia-rtcore.so libnvoptix.so.1 libnvidia-gpucomp.so; do
    hit="$(ldconfig -p 2>/dev/null | grep -m1 "${n}" || true)"
    echo "[entrypoint]   ${n}: ${hit:-NOT FOUND}"
  done

  # Isaac Sim's RTX renderer needs a working Vulkan instance. The system ICD points at
  # libGLX_nvidia.so.0, which fails vkCreateInstance with ERROR_INCOMPATIBLE_DRIVER in this
  # headless container; synthesize an ICD pointing at the EGL driver instead (verified working
  # via vulkaninfo --summary in the eval entrypoint).
  EGL_PATH="$(ldconfig -p 2>/dev/null | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
  NV_STAGE="${NV_STAGE:-/run/determined/NAS1/public/wangshuxun/nvidia_userspace/575.57.08}"
  if [[ -z "${EGL_PATH}" && -e "${NV_STAGE}/libEGL_nvidia.so.0" ]]; then
    export LD_LIBRARY_PATH="${NV_STAGE}:${LD_LIBRARY_PATH:-}"
    EGL_PATH="${NV_STAGE}/libEGL_nvidia.so.0"
    echo "[entrypoint] graphics libs not runtime-injected; using NAS-staged driver libs at ${NV_STAGE}"
  fi

  if [[ -n "${EGL_PATH}" && -e "${EGL_PATH}" ]]; then
    SYNTH_ICD=/tmp/nvidia_icd.json
    printf '{\n  "file_format_version": "1.0.1",\n  "ICD": { "library_path": "%s", "api_version": "1.3.255" }\n}\n' \
      "${EGL_PATH}" > "${SYNTH_ICD}"
    export VK_ICD_FILENAMES="${SYNTH_ICD}"
    export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
  else
    echo "[entrypoint] WARNING: libEGL_nvidia.so.0 not found; Vulkan RTX renderer may fail"
  fi
  echo "[entrypoint] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
  if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | sed -n '1,40p' || true
  fi

  echo "[entrypoint] launching collect_lift_bottle_neg_batch.sh (no timeout)"
  bash collect_lift_bottle_neg_batch.sh "${TOTAL}" "${GPUS}" "${START_SEED}" "${PER_GPU_WORKERS}"
} 2>&1 | tee -a "${LOG_FILE}"
