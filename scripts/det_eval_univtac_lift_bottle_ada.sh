#!/bin/bash
# Determined entrypoint: 2-GPU UniVTAC lift_bottle closed-loop VT-WM evaluation on ada-24g.
set -euo pipefail

cd /home/wangshuxun/VLA/vt-wm

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/wangshuxun/VLA/vt-wm:/home/wangshuxun/VLA/vt-wm/.univtac_pydeps${PYTHONPATH:+:${PYTHONPATH}}
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export XFORMERS_DISABLED=1
export TOKENIZERS_PARALLELISM=false

# AppLauncher defaults to RaytracedLighting; eval_policy.py also forces enable_cameras=True.
export UNIVTAC_LIVESTREAM=0
export MODEL_GPU="${MODEL_GPU:-1}"
export RENDER_GPU="${RENDER_GPU:-0}"
export TASK_CONFIG="${TASK_CONFIG:-demo}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

LOG_DIR=/home/wangshuxun/VLA/vt-wm/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ada24g_univtac_lift_bottle_eval_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[entrypoint] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[entrypoint] render_gpu=${RENDER_GPU} model_gpu=${MODEL_GPU} task_config=${TASK_CONFIG} timeout=${TIMEOUT_SEC}s"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
  ls -lh runs/univtac_lift_bottle/predictor.pt

  echo "[entrypoint] --- NVIDIA graphics/RTX lib presence ---"
  for n in libGLX_nvidia.so.0 libnvidia-glcore.so libnvidia-rtcore.so libnvoptix.so.1 libnvidia-gpucomp.so; do
    hit="$(ldconfig -p 2>/dev/null | grep -m1 "${n}" || true)"
    echo "[entrypoint]   ${n}: ${hit:-NOT FOUND}"
  done

  EGL_PATH="$(ldconfig -p 2>/dev/null | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
  NV_STAGE="${NV_STAGE:-/run/determined/NAS1/public/wangshuxun/nvidia_userspace/575.57.08}"
  if [[ -z "${EGL_PATH}" && -e "${NV_STAGE}/libEGL_nvidia.so.0" ]]; then
    export LD_LIBRARY_PATH="${NV_STAGE}:${LD_LIBRARY_PATH:-}"
    EGL_PATH="${NV_STAGE}/libEGL_nvidia.so.0"
    echo "[entrypoint] graphics libs not runtime-injected; using NAS-staged driver libs at ${NV_STAGE}"
  fi

  if [[ -n "${EGL_PATH}" && -e "${EGL_PATH}" ]]; then
    # Always synthesize an ICD pointing at the EGL driver. The system
    # /usr/share/vulkan/icd.d/nvidia_icd.json points at libGLX_nvidia.so.0, which fails
    # vkCreateInstance with ERROR_INCOMPATIBLE_DRIVER in this headless container; the EGL lib
    # creates a working Vulkan instance (verified via vulkaninfo --summary).
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
    vulkaninfo --summary 2>&1 | sed -n '1,80p' || true
  else
    echo "[entrypoint] vulkaninfo not installed"
  fi

  scripts/run_univtac_eval.sh \
    --deploy-config VTWM/deploy_lift_bottle \
    --task lift_bottle \
    --task-config "${TASK_CONFIG}" \
    --total-num "${TOTAL_NUM:-1}" \
    --start-seed "${START_SEED:-0}" \
    --max-seed "${MAX_SEED:-0}" \
    --render-gpu "${RENDER_GPU}" \
    --model-gpu "${MODEL_GPU}" \
    --timeout-sec "${TIMEOUT_SEC}" \
    --log-to-file
} 2>&1 | tee -a "${LOG_FILE}"
