#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIVTAC_DIR="${ROOT_DIR}/third_party/UniVTAC"
PYTHON_BIN="${PYTHON_BIN:-/home/wangshuxun/miniconda3/envs/UniVTAC/bin/python}"

# Two-GPU split to avoid the single-card Vulkan-render + CUDA-inference driver deadlock
# (the same class of hang seen in RMBench). Isaac Sim's RTX renderer + PhysX run on RENDER_GPU
# and the VT-WM policy's CEM inference runs on MODEL_GPU. The UniVTAC eval is in-process, so both
# live in one process; we expose exactly these two cards and pin each component below.
# IMPORTANT (Isaac Sim constraint): Isaac's USD/Fabric GPU scenegraph (usdrt) only supports the
# process's cuda:0, so the RENDER card must be cuda:0. We therefore default RENDER_GPU=0 and put
# the model on GPU1 — the reverse of RMBench's "model on cuda0", and not a free choice.
# Set MODEL_GPU == RENDER_GPU to fall back to single-card (reintroduces the deadlock risk).
MODEL_GPU="${MODEL_GPU:-1}"
RENDER_GPU="${RENDER_GPU:-0}"
TASK="${TASK:-grasp_classify}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-ACT/deploy}"
TOTAL_NUM="${TOTAL_NUM:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
PRINT_ONLY="${PRINT_ONLY:-1}"
WEBRTC="${WEBRTC:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_univtac_eval.sh [options]

Options:
  --webrtc                 Enable Isaac Sim WebRTC livestream.
  --model-gpu N            Physical GPU for VT-WM CEM inference. Default: env MODEL_GPU or 1.
  --render-gpu N           Physical GPU for Isaac Sim RTX render + PhysX (becomes the process
                           cuda:0 — Isaac requires it). Default: env RENDER_GPU or 0.
  --gpu N                  Deprecated single-card alias: sets both model and render gpu to N
                           (reintroduces the render+inference same-card deadlock risk).
  --task NAME              Default: grasp_classify.
  --task-config NAME       Default: demo.
  --deploy-config NAME     Default: ACT/deploy. Use VTWM/deploy for the VT-WM world model.
  --total-num N            Default: 1.
  --start-seed N           Default: 0.
  --max-seed N             Default: 0.
  --timeout-sec N          Default: 900.
  --log-to-file            Disable --print_only so UniVTAC writes its log file.
  -h, --help               Show this help.

Environment:
  WEBRTC_PUBLIC_IP         Optional public IP advertised to WebRTC clients.
  WEBRTC_SIGNAL_PORT       Default: 49100 when WEBRTC_PUBLIC_IP is set.
  WEBRTC_STREAM_PORT       Default: 47998 when WEBRTC_PUBLIC_IP is set.
  UNIVTAC_LIVESTREAM       Override livestream mode directly: 0, 1, or 2.

Examples:
  scripts/run_univtac_eval.sh
  scripts/run_univtac_eval.sh --webrtc --max-seed 0
  WEBRTC_PUBLIC_IP=1.2.3.4 scripts/run_univtac_eval.sh --webrtc
  # Closed-loop VT-WM world-model policy (CEM planning), render on gpu0 / model on gpu1:
  scripts/run_univtac_eval.sh --deploy-config VTWM/deploy --task grasp_classify --task-config demo --total-num 1 --max-seed 0 --log-to-file
  # Override the split (render on gpu2 -> its cuda:0, model on gpu3):
  scripts/run_univtac_eval.sh --deploy-config VTWM/deploy --render-gpu 2 --model-gpu 3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --webrtc) WEBRTC=1; shift ;;
    --model-gpu) MODEL_GPU="$2"; shift 2 ;;
    --render-gpu) RENDER_GPU="$2"; shift 2 ;;
    --gpu) MODEL_GPU="$2"; RENDER_GPU="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --task-config) TASK_CONFIG="$2"; shift 2 ;;
    --deploy-config) DEPLOY_CONFIG="$2"; shift 2 ;;
    --total-num) TOTAL_NUM="$2"; shift 2 ;;
    --start-seed) START_SEED="$2"; shift 2 ;;
    --max-seed) MAX_SEED="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --log-to-file) PRINT_ONLY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# GPU split, constrained by Isaac Sim. Isaac's USD/Fabric GPU scenegraph (usdrt) only supports
# the process's cuda:0 ("GPUs other than cuda:0 are not currently supported"), so the RENDER card
# MUST be the first visible CUDA device. We expose exactly the two chosen cards as
# CUDA_VISIBLE_DEVICES="RENDER_GPU,MODEL_GPU" -> render is cuda:0 (Isaac happy) and the model is
# cuda:1. Isaac's renderer+PhysX use --device cuda:0; the VT-WM policy uses VTWM_DEVICE=cuda:1.
# They land on different physical cards -> no shared-card Vulkan-render + CUDA-inference deadlock.
if [[ "${MODEL_GPU}" == "${RENDER_GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${RENDER_GPU}"
  SIM_DEVICE="cuda:0"
  export VTWM_DEVICE="cuda:0"
else
  export CUDA_VISIBLE_DEVICES="${RENDER_GPU},${MODEL_GPU}"
  SIM_DEVICE="cuda:0"
  export VTWM_DEVICE="cuda:1"
fi

cmd=(
  "${PYTHON_BIN}"
  scripts/eval_policy.py
  "${TASK}"
  "${TASK_CONFIG}"
  "${DEPLOY_CONFIG}"
  --total_num "${TOTAL_NUM}"
  --start_seed "${START_SEED}"
  --max_seed "${MAX_SEED}"
  --device "${SIM_DEVICE}"
  --headless
)

if [[ "${PRINT_ONLY}" == "1" ]]; then
  cmd+=(--print_only)
fi

if [[ "${WEBRTC}" == "1" ]]; then
  export UNIVTAC_LIVESTREAM="${UNIVTAC_LIVESTREAM:-2}"
  if [[ -n "${WEBRTC_PUBLIC_IP:-}" ]]; then
    WEBRTC_SIGNAL_PORT="${WEBRTC_SIGNAL_PORT:-49100}"
    WEBRTC_STREAM_PORT="${WEBRTC_STREAM_PORT:-47998}"
    cmd+=(
      --kit_args
      "--/exts/omni.kit.livestream.app/primaryStream/publicIp=${WEBRTC_PUBLIC_IP} --/exts/omni.kit.livestream.app/primaryStream/signalPort=${WEBRTC_SIGNAL_PORT} --/exts/omni.kit.livestream.app/primaryStream/streamPort=${WEBRTC_STREAM_PORT}"
    )
  fi
else
  export UNIVTAC_LIVESTREAM="${UNIVTAC_LIVESTREAM:-0}"
fi

export OMNI_KIT_ACCEPT_EULA=YES
# NOTE: CUDA_VISIBLE_DEVICES / VTWM_DEVICE are set above for the render/inference gpu split.
export PYTHONUNBUFFERED=1
# Make the vt-wm package (+ vendored pure-python deps) importable from the UniVTAC python,
# so the VTWM deploy policy can `import vtwm` in-process.
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/.univtac_pydeps${PYTHONPATH:+:${PYTHONPATH}}"

cd "${UNIVTAC_DIR}"
echo "Running UniVTAC eval from ${UNIVTAC_DIR}"
echo "render_gpu=${RENDER_GPU} (sim ${SIM_DEVICE}) model_gpu=${MODEL_GPU} (vtwm ${VTWM_DEVICE}) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "livestream=${UNIVTAC_LIVESTREAM} timeout=${TIMEOUT_SEC}s"
exec timeout --foreground "${TIMEOUT_SEC}s" "${cmd[@]}"
