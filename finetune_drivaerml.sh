#!/usr/bin/env bash
# File: finetune_drivaerml_run11_slices.sh
set -euo pipefail

# ========== Config ==========
HF_OWNER="neashton"
HF_PREFIX="drivaerml"
REPO_ID="${HF_OWNER}/${HF_PREFIX}"

RUN_ID=11
RUN_DIR="run_${RUN_ID}"
LOCAL_ROOT="${LOCAL_ROOT:-./drivaer_data}"   # where to store the dataset locally
ENV_NAME="${ENV_NAME:-MoLMamba}"             # conda env with your phoenix deps

# VTK loader/training params
VTK_FIELDS="${VTK_FIELDS:-U,p}"
VTK_PLANE="${VTK_PLANE:-y}"
VTK_SLICES="${VTK_SLICES:-8,20,32,44,56}"
VTK_DIMS="${VTK_DIMS:-256,256,64}"           # H,W,D for resampling (kept for consistency)

EPOCHS="${EPOCHS:-10}"
STEPS="${STEPS:-400}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-0}"
SAVE_DIR="${SAVE_DIR:-./ckpt_drivaerml_run11_slices}"

# ========== Env ==========
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${ENV_NAME}" || echo "[WARN] conda activate ${ENV_NAME} failed; continuing…"
fi

# Recommend the HF CLI for directory includes
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[INFO] Installing huggingface_hub (user site)"
  python - <<'PY'
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
PY
fi

# ========== Download only the slices ==========
VTK_ROOT="${LOCAL_ROOT}/${RUN_DIR}"
mkdir -p "${VTK_ROOT}"

echo "[INFO] Downloading slices for ${RUN_DIR} → ${VTK_ROOT}"
huggingface-cli download "${REPO_ID}" \
  --repo-type dataset \
  --local-dir "${VTK_ROOT}" \
  --local-dir-use-symlinks False \
  --include "${RUN_DIR}/slices/*" \
  >/dev/null

if ! compgen -G "${VTK_ROOT}/slices/*.vtp" > /dev/null; then
  echo "[ERROR] No VTP slices found in ${VTK_ROOT}/slices"
  echo "Check the dataset folder: https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/tree/main/${RUN_DIR}/slices"
  exit 2
fi

echo "[INFO] Found $(ls -1 ${VTK_ROOT}/slices/*.vtp | wc -l) slice files."

# ========== Train (uses VTP slices) ==========
PYMOD="phoenix.finetune_vtk"
COMMON_ARGS=(
  -m "${PYMOD}"
  --vtk_root "${VTK_ROOT}"
  --vtk_glob "slices/*.vtp"
  --vtk_fields "${VTK_FIELDS}"
  --vtk_plane "${VTK_PLANE}"
  --vtk_slices "${VTK_SLICES}"
  --vtk_dims "${VTK_DIMS}"
  --epochs "${EPOCHS}"
  --steps_per_epoch "${STEPS}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --save_dir "${SAVE_DIR}"
  --no_amp
)

# GPU count → torchrun if >1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NGPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
fi
NGPUS=${NGPUS:-0}

if (( NGPUS > 1 )); then
  MASTER_PORT=$(( (${RANDOM} + ${LSB_JOBID:-0}) % 400 + 29600 ))
  export MASTER_ADDR=127.0.0.1 MASTER_PORT
  echo "[INFO] Launching DDP with ${NGPUS} GPUs | MASTER_PORT=${MASTER_PORT}"
  python -m torch.distributed.run --nproc_per_node="${NGPUS}" "${COMMON_ARGS[@]}"
else
  echo "[INFO] Launching single-process training"
  python "${COMMON_ARGS[@]}"
fi

# ========== Visualize predictions vs GT ==========
CKPT_PATH="${SAVE_DIR}/finetune_epoch$(printf "%03d" "${EPOCHS}").pt"
if [[ ! -f "${CKPT_PATH}" ]]; then
  # Fall back to the latest matching checkpoint if exact epoch file not present
  CKPT_PATH="$(ls -1t "${SAVE_DIR}"/finetune_epoch*.pt | head -n1 || true)"
fi

if [[ -n "${CKPT_PATH}" && -f "${CKPT_PATH}" ]]; then
  echo "[INFO] Visualizing with checkpoint: ${CKPT_PATH}"
  python tools/visualize_vtk_preds.py \
    --ckpt "${CKPT_PATH}" \
    --vtk_root "${VTK_ROOT}" \
    --vtk_glob "slices/*.vtp" \
    --vtk_fields "${VTK_FIELDS}" \
    --vtk_plane "${VTK_PLANE}" \
    --vtk_slices "${VTK_SLICES}" \
    --vtk_dims "${VTK_DIMS}" \
    --outdir ./viz_samples_run11_slices \
    --vis_field p
  echo "✅ Visualization written to ./viz_samples_run11_slices"
else
  echo "[WARN] No checkpoint found in ${SAVE_DIR}; skipping visualization."
fi
