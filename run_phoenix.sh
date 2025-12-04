#!/usr/bin/env bash
set -euo pipefail

# ========== Basics ==========
ENV_NAME="${ENV_NAME:-MoLMamba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
ulimit -n 65535 || true

# ========== Conda ==========
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${ENV_NAME}" || echo "[WARN] conda activate ${ENV_NAME} failed, continuing…"
fi

# ========== HF token ==========
# Put your real token here (no quotes):
export HF_TOKEN=${HF_TOKEN:-hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX}
export HUGGINGFACE_TOKEN="${HF_TOKEN}"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# ========== Environment summary ==========
echo "[INFO] Python: $(which python) | $(python -V)"
python - <<'PY'
import torch
print(f"Torch {torch.__version__} | CUDA={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
PY

# ========== Dataset config ==========
BASE_URI=${BASE_URI:-"/dccstor/kcsys/edusoares/FM4PDE/the_well_local"}  # local cache root
REMOTE_URI=${REMOTE_URI:-"hf://datasets/polymathic-ai/"}                 # org source
NEW_DS=${NEW_DS:-"viscoelastic_instability"}                             # dataset name under org

# ========== Train schedule ==========
EPOCHS=${EPOCHS:-20}
STEPS=${STEPS:-800}
BATCH=${BATCH:-4}
WORKERS=${WORKERS:-0}
HISTORY=${HISTORY:-1}

# ========== Model defaults (fallback if ckpt lacks cfg) ==========
DIM=${DIM:-256}
LATENT=${LATENT:-8}
PATCH=${PATCH:-16}
MODES=${MODES:-12}

# ========== Finetune knobs ==========
FINETUNE_MODE=${FINETUNE_MODE:-adapters}   # adapters | adapters_decoder | full
UNFREEZE_LAST_N=${UNFREEZE_LAST_N:-2}
LR_ADAPTERS=${LR_ADAPTERS:-3e-4}
LR_DECODER=${LR_DECODER:-3e-4}
LR_BACKBONE=${LR_BACKBONE:-1e-4}
WD=${WD:-1e-2}

LOSS_VRMSE_W=${LOSS_VRMSE_W:-1.0}
LOSS_SPEC_W=${LOSS_SPEC_W:-0.0}
LOSS_L1_W=${LOSS_L1_W:-0.0}

# ========== Numerics ==========
NO_AMP=${NO_AMP:-1}
export PHOENIX_CLIP_IN=${PHOENIX_CLIP_IN:-5}
export PHOENIX_CLIP_OUT=${PHOENIX_CLIP_OUT:-5}
unset PYTORCH_CUDA_ALLOC_CONF

# ========== NCCL hardening ==========
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-7200}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ========== GPU discovery ==========
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi
[[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=0
MASTER_PORT=$(( (${RANDOM} + ${LSB_JOBID:-0}) % 400 + 29600 ))
echo "[INFO] Using master port: ${MASTER_PORT} | NUM_GPUS=${NUM_GPUS}"

# ========== Auto-download if local HDF5s missing (fixed) ==========
REQ_DIR="${BASE_URI%/}/${NEW_DS}/data/train"
if ! compgen -G "${REQ_DIR}/*.hdf5" > /dev/null; then
  echo "[WARN] No local HDF5 found — will download to ${BASE_URI%/}/${NEW_DS}"
  python - <<'PY'
import os, sys
from pathlib import Path
from glob import glob
from huggingface_hub import snapshot_download

remote = (os.environ.get("REMOTE_URI","") or "").strip()
ds     = (os.environ.get("NEW_DS","") or "").strip()
base   = Path(os.environ.get("BASE_URI","./the_well_local")).expanduser()
tok    = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or ""

if not ds:
    print("[ERROR] NEW_DS is empty; set NEW_DS to the dataset name.", file=sys.stderr)
    sys.exit(2)

# Build a valid repo_id robustly
repo_id = None
if remote.startswith("hf://datasets/"):
    org = remote.split("hf://datasets/",1)[1].strip("/")
    if not org:
        print("[ERROR] REMOTE_URI malformed (empty org after hf://datasets/).", file=sys.stderr); sys.exit(2)
    repo_id = f"{org}/{ds}"
elif "/" in remote and not remote.startswith("http"):
    parts = [p for p in remote.split("/") if p]
    if len(parts) == 1:
        repo_id = f"{parts[0]}/{ds}"
    else:
        repo_id = "/".join(parts[:2])      # treat remote as fully-specified org/dataset
else:
    repo_id = f"polymathic-ai/{ds}"

if not repo_id or repo_id.count("/") != 1 or any(x.strip()=="" for x in repo_id.split("/")):
    print(f"[ERROR] Computed invalid repo_id='{repo_id}'. "
          f"Set REMOTE_URI to 'hf://datasets/<org>/' or '<org>' or '<org>/<dataset>'.", file=sys.stderr)
    sys.exit(2)

dst = base / ds
dst.mkdir(parents=True, exist_ok=True)
print(f"[INFO] snapshot_download(repo_id='{repo_id}') -> {dst}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(dst),
    allow_patterns=["data/**"],
    max_workers=8,
    token=tok
)

if not glob(str(dst / "data/train/*.hdf5")):
    print("[ERROR] No HDF5 files found after download in 'data/train'.", file=sys.stderr)
    sys.exit(2)
print("[INFO] Precache OK.")
PY
fi

# ========== Checkpoint / outputs ==========
CKPT=${CKPT:-"./checkpoints_foundation_model_ablation_full/foundation_epoch48.pt"}
SAVE_DIR=${SAVE_DIR:-"./ckpt_${NEW_DS}_ft"}
mkdir -p "${SAVE_DIR}"
export PHX_JSON_DEFAULT_STR=1

# ========== Common args ==========
COMMON=(
  -m phoenix.finetune
  --base "${BASE_URI}"
  --datasets "${NEW_DS}"
  --epochs "${EPOCHS}"
  --steps_per_epoch "${STEPS}"
  --batch "${BATCH}"
  --workers "${WORKERS}"
  --history "${HISTORY}"
  --dim "${DIM}" --latent "${LATENT}" --patch "${PATCH}" --modes "${MODES}"
  --finetune_mode "${FINETUNE_MODE}" --unfreeze_last_n "${UNFREEZE_LAST_N}"
  --lr_adapters "${LR_ADAPTERS}" --lr_decoder "${LR_DECODER}"
  --lr_backbone "${LR_BACKBONE}" --weight_decay "${WD}"
  --loss_vrmse_w "${LOSS_VRMSE_W}" --loss_spec_w "${LOSS_SPEC_W}" --loss_l1_w "${LOSS_L1_W}"
  --save_dir "${SAVE_DIR}" --checkpoint "${CKPT}"
)
[[ "${NO_AMP}" == "1" ]] && COMMON+=( --no_amp )

# ========== Launch ==========
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[INFO] Launching DDP with ${NUM_GPUS} GPUs"
  exec torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${COMMON[@]}"
else
  echo "[INFO] Launching single process"
  exec python "${COMMON[@]}"
fi
