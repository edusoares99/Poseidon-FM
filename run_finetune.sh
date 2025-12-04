#!/usr/bin/env bash
# File: run_finetune.sh
# Purpose: Robust Phoenix finetune launcher for FM4PDE (local ou LSF via bsub)
set -euo pipefail

############################
# ====== BASIC SETUP ======
############################
ENV_NAME="${ENV_NAME:-MoLMamba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
ulimit -n 65535 || true

HOST="$(hostname 2>/dev/null || echo unknown)"
DATE="$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo unknown)"
echo "[INFO] Host: ${HOST} | Started: ${DATE}"
[[ -n "${LSB_JOBID:-}" ]] && echo "[INFO] LSF JobID: ${LSB_JOBID} | Queue: ${LSB_QUEUE:-?} | CPUs: ${LSB_DJOB_NUMPROC:-?}"

#################################
# ====== CONDA ENV (Bash) ======
#################################
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${ENV_NAME}" || echo "[WARN] conda activate ${ENV_NAME} failed, continuing…"
fi

#########################################
# ====== HUGGING FACE AUTH/FLAGS =======
#########################################
export HF_TOKEN="${HF_TOKEN:-}"                             # opcional
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"  # herdado
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

#################################
# ====== ENV SUMMARY (Py) ======
#################################
echo "[INFO] Python: $(which python) | $(python -V)"
python - <<'PY' || { echo "[ERROR] Python env check failed"; exit 2; }
import sys, torch
print(f"Torch {torch.__version__} | CUDA={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
PY

#####################################
# ====== DATASET CONFIG/INPUTS =====
#####################################
# Defina REMOTE_URI e NEW_DS conforme o dataset alvo
BASE_URI=${BASE_URI:-"/dccstor/kcsys/edusoares/FM4PDE/the_well_local"}
REMOTE_URI=${REMOTE_URI:-"hf://datasets/polymathic-ai/"}      # pode ser: hf://datasets/<org>/  ou  <org>  ou  <org>/<dataset>
NEW_DS=${NEW_DS:-"acoustic_scattering_discontinuous"}                  # e.g. viscoelastic_instability

# Garanta que o Python também veja essas variáveis
export BASE_URI REMOTE_URI NEW_DS

# Debug rápido do que o Python vai herdar
echo "[DEBUG] NEW_DS='${NEW_DS}' | REMOTE_URI='${REMOTE_URI}' | BASE_URI='${BASE_URI}'"
python - <<'PY'
import os
print("[DEBUG] PY NEW_DS=", os.environ.get("NEW_DS"))
print("[DEBUG] PY REMOTE_URI=", os.environ.get("REMOTE_URI"))
print("[DEBUG] PY BASE_URI=", os.environ.get("BASE_URI"))
PY

########################################
# ====== TRAINING SCHEDULE/NUMERICS ===
########################################
EPOCHS=${EPOCHS:-20}
STEPS=${STEPS:-800}
BATCH=${BATCH:-4}
WORKERS=${WORKERS:-0}
HISTORY=${HISTORY:-1}

# Model fallback se o ckpt não tiver cfg
DIM=${DIM:-256}
LATENT=${LATENT:-8}
PATCH=${PATCH:-16}
MODES=${MODES:-12}

# Finetune knobs
FINETUNE_MODE=${FINETUNE_MODE:-adapters}   # adapters | adapters_decoder | full
UNFREEZE_LAST_N=${UNFREEZE_LAST_N:-2}
LR_ADAPTERS=${LR_ADAPTERS:-3e-4}
LR_DECODER=${LR_DECODER:-3e-4}
LR_BACKBONE=${LR_BACKBONE:-1e-4}
WD=${WD:-1e-2}

# Loss weights
LOSS_VRMSE_W=${LOSS_VRMSE_W:-1.0}
LOSS_SPEC_W=${LOSS_SPEC_W:-0.0}
LOSS_L1_W=${LOSS_L1_W:-0.0}

# AMP / clipping
NO_AMP=${NO_AMP:-1}
export PHOENIX_CLIP_IN=${PHOENIX_CLIP_IN:-5}
export PHOENIX_CLIP_OUT=${PHOENIX_CLIP_OUT:-5}
unset PYTORCH_CUDA_ALLOC_CONF

##################################
# ====== NCCL/COMM SETTINGS =====
##################################
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-7200}

# Se seu cluster tiver IB bom, troque ambos para 0; caso contrário mantenha 1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

#########################################
# ====== GPU DISCOVERY / LSF HELPERS ===
#########################################
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi
[[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=0

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(( (${RANDOM} + ${LSB_JOBID:-0}) % 400 + 29600 ))}
export MASTER_ADDR MASTER_PORT

echo "[INFO] NUM_GPUS=${NUM_GPUS} | MASTER_ADDR=${MASTER_ADDR} | MASTER_PORT=${MASTER_PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

########################################################
# ====== AUTO-DOWNLOAD HF DATASET IF MISSING HDF5 =====
########################################################
REQ_DIR="${BASE_URI%/}/${NEW_DS}/data/train"
if ! compgen -G "${REQ_DIR}/*.hdf5" > /dev/null; then
  echo "[WARN] No local HDF5 in '${REQ_DIR}'. Will download '${NEW_DS}' from '${REMOTE_URI}'."
  python - <<'PY'
import os, sys
from pathlib import Path
from glob import glob
from huggingface_hub import snapshot_download

def norm(s: str) -> str:
    return (s or "").strip()

REMOTE_URI = norm(os.environ.get("REMOTE_URI",""))
NEW_DS     = norm(os.environ.get("NEW_DS",""))
BASE_URI   = Path(os.environ.get("BASE_URI","./the_well_local")).expanduser()
TOKEN      = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or ""

if not NEW_DS:
    print("[ERROR] NEW_DS is empty; set e.g. viscoelastic_instability.", file=sys.stderr)
    sys.exit(2)

# strip trailing slash
if REMOTE_URI.endswith("/"):
    REMOTE_URI = REMOTE_URI[:-1]

# Build repo_id from supported forms:
#  1) hf://datasets/<org>
#  2) <org>
#  3) <org>/<dataset>   (neste caso, NEW_DS é ignorado)
repo_id = None
if REMOTE_URI.startswith("hf://datasets/"):
    org = REMOTE_URI.split("hf://datasets/",1)[1].strip("/")
    if not org:
        print("[ERROR] REMOTE_URI malformed: empty org after hf://datasets/.", file=sys.stderr); sys.exit(2)
    repo_id = f"{org}/{NEW_DS}"
elif "/" in REMOTE_URI and not REMOTE_URI.startswith("http"):
    parts = [p for p in REMOTE_URI.split("/") if p]
    if len(parts) >= 2:
        repo_id = "/".join(parts[:2])
    else:
        print("[ERROR] REMOTE_URI contains '/', but not enough components.", file=sys.stderr); sys.exit(2)
else:
    if not REMOTE_URI:
        print("[ERROR] REMOTE_URI empty; use 'hf://datasets/<org>/' or '<org>' or '<org>/<dataset>'.", file=sys.stderr)
        sys.exit(2)
    repo_id = f"{REMOTE_URI}/{NEW_DS}"

if not repo_id or repo_id.count("/") != 1:
    print(f"[ERROR] Invalid repo_id='{repo_id}'.", file=sys.stderr); sys.exit(2)
org, ds = repo_id.split("/", 1)
if not org.strip() or not ds.strip():
    print(f"[ERROR] repo_id has empty fields: '{repo_id}'.", file=sys.stderr); sys.exit(2)

dst = BASE_URI / ds
dst.mkdir(parents=True, exist_ok=True)
print(f"[INFO] snapshot_download(repo_id='{repo_id}', dst='{dst}')")
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dst),
        allow_patterns=["data/**"],
        max_workers=8,
        token=TOKEN or None
    )
except Exception as e:
    print(f"[ERROR] snapshot_download failed: {e}", file=sys.stderr)
    sys.exit(2)

if not glob(str(dst / "data/train/*.hdf5")):
    print("[ERROR] No HDF5 found in 'data/train' after download.", file=sys.stderr)
    sys.exit(2)

print("[INFO] Dataset precache OK.")
PY
else
  echo "[INFO] HDF5 files already present in '${REQ_DIR}'."
fi

#########################################
# ====== CHECKPOINT / OUTPUT FOLDERS ===
#########################################
CKPT=${CKPT:-"./checkpoints_foundation_model_ablation_full/foundation_epoch48.pt"}
SAVE_DIR=${SAVE_DIR:-"./ckpt_${NEW_DS}_ft"}
mkdir -p "${SAVE_DIR}"
export PHX_JSON_DEFAULT_STR=1

#########################################
# ====== COMMON TRAINING ARGUMENTS =====
#########################################
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

#########################################
# ====== LAUNCH (TORCHRUN vs PYTHON) ===
#########################################
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[INFO] Launching DDP with ${NUM_GPUS} GPUs"
  exec torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${COMMON[@]}"
else
  echo "[INFO] Launching single process"
  exec python "${COMMON[@]}"
fi
