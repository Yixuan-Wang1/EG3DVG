#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/jiajun.xie/3D_Box/data}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/eg3dvg_assets}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/splits}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/dataset_info}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/eg3dvg_outputs}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29531}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCH="${MAX_EPOCH:-100}"

for required in \
  "${SPLIT_ROOT}/train.json" \
  "${SPLIT_ROOT}/valid.json" \
  "${MODEL_ROOT}/roberta-base/config.json" \
  "${MODEL_ROOT}/pointnet_backbone.pth"; do
  [[ -e "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 2; }
done

for source in dopose hope housecat omni_filter ycbv; do
  [[ -f "${MANIFEST_ROOT}/${source}/manifest.json" ]] || {
    echo "Missing dataset manifest: ${MANIFEST_ROOT}/${source}/manifest.json" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if command -v torchrun >/dev/null 2>&1; then
  LAUNCHER=(torchrun --standalone --nnodes=1
    --nproc-per-node="${NPROC_PER_NODE}" --master-port="${MASTER_PORT}")
else
  echo "torchrun not found; using torch.distributed.launch" >&2
  LAUNCHER=(python -m torch.distributed.launch
    --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}")
fi

"${LAUNCHER[@]}" train_dist_mod.py \
  --custom_data_root "${DATA_ROOT}" \
  --custom_split_root "${SPLIT_ROOT}" \
  --custom_manifest_root "${MANIFEST_ROOT}" \
  --data_root "${MODEL_ROOT}/" \
  --dataset placement --test_dataset placement \
  --model EG --exp canonical_placement --log_dir "${OUTPUT_ROOT}" \
  --pp_checkpoint "${MODEL_ROOT}/pointnet_backbone.pth" \
  --num_decoder_layers 6 --num_points 50000 --superpoint_voxel_size 0.03 \
  --use_color --use_soft_token_loss --use_contrastive_align --self_attend \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --weight_decay 0.0005 --lr_backbone 1e-5 --lr 2e-4 --text_encoder_lr 1e-5 \
  --lr_decay_epochs 50 75 --val_freq 1 --save_freq 1 --print_freq 100 \
  --max_epoch "${MAX_EPOCH}" "$@"
