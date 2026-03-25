#!/bin/bash

set -euo pipefail

# Usage:
#   ./train_lamm3d_openlamm.sh <exp_name> [cuda_visible_devices]
# Example:
#   ./train_lamm3d_openlamm.sh lamm3d_openlamm 1
#   ./train_lamm3d_openlamm.sh lamm3d_openlamm 1,2
#
# The second argument uses physical GPU ids for CUDA_VISIBLE_DEVICES.
# Inside DeepSpeed these GPUs are remapped to localhost:0..N-1.

EXP_NAME=${1:-lamm3d_openlamm}
CUDA_DEVICES=${2:-0}
MASTER_PORT=${MASTER_PORT:-28461}
NOW=$(date +"%Y%m%d_%H%M%S")
VISFEAT_TYPE=local

PROJECT_ROOT=/data/HTC/Project/llm
SRC_DIR=${PROJECT_ROOT}/src

PYTHON_BIN=/data/HTC/Library/lamm/bin/python
DEEPSPEED_BIN=/data/HTC/Library/lamm/bin/deepspeed

CFG_PATH=${SRC_DIR}/config/train_ds3.yaml
DATA_ROOT=/data/HTC/Data/dataset/Benchmark/data/LAMM/3D_Instruct
DATA_PATH=${DATA_ROOT}/meta_file/LAMM_3dinstruct_10k.json
EPCL_CKPT=/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth
VICUNA_CKPT=/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0

CKPT_ROOT=/data/HTC/Data/model_zoo/llm_exe
SAVE_PATH=${CKPT_ROOT}/${EXP_NAME}
LOG_PATH=${SAVE_PATH}/log_rest

mkdir -p "${LOG_PATH}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -x "${DEEPSPEED_BIN}" ]; then
    echo "DeepSpeed not found: ${DEEPSPEED_BIN}" >&2
    exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
    echo "Training meta file not found: ${DATA_PATH}" >&2
    exit 1
fi

if [ ! -f "${EPCL_CKPT}" ]; then
    echo "EPCL checkpoint not found: ${EPCL_CKPT}" >&2
    exit 1
fi

if [ ! -d "${VICUNA_CKPT}" ]; then
    echo "Vicuna checkpoint directory not found: ${VICUNA_CKPT}" >&2
    exit 1
fi

if [ ! -d "${DATA_ROOT}/3rscan_pcls" ] || [ ! -d "${DATA_ROOT}/shapenet_pcls" ]; then
    echo "OpenLAMM 3D data is still zipped. Please unzip 3rscan_pcls.zip and shapenet_pcls.zip under ${DATA_ROOT} before training." >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_DEVICES}"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ "${GPU_COUNT}" -le 0 ]; then
    echo "No GPU selected. Example: 0 or 1,2" >&2
    exit 1
fi

DEEPSPEED_SLOTS=$(seq -s, 0 $((GPU_COUNT - 1)))

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

cd "${SRC_DIR}"

echo "EXP_NAME=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DeepSpeed include=localhost:${DEEPSPEED_SLOTS}"
echo "DATA_PATH=${DATA_PATH}"
echo "SAVE_PATH=${SAVE_PATH}"

"${DEEPSPEED_BIN}" --include "localhost:${DEEPSPEED_SLOTS}" --master_addr 127.0.0.1 --master_port "${MASTER_PORT}" train.py \
    --train_stage 3 \
    --cfg "${CFG_PATH}" \
    --data_path "${DATA_PATH}" \
    --vision_root_path "${DATA_ROOT}" \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path "${EPCL_CKPT}" \
    --vicuna_ckpt_path "${VICUNA_CKPT}" \
    --vision_feature_type "${VISFEAT_TYPE}" \
    --num_vision_token 256 \
    --save_path "${SAVE_PATH}" \
    --log_path "${LOG_PATH}" \
    2>&1 | tee "${LOG_PATH}/train_${NOW}.log"
