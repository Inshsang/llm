#!/bin/bash

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
FREE_MB_THRESHOLD="${FREE_MB_THRESHOLD:-30000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

PROJECT_ROOT=/data/HTC/Project/llm
PYTHON_BIN=/data/HTC/Library/lamm/bin/python
DELTA_CKPT=/data/HTC/Data/model_zoo/llm_exe/agent_zero/pytorch_model.pt
VICUNA_CKPT=/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0
ENCODER_CKPT=/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth

ANS_DIR=${PROJECT_ROOT}/answers/agent_zero_inference3d
EVAL_DIR=${PROJECT_ROOT}/answers/agent_zero_eval
BACKUP_DIR=${EVAL_DIR}/backup
LOG_PATH=${EVAL_DIR}/inference3d_Classification.log

mkdir -p "${ANS_DIR}" "${EVAL_DIR}" "${BACKUP_DIR}"

query_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d '[:space:]'
}

echo "[wait] monitoring gpu ${GPU_ID}; target free >= ${FREE_MB_THRESHOLD} MiB"
while true; do
    free_mb="$(query_free_mb)"
    ts="$(date +"%F %T")"
    echo "[wait] ${ts} gpu${GPU_ID} free=${free_mb} MiB"
    if [ "${free_mb}" -ge "${FREE_MB_THRESHOLD}" ]; then
        break
    fi
    sleep "${CHECK_INTERVAL}"
done

ts="$(date +%Y%m%d_%H%M%S)"
if [ -f "${ANS_DIR}/Classification.jsonl" ]; then
    mv "${ANS_DIR}/Classification.jsonl" "${BACKUP_DIR}/Classification.jsonl.${ts}"
fi
if [ -f "${LOG_PATH}" ]; then
    mv "${LOG_PATH}" "${BACKUP_DIR}/inference3d_Classification.log.${ts}"
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${PYTHON_BIN}" src/inference_3d.py \
    --task_type Classification \
    --delta_ckpt_path "${DELTA_CKPT}" \
    --vicuna_ckpt_path "${VICUNA_CKPT}" \
    --encoder_ckpt_path "${ENCODER_CKPT}" \
    --answers-dir "${ANS_DIR}" \
    --max_tgt_len 1200 \
    2>&1 | tee "${LOG_PATH}"
