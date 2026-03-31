#!/bin/bash

set -euo pipefail

GPU_ID="${GPU_ID:-1}"
FREE_MB_THRESHOLD="${FREE_MB_THRESHOLD:-30000}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-20}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

PROJECT_ROOT=/data/HTC/Project/llm
PROJECT_SRC=${PROJECT_ROOT}/src
PYTHON_BIN=/data/HTC/Library/lamm/bin/python
MODEL_DIR=/data/HTC/Data/model_zoo/llm_exe/agent_zero
DELTA_CKPT=${MODEL_DIR}/pytorch_model.pt
VICUNA_CKPT=/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0
ENCODER_CKPT=/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth

INFER_DIR=${PROJECT_ROOT}/answers/agent_zero_inference3d
AGENT_DIR=${PROJECT_ROOT}/answers/agent_zero_agent
EVAL_DIR=${PROJECT_ROOT}/answers/agent_zero_eval
INFER_PRED_DIR=${EVAL_DIR}/inference3d_pred

mkdir -p "${INFER_DIR}" "${AGENT_DIR}" "${EVAL_DIR}" "${INFER_PRED_DIR}"

INFER_TASKS=(
  Classification
  Detection
  Counting
  PositionRelation
  RoomDetection
  VisualGrounding_plus
)

AGENT_TASKS=(
  Classification
  Detection
  Counting
  PositionRelation
  RoomDetection
  VisualGrounding_plus
)

query_gpu() {
    nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "${GPU_ID}" | head -n 1
}

wait_for_gpu() {
    while true; do
        read -r free_mb util <<<"$(query_gpu | tr ',' ' ')"
        ts=$(date +"%F %T")
        echo "[wait] ${ts} gpu=${GPU_ID} free=${free_mb}MB util=${util}%"
        if [ "${free_mb}" -ge "${FREE_MB_THRESHOLD}" ] && [ "${util}" -le "${UTIL_THRESHOLD}" ]; then
            break
        fi
        sleep "${CHECK_INTERVAL}"
    done
}

run_inference3d() {
    local task="$1"
    local log_path="${EVAL_DIR}/inference3d_${task}.log"
    echo "[run] inference_3d.py task=${task}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_SRC}/inference_3d.py" \
        --task_type "${task}" \
        --delta_ckpt_path "${DELTA_CKPT}" \
        --vicuna_ckpt_path "${VICUNA_CKPT}" \
        --encoder_ckpt_path "${ENCODER_CKPT}" \
        --answers-dir "${INFER_DIR}" \
        --max_tgt_len 1200 \
        2>&1 | tee "${log_path}"

    if [ -f "${INFER_DIR}/${task}.jsonl" ]; then
        cp "${INFER_DIR}/${task}.jsonl" "${INFER_PRED_DIR}/Agent_${task}.jsonl"
    fi
}

run_agent() {
    local task="$1"
    local log_path="${EVAL_DIR}/agent_${task}.log"
    echo "[run] agent_inference.py task=${task}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_SRC}/agent_inference.py" \
        --gpu 0 \
        --task_type "${task}" \
        --delta_ckpt_path "${DELTA_CKPT}" \
        --vicuna_ckpt_path "${VICUNA_CKPT}" \
        --encoder_ckpt_path "${ENCODER_CKPT}" \
        --answers-dir "${AGENT_DIR}" \
        2>&1 | tee "${log_path}"
}

run_summary() {
    local pred_dir="$1"
    local prefix="$2"
    "${PYTHON_BIN}" "${PROJECT_SRC}/ablation_stats.py" \
        --agent-dir "${pred_dir}" \
        --agent-classification "${pred_dir}/Agent_Classification.jsonl" \
        --agent-detection "${pred_dir}/Agent_Detection.jsonl" \
        --out-csv "${EVAL_DIR}/${prefix}_summary.csv" \
        --out-md "${EVAL_DIR}/${prefix}_summary.md" \
        --out-plot "${EVAL_DIR}/${prefix}_summary.png"
}

main() {
    wait_for_gpu

    cd "${PROJECT_ROOT}"
    for task in "${INFER_TASKS[@]}"; do
        run_inference3d "${task}"
    done
    run_summary "${INFER_PRED_DIR}" "inference3d"

    for task in "${AGENT_TASKS[@]}"; do
        run_agent "${task}"
    done
    run_summary "${AGENT_DIR}" "agent"

    echo "[done] results under ${EVAL_DIR}"
}

main "$@"
