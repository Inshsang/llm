#!/bin/bash

set -euo pipefail

GPU_ID="${GPU_ID:-1}"
TARGET_HOLD_MB_REQUESTED="${TARGET_HOLD_MB:-30000}"
STEP_TRIGGER_MB="${STEP_TRIGGER_MB:-4096}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
CHUNK_MB="${CHUNK_MB:-512}"

TARGET_HOLD_MB=$(( TARGET_HOLD_MB_REQUESTED / CHUNK_MB * CHUNK_MB ))
if [ "${TARGET_HOLD_MB}" -le 0 ]; then
    TARGET_HOLD_MB="${CHUNK_MB}"
fi

PROJECT_ROOT=/data/HTC/Project/llm
PYTHON_BIN=/data/HTC/Library/lamm/bin/python
DELTA_CKPT=/data/HTC/Data/model_zoo/llm_exe/agent_zero/pytorch_model.pt
VICUNA_CKPT=/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0
ENCODER_CKPT=/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth

ANS_DIR=${PROJECT_ROOT}/answers/agent_zero_inference3d
EVAL_DIR=${PROJECT_ROOT}/answers/agent_zero_eval
BACKUP_DIR=${EVAL_DIR}/backup
LOG_PATH=${EVAL_DIR}/inference3d_Classification.log
HOLDER_LOG=${EVAL_DIR}/gpu${GPU_ID}_classification_holder.log
STATE_FILE=${EVAL_DIR}/gpu${GPU_ID}_classification_holder_state.json
PRED_DIR=${EVAL_DIR}/inference3d_pred

TASKS=(
    Classification
    Detection
    Counting
    PositionRelation
    RoomDetection
    VisualGrounding_plus
)

mkdir -p "${ANS_DIR}" "${EVAL_DIR}" "${BACKUP_DIR}" "${PRED_DIR}"

holder_pid=""

cleanup() {
    if [ -n "${holder_pid}" ] && kill -0 "${holder_pid}" 2>/dev/null; then
        kill "${holder_pid}" 2>/dev/null || true
        wait "${holder_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

query_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d '[:space:]'
}

query_state_mb() {
    if [ -f "${STATE_FILE}" ]; then
        "${PYTHON_BIN}" - <<'PY' "${STATE_FILE}"
import json
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
    print(int(data.get("allocated_mb", 0)))
except Exception:
    print(0)
PY
    else
        echo 0
    fi
}

start_holder() {
    : > "${HOLDER_LOG}"
    "${PYTHON_BIN}" -u - "${GPU_ID}" "${TARGET_HOLD_MB}" "${STEP_TRIGGER_MB}" "${STATE_FILE}" > "${HOLDER_LOG}" 2>&1 <<'PY' &
import json
import signal
import sys
import time

import torch

if len(sys.argv) < 5:
    raise SystemExit("usage: holder gpu_id target_mb step_trigger_mb state_file")

gpu_id = int(sys.argv[1])
target_mb = int(sys.argv[2])
step_trigger_mb = int(sys.argv[3])
state_file = sys.argv[4]
chunk_mb = 512
buffers = []
running = True
allocated = 0
elems_per_mb = (1024 * 1024) // 4

def write_state():
    with open(state_file, "w") as f:
        json.dump({"allocated_mb": allocated}, f)

def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

torch.cuda.set_device(gpu_id)

def free_mb():
    free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
    return free_bytes // 1024 // 1024

def alloc_mb(mb):
    global allocated
    remain = mb
    while running and remain > 0:
        this_mb = min(chunk_mb, remain)
        try:
            buf = torch.empty(this_mb * elems_per_mb, dtype=torch.float32, device=f"cuda:{gpu_id}")
            buffers.append(buf)
            allocated += this_mb
            remain -= this_mb
            write_state()
            print(f"[holder] allocated_mb={allocated}", flush=True)
        except RuntimeError as exc:
            print(f"[holder] allocation stopped at {allocated} MB: {exc}", flush=True)
            break

print(f"[holder] gpu={gpu_id} target_mb={target_mb} step_trigger_mb={step_trigger_mb}", flush=True)
write_state()
while running and allocated < target_mb:
    current_free = free_mb()
    if current_free >= step_trigger_mb:
        grant = min(current_free, target_mb - allocated)
        grant = (grant // chunk_mb) * chunk_mb
        if grant <= 0 and (target_mb - allocated) >= chunk_mb and current_free >= chunk_mb:
            grant = chunk_mb
        if grant > 0:
            print(f"[holder] free_mb={current_free}, trying to allocate {grant} MB", flush=True)
            alloc_mb(grant)
    time.sleep(2)

print(f"[holder] ready allocated_mb={allocated}", flush=True)
while running:
    time.sleep(1)
PY
    holder_pid=$!
}

backup_task_outputs() {
    local task="$1"
    local ts="$2"
    local json_path="${ANS_DIR}/${task}.jsonl"
    local log_path="${EVAL_DIR}/inference3d_${task}.log"
    if [ -f "${json_path}" ]; then
        mv "${json_path}" "${BACKUP_DIR}/${task}.jsonl.${ts}"
    fi
    if [ -f "${log_path}" ]; then
        mv "${log_path}" "${BACKUP_DIR}/inference3d_${task}.log.${ts}"
    fi
}

run_task() {
    local task="$1"
    local log_path="${EVAL_DIR}/inference3d_${task}.log"
    echo "[run] start ${task}"
    "${PYTHON_BIN}" src/inference_3d.py \
        --task_type "${task}" \
        --delta_ckpt_path "${DELTA_CKPT}" \
        --vicuna_ckpt_path "${VICUNA_CKPT}" \
        --encoder_ckpt_path "${ENCODER_CKPT}" \
        --answers-dir "${ANS_DIR}" \
        --max_tgt_len 1200 \
        2>&1 | tee "${log_path}"

    if [ -f "${ANS_DIR}/${task}.jsonl" ]; then
        cp "${ANS_DIR}/${task}.jsonl" "${PRED_DIR}/Agent_${task}.jsonl"
    fi
}

echo "[waiter] monitor gpu ${GPU_ID}; requested hold ${TARGET_HOLD_MB_REQUESTED} MB; effective hold ${TARGET_HOLD_MB} MB"
start_holder

while true; do
    if ! kill -0 "${holder_pid}" 2>/dev/null; then
        echo "[waiter] holder exited unexpectedly; restarting" >&2
        holder_pid=""
        start_holder
        sleep 3
    fi

    free_mb="$(query_free_mb)"
    held_mb="$(query_state_mb)"
    ts="$(date +"%F %T")"
    echo "[waiter] ${ts} gpu${GPU_ID} free=${free_mb} MB held=${held_mb} MB"

    if [ "${held_mb}" -ge "${TARGET_HOLD_MB}" ]; then
        echo "[waiter] target hold reached; launch inference"
        break
    fi

    sleep "${CHECK_INTERVAL}"
done

kill "${holder_pid}" 2>/dev/null || true
wait "${holder_pid}" 2>/dev/null || true
holder_pid=""

ts="$(date +%Y%m%d_%H%M%S)"
for task in "${TASKS[@]}"; do
    backup_task_outputs "${task}" "${ts}"
done

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for task in "${TASKS[@]}"; do
    run_task "${task}"
done

exec "${PYTHON_BIN}" src/ablation_stats.py \
    --agent-dir "${PRED_DIR}" \
    --agent-classification "${PRED_DIR}/Agent_Classification.jsonl" \
    --agent-detection "${PRED_DIR}/Agent_Detection.jsonl" \
    --out-csv "${EVAL_DIR}/inference3d_summary.csv" \
    --out-md "${EVAL_DIR}/inference3d_summary.md" \
    --out-plot "${EVAL_DIR}/inference3d_summary.png"
