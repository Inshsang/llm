#!/bin/bash

set -euo pipefail

GPU_ID=1
TARGET_HOLD_MB=45990
STEP_TRIGGER_MB=10240
CHECK_INTERVAL=60
PROJECT_SRC=/data/HTC/Project/llm/src
PYTHON_BIN=/data/HTC/Library/lamm/bin/python
TRAIN_CMD="bash scripts/train_lamm3d_openlamm.sh agent_zero lamm"
LOG_DIR=/data/HTC/Data/model_zoo/llm_exe/agent_zero/log_rest
HOLDER_LOG=${LOG_DIR}/gpu${GPU_ID}_holder.log
STATE_FILE=${LOG_DIR}/gpu${GPU_ID}_holder_state.json

mkdir -p "${LOG_DIR}"

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
        python - <<'PY' "${STATE_FILE}"
import json
import sys
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
    print(int(data.get('allocated_mb', 0)))
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
    raise SystemExit('usage: holder gpu_id target_mb step_trigger_mb state_file')

gpu_id = int(sys.argv[1])
target_mb = int(sys.argv[2])
step_trigger_mb = int(sys.argv[3])
state_file = sys.argv[4]
chunk_mb = 512
buffers = []
running = True
allocated = 0
last_free = 0

def write_state():
    with open(state_file, 'w') as f:
        json.dump({'allocated_mb': allocated}, f)

def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

torch.cuda.set_device(gpu_id)
_elems_per_mb = (1024 * 1024) // 4

def free_mb():
    free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
    return free_bytes // 1024 // 1024

def alloc_mb(mb):
    global allocated
    remain = mb
    while running and remain > 0:
        this_mb = min(chunk_mb, remain)
        try:
            buf = torch.empty(this_mb * _elems_per_mb, dtype=torch.float32, device=f'cuda:{gpu_id}')
            buffers.append(buf)
            allocated += this_mb
            remain -= this_mb
            write_state()
            print(f'[holder] allocated_mb={allocated}', flush=True)
        except RuntimeError as exc:
            print(f'[holder] allocation stopped at {allocated} MB: {exc}', flush=True)
            break

print(f'[holder] gpu={gpu_id} target_mb={target_mb} step_trigger_mb={step_trigger_mb}', flush=True)
write_state()
while running and allocated < target_mb:
    current_free = free_mb()
    if current_free >= step_trigger_mb:
        grant = min(current_free, target_mb - allocated)
        grant = (grant // chunk_mb) * chunk_mb
        if grant <= 0 and (target_mb - allocated) >= chunk_mb and current_free >= chunk_mb:
            grant = chunk_mb
        if grant > 0:
            print(f'[holder] free_mb={current_free}, trying to allocate {grant} MB', flush=True)
            alloc_mb(grant)
    time.sleep(2)

print(f'[holder] ready allocated_mb={allocated}', flush=True)
while running:
    time.sleep(1)
PY
    holder_pid=$!
}

if [ -z "${holder_pid}" ]; then
    start_holder
fi

echo "[waiter] monitoring gpu ${GPU_ID}; step trigger ${STEP_TRIGGER_MB} MB; target hold ${TARGET_HOLD_MB} MB"
while true; do
    if ! kill -0 "${holder_pid}" 2>/dev/null; then
        echo "[waiter] holder exited unexpectedly; restarting" >&2
        holder_pid=""
        start_holder
        sleep 3
    fi
    free_mb=$(query_free_mb)
    held_mb=$(query_state_mb)
    ts=$(date +"%F %T")
    echo "[waiter] ${ts} gpu${GPU_ID} free=${free_mb} MB held=${held_mb} MB"
    if [ "${held_mb}" -ge "${TARGET_HOLD_MB}" ]; then
        echo "[waiter] target hold reached; launching training"
        kill "${holder_pid}" 2>/dev/null || true
        wait "${holder_pid}" 2>/dev/null || true
        holder_pid=""
        cd "${PROJECT_SRC}"
        echo "[waiter] exec: ${TRAIN_CMD}"
        exec bash -lc "${TRAIN_CMD}"
    fi
    sleep "${CHECK_INTERVAL}"
done
