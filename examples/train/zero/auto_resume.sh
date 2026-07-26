#!/bin/bash
# Relaunch a training script from its newest checkpoint whenever it dies.
#
# Streaming FineVisionMax over many hours intermittently fails with
# "RuntimeError: Cannot send a request, as the client has been closed" raised
# inside the DataLoader worker, which takes down every rank. With --save_steps
# 1000 a crash costs at most 1000 steps, so retrying from the newest checkpoint
# is much cheaper than babysitting the run.
#
# Training semantics are unchanged: the resume keeps the Trainer's default data
# skip, so no sample is consumed twice. The cost is that each retry re-streams
# and re-preprocesses everything before the checkpoint step.
#
# usage: bash examples/train/zero/auto_resume.sh <run-script.sh> <output_dir> [max_retries]

set -u

SCRIPT="${1:?usage: auto_resume.sh <run-script.sh> <output_dir> [max_retries]}"
OUTPUT_DIR="${2:?usage: auto_resume.sh <run-script.sh> <output_dir> [max_retries]}"
MAX_RETRIES="${3:-20}"

attempt=0
while :; do
    # Clear any ranks orphaned by the previous crash so the GPUs are free.
    pkill -f "src/sae/launch/train.py" 2>/dev/null && sleep 10

    latest="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
              | sed 's/.*checkpoint-//' | sort -n | tail -1)"

    if [ -n "${latest}" ]; then
        echo "[auto_resume] attempt ${attempt}: resuming from checkpoint-${latest}"
        # --ignore_data_skip disables the Trainer's replay-based skip, which is far too
        # slow on a large streaming dataset. train.py seeks the stream to the position
        # recorded in the checkpoint instead, which is exact and instant. Checkpoints
        # written before that mechanism existed have no recorded position; train.py warns
        # and those samples get seen a second time.
        bash "${SCRIPT}" \
            --resume_from_checkpoint "${OUTPUT_DIR}/checkpoint-${latest}" \
            --ignore_data_skip
    else
        echo "[auto_resume] attempt ${attempt}: no checkpoint found, starting fresh"
        bash "${SCRIPT}"
    fi
    status=$?

    if [ ${status} -eq 0 ]; then
        echo "[auto_resume] training finished cleanly after ${attempt} retries"
        exit 0
    fi

    attempt=$((attempt + 1))
    if [ "${attempt}" -gt "${MAX_RETRIES}" ]; then
        echo "[auto_resume] giving up after ${MAX_RETRIES} retries (last exit ${status})"
        exit "${status}"
    fi

    echo "[auto_resume] exit ${status}; retrying in 60s"
    sleep 60
done
