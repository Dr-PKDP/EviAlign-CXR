#!/usr/bin/env bash
# Waits until a GPU has at least NEED MiB free (the server is shared), then runs the given script on it.
#   bash scripts/wait_and_run.sh scripts/09_uncapped_counterfactuals_and_tables.sh
NEED=${NEED:-20000}
while true; do
  for g in 0 1; do
    free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${free:-0}" -ge "$NEED" ]; then echo "$(date) GPU $g has $free MiB free, starting"; GPU=$g bash "$@"; exit $?; fi
  done
  sleep 120
done
