#!/usr/bin/env bash
# (1) Model-randomisation variants (class held fixed, unrectified, absolute, random-vs-random control).
# (2) Zero-shot open VLM with a 512-token generation cap.
#   tmux new -s checks 'GPU=0 bash scripts/07_randomisation_and_zeroshot.sh 2>&1 | tee logs/checks.log'
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
python tools/randomisation_check.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 256 --out results/randomisation_diagnostic.csv
python baselines/zeroshot_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --sets configs/eval_zs.yaml --out runs/zs_qwen25vl_512 --max_items 2000 --batch_size 8 --max_new_tokens 512
echo checks finished
