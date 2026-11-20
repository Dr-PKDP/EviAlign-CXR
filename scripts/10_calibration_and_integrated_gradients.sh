#!/usr/bin/env bash
# Reliability curves and the Integrated Gradients convergence check. On a shared GPU, start it through wait_and_run.sh:
#   NEED=20000 bash scripts/wait_and_run.sh scripts/10_calibration_and_integrated_gradients.sh 2>&1 | tee logs/calibration_ig.log
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results/analysis
export CUDA_VISIBLE_DEVICES=${GPU:-0}
python tools/calibration_curves.py --runs runs/main --out results/analysis
python tools/ig_check.py --ckpt runs/main/best.pt --loc_dir runs/main/eval --n 150 --out results/analysis
echo done
