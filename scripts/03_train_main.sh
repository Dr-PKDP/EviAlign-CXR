#!/usr/bin/env bash
# Full protocol (12 epochs, all 98,770 training studies): the main model and its two reference baselines, one after the other.
# ~0.9 s/step at batch 32 on one H200 (about 45 min per epoch for the LLM variants).
#   tmux new -s main 'GPU=0 bash scripts/03_train_main.sh'
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p logs
run () { CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --config $1 > logs/$(basename $1 .yaml).log 2>&1; }
[ -f runs/main/last.pt ]           || run configs/main.yaml
[ -f runs/plain_vlm/last.pt ]      || run configs/ablations/plain_vlm.yaml
[ -f runs/classifier_only/last.pt ] || run configs/ablations/classifier_only.yaml
