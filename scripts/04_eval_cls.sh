#!/usr/bin/env bash
# Evaluate classifier_only (light; fits beside a running LM job). Skipped if already done.
#   tmux new -s evcls 'GPU=0 bash scripts/04_eval_cls.sh'
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p logs
[ -f runs/classifier_only/eval/summary.json ] || CUDA_VISIBLE_DEVICES=${GPU:-0} python evaluate.py --ckpt runs/classifier_only/best.pt \
  --sets configs/eval_sets.yaml --out runs/classifier_only/eval --del_ins --ig > logs/eval_classifier_only.log 2>&1
[ -f runs/classifier_only/eval_template/summary.json ] || CUDA_VISIBLE_DEVICES=${GPU:-0} python evaluate.py --ckpt runs/classifier_only/best.pt \
  --sets configs/eval_sets.yaml --out runs/classifier_only/eval_template --template_text > logs/eval_classifier_only_template.log 2>&1
tail -n 5 logs/eval_classifier_only.log
