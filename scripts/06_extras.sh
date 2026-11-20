#!/usr/bin/env bash
# Evaluations that need the tuberculosis manifests, the zero-shot VLM baseline and the model-randomisation check.
#   tmux new -s extras 'GPU=0 bash scripts/06_extras.sh 2>&1 | tee logs/extras.log'
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
# 1. tuberculosis corpora, three full-protocol systems
for r in main plain_vlm classifier_only; do
  d=runs/$r/eval_tb
  if [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ]; then
    mkdir -p $d && cp -n runs/$r/eval/thresholds*.json $d/     # same operating points as the main evaluation
    if [ $r = classifier_only ]; then X=""; else X="$F"; fi
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume --sets configs/eval_tb.yaml $X > logs/eval_${r}_eval_tb.log 2>&1 && touch $d/DONE
  fi
done
python tools/make_tables.py --runs runs/main runs/plain_vlm runs/classifier_only --eval_subdir eval_tb --out results/tables_tb || true
# 2. model-randomisation check on NIH test images
python tools/randomisation_check.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 256 --out results/randomisation.csv > logs/randomisation.log 2>&1 || true
# 3. zero-shot open VLM (Qwen2.5-VL-7B-Instruct), 2,000-image subsets of NIH and IU X-ray
python baselines/zeroshot_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --sets configs/eval_zs.yaml --out runs/zs_qwen25vl --max_items 2000 --batch_size 8 > logs/zeroshot.log 2>&1
echo extras finished
