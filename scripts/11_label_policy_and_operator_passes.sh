#!/usr/bin/env bash
# Further analyses. GPU=<id> picks the device; on a shared GPU start it through wait_and_run.sh:
#   NEED=30000 bash scripts/wait_and_run.sh scripts/11_label_policy_and_operator_passes.sh 2>&1 | tee logs/label_policy_and_operators.log
# 1. classification metrics under one stated label policy (uncertain labels excluded; the u_ones policy is written beside it),
#    with label counts per corpus and finding, from the saved per-image probabilities (CPU).
# 2. counterfactual passes in which blur, noise and mean infill are applied to one and the same generated explanation, grounding
#    and set of eligible pairs, with random controls keyed to (image, finding, replicate), every eligible pair scored, and the ECS
#    reference averaged over five random masks (ecs_rand_first keeps the single-mask reference).
#    Full protocol: main and plain_vlm. Reduced protocol (REDUCED=1): the three seeds of the full objective, of the model without
#    alignment and instance terms and of the unguided model, and the model without the sufficiency-comprehensiveness term.
# Outputs: results/analysis/classification_policy.csv, label_counts.csv and runs/<run>/eval_ops/.
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results/analysis
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
SUB="--sets configs/eval_faith.yaml --max_batches 100"
OPS="--infill_multi noise,blur,mean --del_frac 0.15 --max_pairs_per_batch 512 --rand_key --ecs_split --n_ecs_masks 5"

python tools/classification_policy.py --runs runs/main runs/plain_vlm runs/classifier_only --out results/analysis

ops_pass () { # run
  local r=$1 d=runs/$1/eval_ops
  if [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ]; then
    mkdir -p $d && cp -n runs/$r/eval*/thresholds*.json $d/ 2>/dev/null
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume $SUB $F $OPS > logs/eval_${r}_eval_ops.log 2>&1 && touch $d/DONE
  fi
}
for r in main plain_vlm; do ops_pass $r; done
if [ "${REDUCED:-0}" = "1" ]; then
  for r in s_main s_main_s1 s_main_s2 s_no_align s_no_align_s1 s_no_align_s2 s_plain s_plain_s1 s_plain_s2 s_no_sc; do ops_pass $r; done
fi
echo done
