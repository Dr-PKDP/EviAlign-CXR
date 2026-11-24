#!/usr/bin/env bash
# Two sets of counterfactual passes in which the F1 thresholds and the 90%-specificity gate come from the full validation split of each
# checkpoint and are the same for every pass of that checkpoint. GPU=<id> picks the device; on a shared GPU start it through wait_and_run.sh:
#   NEED=30000 bash scripts/wait_and_run.sh scripts/12_full_threshold_and_patient_sample_passes.sh 2>&1 | tee logs/full_threshold_passes.log
# 1. runs/<run>/eval_ops_fullthr/  the identical-explanation noise/blur/mean pass of script 11 on the first 3,200 test images, with the
#    thresholds and gate of runs/<run>/eval (full validation split), for main and plain_vlm.
# 2. runs/<run>/eval_patients/     noise infill on a seeded patient-random subset of 3,200 images per corpus (configs/eval_faith_patients.yaml),
#    with the same thresholds and gate, every eligible pair scored, keyed random controls and five ECS masks, for main and plain_vlm and,
#    with REDUCED=1, for the reduced-protocol variants of the principal comparison (their thresholds come from runs/<run>/eval_noise,
#    which already covers their whole 3,000-study validation split).
# Pass 1 is summarised with tools/operator_summary.py --ops_subdir eval_ops_fullthr.
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
OPS="--del_frac 0.15 --max_pairs_per_batch 512 --rand_key --ecs_split --n_ecs_masks 5"

pass_ () { # run outdir thresholds_dir sets [extra args...]
  local r=$1 o=$2 d=runs/$1/$2 t=$3 sets=$4; shift 4
  if [ -f runs/$r/best.pt ] && [ -f $t/thresholds.json ] && [ ! -f $d/DONE ]; then
    mkdir -p $d && cp $t/thresholds.json $t/thresholds_spec.json $d/
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume --sets $sets $F "$@" > logs/eval_${r}_${o}.log 2>&1 && touch $d/DONE
  fi
}
for r in main plain_vlm; do
  pass_ $r eval_ops_fullthr runs/$r/eval configs/eval_faith.yaml --max_batches 100 --infill_multi noise,blur,mean $OPS
done
for r in main plain_vlm; do
  pass_ $r eval_patients runs/$r/eval configs/eval_faith_patients.yaml --infill noise $OPS
done
if [ "${REDUCED:-0}" = 1 ]; then
  for r in s_main s_main_s1 s_main_s2 s_no_align s_no_align_s1 s_no_align_s2 s_plain s_plain_s1 s_plain_s2 s_no_sc; do
    pass_ $r eval_patients runs/$r/eval_noise configs/eval_faith_patients.yaml --infill noise $OPS
  done
fi
