#!/usr/bin/env bash
# Evaluation of the runs of scripts 13. GPU=<id> picks the device; on a shared GPU start it through wait_and_run.sh:
#   tmux new -d -s limev 'NEED=30000 bash scripts/wait_and_run.sh scripts/14_eval_limitation_runs.sh'
# Reduced-protocol runs: eval_noise (noise infill on the fixed subset, as in 04_eval_all.sh) and eval_ops (identical-explanation
# noise/blur/mean pass with keyed random controls, as in script 11).
# Full-protocol replicates (main_s1, main_s2, plain_vlm_s1, plain_vlm_s2): eval (full external evaluation), eval_noise_full (uncapped
# pairs, thresholds of eval) and eval_ops.
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
SUB="--sets configs/eval_faith.yaml --max_batches 100"
ALLS="--sets configs/eval_sets.yaml"
OPS="--infill_multi noise,blur,mean --del_frac 0.15 --max_pairs_per_batch 512 --rand_key --ecs_split --n_ecs_masks 5"
ev () { # run subdir [args...]
  local r=$1 n=$2 d=runs/$1/$2; shift 2
  if [ -f runs/$r/TRAINED ] && [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ]; then
    mkdir -p $d
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume "$@" > logs/eval_${r}_${n}.log 2>&1 && touch $d/DONE
  fi
}
for r in s_sep_main s_sep_no_align s_sep_plain s_minimal; do
  for sfx in "" _s1 _s2; do
    ev $r$sfx eval_noise $SUB $F --infill noise --del_frac 0.15
    ev $r$sfx eval_ops $SUB $F $OPS
  done
done
for r in main_s1 main_s2 plain_vlm_s1 plain_vlm_s2; do
  ev $r eval $ALLS $F --del_ins
  if [ -f runs/$r/eval/thresholds.json ] && [ ! -f runs/$r/eval_noise_full/DONE ]; then
    mkdir -p runs/$r/eval_noise_full && cp -n runs/$r/eval/thresholds*.json runs/$r/eval_noise_full/
    ev $r eval_noise_full $SUB $F --infill noise --del_frac 0.15 --max_pairs_per_batch 512 --rand_key --ecs_split
  fi
  ev $r eval_ops $SUB $F $OPS
done
echo done
