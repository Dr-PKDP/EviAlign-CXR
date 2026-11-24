#!/usr/bin/env bash
# Training runs for the controls that separate the effect of the alignment objective from the effects of its side conditions.
# Reduced protocol (same subset, epochs and stages as 03_train_ablations.sh), three seeds each (seed 0, 1, 2):
#   s_sep_main, s_sep_no_align, s_sep_plain   gradient clipping applied separately to the classifier branch (encoder and evidence head)
#                                             and to the decoder branch (train.clip_mode=separate, each branch clipped at norm 1.0), so
#                                             that the classifier receives the same update whatever the decoder loss is
#   s_minimal                                 classification, language-model and alignment losses only (no instance, anatomy,
#                                             sufficiency or comprehensiveness term)
# FULL=1 adds full-protocol seed replicates of the main model and the unguided VLM (about 9 h per run):
#   main_s1, main_s2, plain_vlm_s1, plain_vlm_s2
# Finished runs are skipped, so stopping and restarting is safe. On a shared GPU start it through wait_and_run.sh:
#   tmux new -d -s lim 'NEED=30000 bash scripts/wait_and_run.sh scripts/13_train_limitation_runs.sh'
#   tmux new -d -s limfull 'FULL=1 NEED=30000 bash scripts/wait_and_run.sh scripts/13_train_limitation_runs.sh'
set -e
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs
SMALL="data.max_train_items=30000 data.max_val_items=3000 train.epochs=6 train.select_from_epoch=3 train.warmup_steps=300 \
train.stages.cls_only_epochs=1 train.stages.align_start_epoch=2 train.stages.align_ramp_epochs=1 train.stages.sc_start_epoch=2"
run () { # config name [extra overrides]
  local cfg=$1 name=$2; shift 2
  [ -f runs/$name/last.pt ] || CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --config $cfg output_dir=runs/$name "$@" > logs/$name.log 2>&1
}
if [ "${FULL:-0}" = "1" ]; then
  for s in 1 2; do
    run configs/main.yaml main_s$s seed=$s
    run configs/ablations/plain_vlm.yaml plain_vlm_s$s seed=$s
  done
else
  for s in 0 1 2; do
    sfx=""; [ $s -gt 0 ] && sfx="_s$s"
    run configs/main.yaml s_sep_main$sfx seed=$s $SMALL train.clip_mode=separate
    run configs/ablations/no_align.yaml s_sep_no_align$sfx seed=$s $SMALL train.clip_mode=separate
    run configs/ablations/plain_vlm.yaml s_sep_plain$sfx seed=$s $SMALL train.clip_mode=separate
    run configs/ablations/minimal.yaml s_minimal$sfx seed=$s $SMALL
  done
fi
