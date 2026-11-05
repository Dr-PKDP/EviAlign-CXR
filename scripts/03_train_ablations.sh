#!/usr/bin/env bash
# Reduced protocol for ablations and seed variance: the same 30,000-study subset and 6 epochs for every run, so each ablation is
# compared with the matching reduced-protocol main model (s_main) and not with the full run.
#   tmux new -s abl 'GPU=1 bash scripts/03_train_ablations.sh'
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p logs
SMALL="data.max_train_items=30000 data.max_val_items=3000 train.epochs=6 train.select_from_epoch=3 train.warmup_steps=300 \
train.stages.cls_only_epochs=1 train.stages.align_start_epoch=2 train.stages.align_ramp_epochs=1 train.stages.sc_start_epoch=2"
run () { # config name [extra overrides]
  local cfg=$1 name=$2; shift 2
  [ -f runs/$name/last.pt ] || CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --config $cfg output_dir=runs/$name $SMALL "$@" > logs/$name.log 2>&1
}
# order: the matched comparisons that carry the faithfulness claim first (full vs no alignment, no sufficiency/comprehensiveness, unguided),
# then the remaining components, then seeds. Light runs (classifier_only, crossattn_gpt2) may be started separately with GPU=... in a second tmux.
run configs/main.yaml s_main
run configs/ablations/no_align.yaml s_no_align
run configs/ablations/no_sc.yaml s_no_sc
run configs/ablations/plain_vlm.yaml s_plain
for n in no_inst no_anat aggregate_align bidirectional kl_ba joint_encoder biomedclip_backbone crossattn_gpt2 classifier_only; do
  run configs/ablations/$n.yaml s_$n
done
run configs/main.yaml s_main_s1 seed=1
run configs/ablations/plain_vlm.yaml s_plain_s1 seed=1
run configs/main.yaml s_main_s2 seed=2
run configs/ablations/plain_vlm.yaml s_plain_s2 seed=2
