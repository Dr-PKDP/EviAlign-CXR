#!/usr/bin/env bash
# Seed replicates of the three runs that carry the faithfulness comparison (full objective, no alignment, unguided), then the remaining ablations.
# Same reduced protocol as 03_train_ablations.sh. Finished runs are skipped, so stopping and restarting is safe.
#   tmux new -d -s abl 'GPU=0 bash scripts/03_train_seeds.sh'
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs
SMALL="data.max_train_items=30000 data.max_val_items=3000 train.epochs=6 train.select_from_epoch=3 train.warmup_steps=300 \
train.stages.cls_only_epochs=1 train.stages.align_start_epoch=2 train.stages.align_ramp_epochs=1 train.stages.sc_start_epoch=2"
run () { local cfg=$1 name=$2; shift 2
  [ -f runs/$name/last.pt ] || CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --config $cfg output_dir=runs/$name $SMALL "$@" > logs/$name.log 2>&1; }
for s in 1 2; do
  run configs/main.yaml s_main_s$s seed=$s
  run configs/ablations/no_align.yaml s_no_align_s$s seed=$s
  run configs/ablations/plain_vlm.yaml s_plain_s$s seed=$s
done
bash scripts/03_train_ablations.sh
