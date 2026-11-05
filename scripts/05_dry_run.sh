#!/usr/bin/env bash
# Short real-data run of the full objective (every loss active from step 0) to check memory, speed and loss values.
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p logs
CUDA_VISIBLE_DEVICES=${GPU:-0} python train.py --config configs/main.yaml output_dir=runs/dry \
  data.max_train_items=3000 data.max_val_items=600 data.num_workers=8 \
  train.epochs=2 train.max_steps_per_epoch=60 train.warmup_steps=10 train.log_every=5 train.select_from_epoch=0 \
  train.stages.cls_only_epochs=0 train.stages.align_start_epoch=0 train.stages.align_ramp_epochs=0 \
  train.stages.sc_start_epoch=0 train.stages.anat_start_epoch=0 2>&1 | tee logs/dry_run.log
