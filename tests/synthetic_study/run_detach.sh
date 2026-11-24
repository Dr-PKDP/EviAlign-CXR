#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
D=/tmp/evialign_study
for seed in 0 1 2; do
  R=$D/runs/detach_s$seed
  python train.py --config configs/smoke.yaml seed=$seed output_dir=$R data.train_manifest=$D/data/train.csv \
     data.val_manifest=$D/data/train.csv train.epochs=8 model.decoder_detach_memory=true > $R.log 2>&1
  python evaluate.py --ckpt $R/best.pt --sets $D/sets.yaml --out $R/eval --workers 0 --n_boot 200 --batch_size 32 --max_new_tokens 32 > $R.eval.log 2>&1
  echo "done $R"
done
