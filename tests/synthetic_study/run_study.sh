#!/usr/bin/env bash
# Controlled synthetic study (CPU, ~3 min per run): known lesion locations + a spurious marker.
set -e
cd "$(dirname "$0")/../.."
D=/tmp/evialign_study; mkdir -p $D/runs
[ -f $D/data/train.csv ] || python tests/make_synthetic.py --out $D/data --n 1200 --size 64 --shortcut 0.9
cat > $D/sets.yaml <<YML
sets:
  - {name: internal, manifest: $D/data/train.csv, split: test}
  - {name: external, manifest: $D/data/external.csv, split: test}
YML
for seed in 0 1 2; do
 for v in full no_align no_anat plain; do
  case $v in
   full) O="";;
   no_align) O="loss.w_align=0 loss.w_inst=0";;
   no_anat) O="loss.w_anat=0";;
   plain) O="loss.w_align=0 loss.w_inst=0 loss.w_anat=0 loss.w_suff=0 loss.w_comp=0";;
  esac
  R=$D/runs/${v}_s$seed
  python train.py --config configs/smoke.yaml seed=$seed output_dir=$R data.train_manifest=$D/data/train.csv \
     data.val_manifest=$D/data/train.csv train.epochs=8 $O > $R.log 2>&1 || { echo "fail $R"; continue; }
  python evaluate.py --ckpt $R/best.pt --sets $D/sets.yaml --out $R/eval --workers 0 --n_boot 200 --batch_size 32 \
     --max_new_tokens 32 > $R.eval.log 2>&1 || echo "eval fail $R"
  echo "done $R"
 done
done
