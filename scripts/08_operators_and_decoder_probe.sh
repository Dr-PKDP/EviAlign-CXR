#!/usr/bin/env bash
# Blur versus noise infill on identical pairs, decoder image use and compute cost. GPU=<id> picks the device.
#   bash scripts/08_operators_and_decoder_probe.sh 2>&1 | tee logs/operators_and_probe.log
# 1. blur infill on the same fixed subset as the noise pass, for the two full-protocol decoders, so that the
#    blur-versus-noise comparison is made on identical (image, finding) pairs.
# 2. how often one sentence asserts more than one finding (shared alignment targets).
# 3. what the decoder does with the image: absolute attention mass, image substitution, zero positive evidence.
# 4. parameters, peak memory and latency.
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
SUB="--sets configs/eval_faith.yaml --max_batches 100"

for r in main plain_vlm; do
  d=runs/$r/eval_sub
  if [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ]; then
    mkdir -p $d && cp -n runs/$r/eval/thresholds*.json $d/ 2>/dev/null
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume $SUB $F > logs/eval_${r}_eval_sub.log 2>&1 && touch $d/DONE
  fi
done
python tools/make_tables.py --runs runs/main runs/plain_vlm --eval_subdir eval_sub --out results/tables_sub_full || true

python tools/decoder_probe.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --n 512 --batch_size 8 --out results/decoder_probe_main.json
python tools/decoder_probe.py --ckpt runs/plain_vlm/best.pt --manifest manifests/nih.csv --split test --n 512 --batch_size 8 --out results/decoder_probe_plain.json
python tools/profile_compute.py --ckpt runs/main/best.pt --manifest manifests/nih.csv --split test --out results/compute_profile.csv
echo done
