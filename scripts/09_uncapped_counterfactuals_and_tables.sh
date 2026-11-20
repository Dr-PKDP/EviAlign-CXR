#!/usr/bin/env bash
# Uncapped counterfactual pass, sentence-sharing statistics and the supplementary tables. GPU=<id> picks the device.
#   bash scripts/09_uncapped_counterfactuals_and_tables.sh 2>&1 | tee logs/uncapped.log
# 1. counterfactual pass with every eligible pair scored (no per-batch cap), random-removal masks keyed to the pair so that
#    both models receive identical controls, and the sentence likelihood split into the forced opening and the decoder's
#    continuation.
# 2. sentence-sharing statistics with exact alignment-target equality, restricted to the modelled classes.
# 3. tables from evaluation output that already exists: tuberculosis intervals clustered by image, the gate and cap ledger,
#    external behaviour of the generation gate, per-finding metrics, localisation counts, per-seed rows for every corpus.
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
F="--decode forced --force_spec 0.9"
SUB="--sets configs/eval_faith.yaml --max_batches 100"

for r in main plain_vlm; do
  d=runs/$r/eval_noise_full
  if [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ]; then
    mkdir -p $d && cp -n runs/$r/eval/thresholds*.json $d/ 2>/dev/null
    python evaluate.py --ckpt runs/$r/best.pt --out $d --resume $SUB $F \
      --infill noise --del_frac 0.15 --max_pairs_per_batch 512 --rand_key --ecs_split \
      > logs/eval_${r}_eval_noise_full.log 2>&1 && touch $d/DONE
  fi
done

python tools/sentence_stats.py --manifest manifests/rexgradient.csv --split train --n 20000 --out results/analysis/sentence_sharing.csv
python tools/analysis_tables.py --out results/analysis

python tools/localisation_priors.py --runs runs/main runs/plain_vlm --out results/analysis
echo done
