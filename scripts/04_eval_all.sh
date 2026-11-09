#!/usr/bin/env bash
# Evaluate finished runs, then build tables and figures. Safe to re-run: a finished pass leaves a DONE file; an interrupted pass resumes at the first unfinished set.
#   tmux new -s eval 'GPU=0 bash scripts/04_eval_all.sh'
# eval/        full external evaluation, finding-conditioned explanations (--decode forced, 90%-specificity gate), blur infill   [main, plain_vlm, classifier_only]
# eval_free/   free-text generation, all sets                                                                                    [main, plain_vlm]
# eval_noise/  Gaussian-noise infill (unseen by the sufficiency/comprehensiveness regulariser), NIH + ChestX-Det + RSNA subset    [all runs]
# eval_sub/    blur infill on the same fixed subset, so reduced-protocol ablations are compared like for like                   [reduced runs]
# eval_mean|blur_*/ further removal-operator checks on the subset                                                                 [main, plain_vlm]
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
F="--decode forced --force_spec 0.9"
SUB="--sets configs/eval_faith.yaml --max_batches 100"
ALLS="--sets configs/eval_sets.yaml"
ev () { # run subdir [args...]
  local r=$1 n=$2 d=runs/$1/$2; shift 2
  [ -f runs/$r/best.pt ] && [ ! -f $d/DONE ] && \
    CUDA_VISIBLE_DEVICES=${GPU:-0} python evaluate.py --ckpt runs/$r/best.pt --out $d --resume "$@" > logs/eval_${r}_${n}.log 2>&1 && touch $d/DONE || true
}
# 1. full evaluation of the three full-protocol models
ev main eval $ALLS $F --del_ins
ev classifier_only eval $ALLS --del_ins
ev plain_vlm eval $ALLS $F --del_ins
# 2. matched comparisons on the fixed subset, noise infill first (the evidence that is not circular)
RUNS="s_main s_no_align s_no_sc s_plain s_no_inst s_no_anat s_aggregate_align s_bidirectional s_kl_ba s_joint_encoder s_crossattn_gpt2 s_biomedclip_backbone s_main_s1 s_no_align_s1 s_plain_s1 s_main_s2 s_no_align_s2 s_plain_s2"
for r in main plain_vlm $RUNS; do
  ev $r eval_noise $SUB $F --infill noise --del_frac 0.15
done
for r in $RUNS; do
  ev $r eval_sub $SUB $F --del_ins
done
ev s_classifier_only eval $SUB
# 3. unseen-operator checks and free text for the full models
for r in main plain_vlm; do
  ev $r eval_mean $SUB $F --infill mean --del_frac 0.15
  for f in 0.05 0.30; do ev $r eval_blur_$f $SUB $F --del_frac $f; done
done
ev main eval_ig $SUB $F --del_ins --ig
ev main eval_free $ALLS --del_ins
ev plain_vlm eval_free $ALLS --del_ins
ev classifier_only eval_template $ALLS --template_text
# 4. tables and figures
ALL="runs/main runs/plain_vlm runs/classifier_only"
python tools/make_tables.py --runs $ALL --out results/tables || true
SUBRUNS=""; for r in $RUNS s_classifier_only; do [ -f runs/$r/eval_sub/DONE ] || [ -f runs/$r/eval/DONE ] && SUBRUNS="$SUBRUNS runs/$r"; done
python tools/make_tables.py --runs $SUBRUNS --eval_subdir eval_sub --out results/tables_ablation || true
python tools/make_tables.py --runs runs/main runs/plain_vlm $SUBRUNS --eval_subdir eval_noise --out results/tables_noise || true
for r in main plain_vlm; do python tools/text_by_finding.py --run runs/$r --set nih_test --manifest manifests/nih.csv --out results/text_by_finding_$r.csv > /dev/null 2>&1 || true; done
python tools/make_figures.py --tables results/tables --main_run runs/main --sets_yaml configs/eval_sets.yaml --out results/figures || true
