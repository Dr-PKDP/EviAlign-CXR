#!/usr/bin/env bash
# Evaluate every finished run on all external sets, then build tables and figures. Safe to re-run; finished evaluations are skipped.
#   tmux new -s eval 'GPU=0 bash scripts/04_eval_all.sh'
# eval/        : finding-conditioned explanations (--decode forced, 90%-specificity gate), blur infill   -> primary tables
# eval_noise/  : same, Gaussian-noise infill (unseen by the sufficiency/comprehensiveness regulariser)  -> faithfulness that is not circular
# eval_free/   : free-text generation (main, plain_vlm only)                                          -> coverage of the unconditioned decoder
# eval_mean|blur*/ : further removal-operator checks (main, plain_vlm only)
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export HF_HOME=$PROJ/hf_cache
source $PROJ/miniconda3/bin/activate evialign
ulimit -n 65535 2>/dev/null; cd $PROJ/evialign-cxr && mkdir -p logs results
F="--decode forced --force_spec 0.9"
ev () { # run subdir [extra args] ; skipped when summary.json exists
  local r=$1 d=runs/$1/$2; shift 2
  [ -f runs/$r/best.pt ] && [ ! -f $d/summary.json ] && \
    CUDA_VISIBLE_DEVICES=${GPU:-0} python evaluate.py --ckpt runs/$r/best.pt --sets configs/eval_sets.yaml --out $d "$@" > logs/eval_${r}_$2.log 2>&1 || true
}
# order: matched comparisons first, so partial results are already interpretable
RUNS="s_main s_no_align s_no_sc s_plain s_no_inst s_no_anat s_aggregate_align s_bidirectional s_kl_ba s_joint_encoder s_crossattn_gpt2 s_biomedclip_backbone s_main_s1 s_main_s2 s_plain_s1 s_plain_s2"
for r in main plain_vlm $RUNS; do
  ev $r eval $F --del_ins
  ev $r eval_noise $F --infill noise --del_frac 0.15
done
ev main eval_ig $F --del_ins --ig
ev classifier_only eval --del_ins --ig
ev s_classifier_only eval
for r in main plain_vlm; do
  ev $r eval_free --del_ins
  ev $r eval_mean $F --infill mean --del_frac 0.15
  for f in 0.05 0.30; do ev $r eval_blur_$f $F --del_frac $f; done
done
[ -f runs/classifier_only/best.pt ] && [ ! -f runs/classifier_only/eval_template/summary.json ] && \
  CUDA_VISIBLE_DEVICES=${GPU:-0} python evaluate.py --ckpt runs/classifier_only/best.pt --sets configs/eval_sets.yaml --out runs/classifier_only/eval_template --template_text > logs/eval_classifier_only_template.log 2>&1 || true
ALL="runs/main runs/plain_vlm runs/classifier_only"; for r in $RUNS s_classifier_only; do [ -f runs/$r/eval/summary.json ] && ALL="$ALL runs/$r"; done
python tools/make_tables.py --runs $ALL --out results/tables
python tools/make_tables.py --runs $ALL --eval_subdir eval_noise --out results/tables_noise || true
[ -f runs/classifier_only/eval_template/summary.json ] && python tools/make_tables.py --runs runs/classifier_only --eval_subdir eval_template --out results/tables_template
for r in main plain_vlm $RUNS; do [ -f runs/$r/eval/per_image_nih_test.csv ] && python tools/text_by_finding.py --run runs/$r --set nih_test --manifest manifests/nih.csv --out results/text_by_finding_$r.csv > /dev/null 2>&1 || true; done
python tools/make_figures.py --tables results/tables --main_run runs/main --sets_yaml configs/eval_sets.yaml --out results/figures
