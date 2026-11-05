# EviAlign-CXR

Code for *Evidence-aligned vision–language explanations for chest radiograph diagnosis*.
A multi-label classifier with an exactly decomposable evidence head is trained jointly with an explanation
decoder (LLaVA-style prefix LLM or GPT-2 cross-attention decoder). For every finding asserted in the reference
text, the decoder's token-to-patch grounding is aligned with the classifier's class-specific evidence map
(stop-gradient teacher, Jensen–Shannon), an instance-level contrastive term blocks location-prior shortcuts,
an anatomical prior keeps evidence inside plausible anatomy, and a corrected sufficiency/comprehensiveness
regulariser ties the evidence to the classifier's causal response. Evaluation adds counterfactual faithfulness
tests (text-guided deletion, explanation counterfactual sensitivity) that are not optimised during training.

## Layout
```
evialign/            findings vocabulary, labeler, data, models (encoder, evidence head, decoders), losses, objective, metrics
evialign/prepare/    corpus adapters -> manifests (ReXGradient-160K, NIH, ChestX-Det, IU, Shenzhen, Montgomery, TBX11K, RSNA, VinDr)
configs/             main.yaml, open_only.yaml, ablations/*.yaml, eval_sets.yaml, smoke.yaml
scripts/             00_setup_env.sh  01_download_data.sh  02_prepare.sh  03_train_main.sh  03_train_ablations.sh  04_eval_all.sh  05_dry_run.sh
tools/               anatomy masks, view filter, labeler QC, tables, figures
baselines/           zero-shot open VLM report generation
tests/               synthetic data, tiny tokenizer, controlled synthetic study (CPU)
```

## Server workflow (2 x H200)
```bash
export PROJ=/data/pduttapramanik/xai_vlm_cxr
cd $PROJ && git clone <repo> evialign-cxr      # or unzip the archive here
bash evialign-cxr/scripts/00_setup_env.sh
tmux new -s data   'bash evialign-cxr/scripts/01_download_data.sh'   # some corpora need a browser step (see comments)
tmux new -s prep   'bash evialign-cxr/scripts/02_prepare.sh'
tmux new -s main   'GPU=0 bash evialign-cxr/scripts/03_train_main.sh'       # full protocol: main, plain VLM, classifier-only (about 9 h each for the LLM models)
tmux new -s abl    'GPU=1 bash evialign-cxr/scripts/03_train_ablations.sh'   # reduced protocol (30k studies, 6 epochs): seeds and ablations
tmux new -s eval   'bash evialign-cxr/scripts/04_eval_all.sh'        # tables in results/tables, SVG figures in results/figures
```
Send back `results/tables/*.csv`, `results/tables/tables.md`, `results/figures/*.svg`, `results/labeler_agreement.csv`,
and `runs/*/log.jsonl`; the manuscript tables are filled from these.

## Data licences (verified November 2026)
| Corpus | Role | Access |
|---|---|---|
| ReXGradient-160K | training / development / internal test | Hugging Face, auto-approved click-through, non-commercial |
| NIH ChestX-ray14 (+ BBox_List_2017) | external, localisation | fully open |
| ChestX-Det | external, localisation (NIH subset, 3 radiologists) | fully open, Apache-2.0 |
| RSNA Pneumonia | external, pneumonia boxes (NIH-derived) | open with attribution |
| IU X-ray (Open-i) | external, reports | fully open, CC BY-NC-ND 4.0 |
| Shenzhen, Montgomery | external, TB (abnormality) | fully open |
| TBX11K (val) | external, TB boxes | CC BY 4.0 |
| VinDr-CXR (Kaggle) | optional external, boxes | click-through, non-commercial |

`configs/open_only.yaml` trains on NIH with label-derived explanation targets if a click-through licence is not acceptable.
Backbones: DINOv2 (natural images) by default. RAD-DINO is **not** used: its pretraining includes all NIH and PadChest images.

## CPU smoke test
```bash
python tests/make_synthetic.py --out /tmp/evialign_smoke/data --n 480 --size 64
python train.py --config configs/smoke.yaml
python evaluate.py --ckpt /tmp/evialign_smoke/run/best.pt --sets <sets.yaml> --out /tmp/evialign_smoke/eval --workers 0
bash tests/synthetic_study/run_study.sh     # controlled study with a spurious marker, 3 seeds x 4 variants
```
