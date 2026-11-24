# EviAlign-CXR

Code for *Aligning Vision-Language Grounding with Classifier Evidence for Chest Radiograph Diagnosis*.
A multi-label classifier with an exactly decomposable evidence head is trained jointly with an explanation
decoder (prefix LLM or GPT-2 cross-attention decoder). For every finding asserted in the reference text, the
decoder's token-to-patch grounding is aligned with the classifier's class-specific evidence map (stop-gradient
teacher, Jensen–Shannon). An instance-level contrastive term, an anatomical prior and a
sufficiency/comprehensiveness regulariser complete the objective. Evaluation adds counterfactual tests
(text-guided deletion, explanation counterfactual sensitivity) that are not optimised during training.

## Layout
```
evialign/            findings vocabulary, labeler, data, models, losses, objective, metrics
evialign/prepare/    corpus adapters that write the manifests
configs/             main.yaml, ablations/, evaluation set lists (eval_*.yaml), smoke.yaml (synthetic study and CPU test)
scripts/             00-02 environment, data and manifests; 03 training; 04 evaluation; 06-12 further evaluations and analyses
tools/               anatomy masks, view filter, labeler checks, tables, figures and summaries of the evaluation passes
baselines/           zero-shot open VLM report generation
tests/               synthetic data, tiny tokenizer and the controlled synthetic study (CPU)
results/             analysis/ holds the files behind the supplementary tables, synthetic_study/ the controlled study
reproducibility/     map from each table to its runs, scripts and tools (table_to_run.csv) and software versions
```

## Workflow
```bash
export PROJ=/path/to/project
cd $PROJ && git clone https://github.com/Dr-PKDP/EviAlign-CXR.git evialign-cxr
bash evialign-cxr/scripts/00_setup_env.sh
bash evialign-cxr/scripts/01_download_data.sh     # some corpora need a browser step (see comments)
bash evialign-cxr/scripts/02_prepare.sh
GPU=0 bash evialign-cxr/scripts/03_train_main.sh        # full protocol: main, plain VLM, classifier-only
GPU=1 bash evialign-cxr/scripts/03_train_ablations.sh   # reduced protocol (30,000 studies, 6 epochs): ablations; 03_train_seeds.sh adds seeds
bash evialign-cxr/scripts/04_eval_all.sh
```
Scripts 06 to 12 run the remaining evaluations (tuberculosis, randomisation check, operators and decoder probe, uncapped counterfactuals,
calibration and Integrated Gradients, label-policy metrics and multi-operator passes, full-validation-threshold and patient-random passes).
Script 13 trains the controls for side conditions of the alignment objective (separate gradient clipping of the classifier and decoder
branches, a minimal objective, and full-protocol seed replicates with `FULL=1`); script 14 evaluates them and
`tools/limitation_runs_summary.py` tabulates them. `tools/sentence_location_checks.py` checks laterality and vertical position of the generated
sentences against the NIH boxes and compares single- with multi-finding sentences. `tools/adjudication_package.py` builds a blinded case set
with mismatched controls for human or model raters and `tools/adjudication_analysis.py` summarises the completed sheets.
The fixed evaluation subsets are the first 100 batches of 32 images of each test list in manifest order (`--max_batches 100`);
their thresholds and generation gate are fitted on the first 3,200 validation images, whereas the full-test passes and script 12 use the
full validation split. Random-removal masks are keyed to (image, finding, replicate) with `--rand_key`, which the primary subset passes do not use.
`tools/operator_summary.py`, `tools/threshold_and_sample_summary.py` and `tools/stage_decomposition.py` (`--runs runs --out results/analysis`)
tabulate the passes of scripts 09 to 12; `reproducibility/table_to_run.csv` lists the source of every table.

## Data access
| Corpus | Role | Access |
|---|---|---|
| ReXGradient-160K | training / development / internal test | Hugging Face, auto-approved click-through, non-commercial |
| NIH ChestX-ray14 (+ BBox_List_2017) | external, localisation | fully open |
| ChestX-Det10 | external, localisation (NIH subset, 3 radiologists) | fully open, Apache-2.0 |
| RSNA Pneumonia | external, pneumonia boxes (NIH-derived) | open with attribution |
| IU X-ray (Open-i) | external, reports | fully open, CC BY-NC-ND 4.0 |
| Shenzhen, Montgomery | external, TB (abnormality) | fully open |
| TBX11K (val) | external, TB boxes | CC BY 4.0 |

Backbones: DINOv2 (natural images) by default. RAD-DINO is **not** used: its pretraining includes all NIH and PadChest images.

## CPU smoke test
```bash
python tests/make_synthetic.py --out /tmp/evialign_smoke/data --n 480 --size 64
python train.py --config configs/smoke.yaml
python evaluate.py --ckpt /tmp/evialign_smoke/run/best.pt --sets <sets.yaml> --out /tmp/evialign_smoke/eval --workers 0
bash tests/synthetic_study/run_study.sh     # controlled study with a spurious marker, 3 seeds x 4 variants
```
