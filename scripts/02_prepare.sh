#!/usr/bin/env bash
# Manifests for every corpus whose data directory exists; re-run after downloading more corpora.
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export DATA=$PROJ/data; export HF_HOME=$PROJ/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p manifests results
have () { [ -d "$DATA/$1" ] && [ -n "$(ls -A $DATA/$1 2>/dev/null)" ] || { echo "skip $1 (not downloaded)"; return 1; }; }

# training corpus: adults with one frontal image per study, CheXbert labels, view probe -> manifest;
# then lung/heart masks for the kept images, then the mask paths are attached to the same manifest
if [ ! -f manifests/rexgradient.csv ]; then
  python -m evialign.prepare.rexgradient --root $DATA/rexgradient --img_root $DATA/rexgradient_img \
         --out manifests/rexgradient_nomask.csv --chexbert
  python tools/make_anatomy_masks.py --manifest manifests/rexgradient_nomask.csv --root_in $DATA/rexgradient_img --out_dir $DATA/rexgradient_anat
  python - <<PY
import os, pandas as pd
df = pd.read_csv("manifests/rexgradient_nomask.csv", low_memory=False)
root, anat = "$DATA/rexgradient_img", "$DATA/rexgradient_anat"
df["anat_mask"] = df["image_path"].map(lambda p: os.path.join(anat, os.path.relpath(p, root)).rsplit(".", 1)[0] + ".npz")
ok = df["anat_mask"].map(os.path.exists)
print(f"anatomy masks found for {ok.mean():.4f} of {len(df)} images")
df.loc[~ok, "anat_mask"] = ""
df.to_csv("manifests/rexgradient.csv", index=False)
PY
  python tools/labeler_agreement.py --manifest manifests/rexgradient.csv --out results/labeler_agreement.csv
fi

# external corpora (each is skipped until its data are in place)
have nih        && python -m evialign.prepare.nih --root $DATA/nih --out manifests/nih.csv
have chestxdet  && python -m evialign.prepare.external chestxdet  --root $DATA/chestxdet  --out manifests/chestxdet.csv
have iu         && python -m evialign.prepare.external iu --root $DATA/iu --out manifests/iu.csv --chexbert --view_filter
have shenzhen   && python -m evialign.prepare.external shenzhen   --root $DATA/shenzhen   --out manifests/shenzhen.csv
have montgomery && python -m evialign.prepare.external montgomery --root $DATA/montgomery --out manifests/montgomery.csv
have TBX11K     && python -m evialign.prepare.external tbx11k     --root $DATA/TBX11K     --out manifests/tbx11k.csv
if have rsna; then
  [ -d $DATA/rsna/dicom ] || python3 -m zipfile -e $DATA/rsna/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip $DATA/rsna/dicom
  python -m evialign.prepare.external rsna --root $DATA/rsna --out manifests/rsna.csv
fi
ls -la manifests
