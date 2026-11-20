#!/usr/bin/env bash
# Download the three tuberculosis corpora and build their manifests. Run on the server, inside tmux:
#   tmux new -s tb 'bash scripts/01b_download_tb.sh 2>&1 | tee logs/tb_download.log'
# Shenzhen and Montgomery: NLM public tree (fully open). TBX11K: official Google Drive release of the authors (CC BY 4.0).
set -u
export PROJ=${PROJ:-/data/pduttapramanik/xai_vlm_cxr}
source $PROJ/miniconda3/bin/activate evialign
cd $PROJ/evialign-cxr && mkdir -p logs manifests
DATA=$PROJ/data; mkdir -p $DATA/shenzhen $DATA/montgomery $DATA/TBX11K
NLM=https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets

unzip_to () { python -m zipfile -e "$1" "$2"; }   # the server has no unzip binary
fetch_tree () { # url dest
  timeout 600 wget -nv -r -np -nH --cut-dirs=3 -e robots=off -R "index.html*" -A png,txt -P "$2" "$1" 2>&1 | tail -3 || true
}
n_shen () { find $DATA/shenzhen -name 'CHNCXR_*.png' | wc -l; }
n_mont () { find $DATA/montgomery -name 'MCUCXR_*.png' -not -ipath '*mask*' | wc -l; }   # the zip also holds lung masks with the same file names
OPENI=https://openi.nlm.nih.gov/imgs/collections
if [ "$(n_shen)" -lt 662 ]; then
  echo "Shenzhen: 662 files from the NLM tree (explicit list)"
  rm -f $DATA/shenzhen/NLM-ChinaCXRSet.zip
  mkdir -p $DATA/shenzhen/CXR_png
  sed "s#^#$NLM/Shenzhen-Hospital-CXR-Set/CXR_png/#" scripts/shenzhen_files.txt > $DATA/shenzhen/urls.txt
  timeout 1800 wget -nv -c -P $DATA/shenzhen/CXR_png -i $DATA/shenzhen/urls.txt 2>&1 | tail -3
fi
if [ "$(n_mont)" -lt 138 ]; then
  echo "Montgomery: openi zip"
  [ -s $DATA/montgomery/NLM-MontgomeryCXRSet.zip ] || wget -nv -O $DATA/montgomery/NLM-MontgomeryCXRSet.zip $OPENI/NLM-MontgomeryCXRSet.zip 2>&1 | tail -2
  [ -s $DATA/montgomery/NLM-MontgomeryCXRSet.zip ] && unzip_to $DATA/montgomery/NLM-MontgomeryCXRSet.zip $DATA/montgomery
fi
if [ "$(n_mont)" -lt 138 ]; then echo "Montgomery: NLM tree"; fetch_tree $NLM/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/ $DATA/montgomery; fi
if ! find $DATA/TBX11K -name 'TBX11K_val.json' | grep -q .; then
  echo "TBX11K: Google Drive"
  pip install -q gdown 2>&1 | tail -1
  [ -s $DATA/TBX11K/TBX11K.zip ] || gdown "https://drive.google.com/uc?id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR" -O $DATA/TBX11K/TBX11K.zip
  [ -s $DATA/TBX11K/TBX11K.zip ] && unzip_to $DATA/TBX11K/TBX11K.zip $DATA/TBX11K
fi
echo "counts: shenzhen $(find $DATA/shenzhen -name 'CHNCXR_*.png' | wc -l) (expect 662), montgomery $(find $DATA/montgomery -name 'MCUCXR_*.png' | wc -l) (expect 138), TBX11K val json: $(find $DATA/TBX11K -name 'TBX11K_val.json' | head -1)"
[ "$(n_shen)" -gt 0 ] && python -m evialign.prepare.external shenzhen   --root $DATA/shenzhen   --out manifests/shenzhen.csv
[ "$(n_mont)" -gt 0 ] && python -m evialign.prepare.external montgomery --root $DATA/montgomery --out manifests/montgomery.csv
TBROOT=$(dirname "$(find $DATA/TBX11K -type d -name imgs | head -1)")
echo "TBX11K root: $TBROOT"; ls "$TBROOT" "$TBROOT/imgs" 2>/dev/null | head -12
[ -n "$TBROOT" ] && python -m evialign.prepare.external tbx11k     --root $TBROOT          --out manifests/tbx11k.csv
python - <<'PY'
import pandas as pd
for n in ("shenzhen", "montgomery", "tbx11k"):
    try:
        d = pd.read_csv(f"manifests/{n}.csv")
        print(n, len(d), "abnormal", int(d["y_abnormal"].sum()), "with boxes", int((d.get("boxes", pd.Series([""] * len(d))).fillna("").str.len() > 0).sum()))
    except Exception as e:
        print(n, "FAILED", e)
PY
