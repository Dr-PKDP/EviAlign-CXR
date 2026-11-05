#!/usr/bin/env bash
# Data download. Run inside tmux. DATA must be under /data/pduttapramanik.
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr; export DATA=$PROJ/data; export HF_HOME=$PROJ/hf_cache
mkdir -p $DATA && cd $DATA
source $PROJ/miniconda3/bin/activate evialign

# 1) ReXGradient-160K (training). Accept the licence once on the dataset page while logged in
#    (huggingface.co/datasets/rajpurkarlab/ReXGradient-160K; approval is automatic), then:
hf auth login            # paste a read token
hf download rajpurkarlab/ReXGradient-160K --repo-type dataset --local-dir $DATA/rexgradient
# images ship as a zstd-compressed tar split in 10 parts; extract to a separate directory
mkdir -p $DATA/rexgradient_img
cat $DATA/rexgradient/deid_png.part0* | zstd -dc | tar -xf - -C $DATA/rexgradient_img

# 2) NIH ChestX-ray14 (fully open). The Box folder needs a browser; the batch_download_zips.py script that
#    NIH ships in the same folder downloads images_001..012.tar.gz. Also fetch Data_Entry_2017_v2020.csv,
#    BBox_List_2017.csv, train_val_list.txt, test_list.txt into $DATA/nih.
mkdir -p $DATA/nih
#    python batch_download_zips.py && for f in images_*.tar.gz; do tar -xzf $f -C $DATA/nih; done

# 3) ChestX-Det (fully open, Apache-2.0): github.com/Deepwise-AILab/ChestX-Det-Dataset (train/test zips + json)
git clone --depth 1 https://github.com/Deepwise-AILab/ChestX-Det-Dataset $DATA/chestxdet_repo || true
#    follow the README links for the image zips, unzip into $DATA/chestxdet/{train,test}, copy the json files there

# 4) IU X-ray / Open-i (fully open): reports + PNGs
mkdir -p $DATA/iu && cd $DATA/iu
wget -q https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz && tar -xzf NLMCXR_png.tgz
wget -q https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz && tar -xzf NLMCXR_reports.tgz
cd $DATA

# 5) Shenzhen and Montgomery (NLM, fully open)
#    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html
mkdir -p $DATA/shenzhen $DATA/montgomery

# 6) TBX11K (CC BY 4.0): https://mmcheng.net/tb/  (Google Drive link) -> $DATA/TBX11K
# 7) RSNA Pneumonia Detection Challenge (RSNA site, attribution licence) -> $DATA/rsna
# 8) CheXmask (CC BY 4.0, PhysioNet open access) only if NIH masks are wanted for the open-only config
# 9) Optional, click-through licence: VinBigData (Kaggle) -> $DATA/vinbigdata
echo "downloads staged in $DATA"
