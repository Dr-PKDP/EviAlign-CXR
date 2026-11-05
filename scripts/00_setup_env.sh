#!/usr/bin/env bash
# User-space environment (no sudo). Everything lives under /data/pduttapramanik.
set -e
export PROJ=/data/pduttapramanik/xai_vlm_cxr
mkdir -p $PROJ && cd $PROJ
if [ ! -d $PROJ/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
  bash mc.sh -b -p $PROJ/miniconda3 && rm mc.sh
fi
source $PROJ/miniconda3/bin/activate
conda create -y -n evialign python=3.11 || true
conda activate evialign
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r $PROJ/evialign-cxr/requirements.txt
export HF_HOME=$PROJ/hf_cache
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
