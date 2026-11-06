#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor checkpoints
if [ ! -d vendor/MedSAM2 ]; then git clone https://github.com/bowang-lab/MedSAM2.git vendor/MedSAM2; fi
git -C vendor/MedSAM2 checkout 332f30d420f1d1b08e2a79b3ae6a602458808383
.venv/bin/pip install hydra-core iopath tqdm
if [ ! -s checkpoints/MedSAM2_latest.pt ]; then
 curl -L --fail --retry 3 https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt -o checkpoints/MedSAM2_latest.pt.part
 mv checkpoints/MedSAM2_latest.pt.part checkpoints/MedSAM2_latest.pt
fi
