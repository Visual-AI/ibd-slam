#!/usr/bin/env bash
# Fetch third-party code (pinned commits) and the GMFlow weights used by the training flow loss.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p third_party pretrained

clone() {  # name url commit
  if [[ ! -d third_party/$1 ]]; then
    git clone "$2" "third_party/$1"
    git -C "third_party/$1" checkout -q "$3"
  fi
}
clone IBRNet https://github.com/googleinterns/IBRNet.git 41807b2bb7c598aaeec8c9e5eb144856aa6643bf
clone SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork.git ddcf11f42e7e0732a0c4607648f9448ea8d73590
clone unimatch https://github.com/autonomousvision/unimatch.git 943a99006950ab6c84ea72a72beb147fdd03c19a

GMFLOW=gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
if [[ ! -f pretrained/$GMFLOW ]]; then
  wget -q --show-progress -O "pretrained/$GMFLOW" \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/$GMFLOW"
fi

if [[ ! -f pretrained/model_255000.pth ]]; then
  echo "IBRNet weights missing: download pretrained_model.zip from the IBRNet repository"
  echo "(gdown https://drive.google.com/uc?id=1wNkZkVQGx7rFksnX7uVX3NazrbjqaIgU) and put model_255000.pth in pretrained/"
fi
