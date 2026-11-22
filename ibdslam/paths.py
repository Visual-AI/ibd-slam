"""Default locations of datasets, pretrained weights and third-party code (override with env vars)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('IBDSLAM_DATA', os.path.join(ROOT, 'data'))
SCANNET_ROOT = os.environ.get('IBDSLAM_SCANNET', os.path.join(DATA_ROOT, 'scannet'))
PRETRAINED = os.environ.get('IBDSLAM_PRETRAINED', os.path.join(ROOT, 'pretrained'))
THIRD_PARTY = os.path.join(ROOT, 'third_party')

IBRNET_CKPT = os.path.join(PRETRAINED, 'model_255000.pth')
GMFLOW_CKPT = os.path.join(PRETRAINED, 'gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')


def add_third_party(name):
    path = os.path.join(THIRD_PARTY, name)
    if not os.path.isdir(path):
        raise ImportError(f'{path} not found, run `bash setup.sh` first')
    if path not in sys.path:
        sys.path.insert(0, path)
