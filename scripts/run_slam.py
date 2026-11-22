"""Run IBD-SLAM on one RGB-D sequence.

python scripts/run_slam.py --seq replica/room0 --ckpt pretrained/ibdslam_scannet.pth --out outputs/replica_room0

--seq   replica/<scene> | tum/<rgbd_dataset_...> | scannet/<sceneXXXX_XX>
--ckpt  IBD-SLAM checkpoint, or `ibrnet` for IBRNet weights with an untrained (uniform) xyz head
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ibdslam.datasets import make_sequence  # noqa: E402
from ibdslam.net import IBDModel  # noqa: E402
from ibdslam.paths import IBRNET_CKPT  # noqa: E402
from ibdslam.slam import IBDSLAM  # noqa: E402

TRACK_PIXELS = {'replica': 200, 'scannet': 1000, 'tum': 2000}
MAX_DEPTH = {'replica': 10., 'scannet': 6., 'tum': 6.}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', required=True)
    p.add_argument('--ckpt', default='pretrained/ibdslam_scannet.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--no_geo', action='store_true', help='model without geometric cue features')
    p.add_argument('--n_refs', type=int, default=10)
    p.add_argument('--ref_stride', type=int, default=1)
    # tracking
    p.add_argument('--n_match_refs', type=int, default=3, help='references matched against the current frame')
    p.add_argument('--n_track_pixels', type=int, default=-1, help='top-Mt matches (default: per dataset)')
    p.add_argument('--sp_max_dim', type=int, default=1200, help='max image side for SuperPoint')
    p.add_argument('--track_init', default='kabsch', choices=['const_vel', 'kabsch'])
    p.add_argument('--track_iters', type=int, default=0, help='Eq. 6 refinement iterations (0 = off)')
    p.add_argument('--track_opt', default='gn', choices=['gn', 'adam'])
    p.add_argument('--track_lr', type=float, default=2e-3)
    p.add_argument('--track_robust', type=float, default=0.05, help='Huber threshold (m)')
    p.add_argument('--track_only', action='store_true', help='skip mapping')
    # mapping
    p.add_argument('--n_samples', type=int, default=64)
    p.add_argument('--n_importance', type=int, default=64)
    p.add_argument('--map_every', type=int, default=10)
    p.add_argument('--map_stride', type=int, default=5)
    p.add_argument('--n_novel', type=int, default=1, help='perturbed novel views per mapped frame')
    p.add_argument('--novel_rot_deg', type=float, default=3.)
    p.add_argument('--novel_trans', type=float, default=0.05)
    p.add_argument('--nvs_every', type=int, default=100, help='full-image PSNR/SSIM every N frames')
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--max_frames', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    # TF32 matmul is not precise enough for pose / projection math
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    kind = a.seq.split('/')[0]
    cfg = dict(vars(a))
    cfg['n_track_pixels'] = a.n_track_pixels if a.n_track_pixels > 0 else TRACK_PIXELS[kind]
    cfg['max_depth'] = MAX_DEPTH[kind]

    model = IBDModel(a.n_samples, a.n_importance, use_geo=not a.no_geo).cuda().eval()
    if a.ckpt == 'ibrnet':
        model.load_ibrnet(IBRNET_CKPT)
    else:
        model.load(a.ckpt)
    for q in model.parameters():
        q.requires_grad_(False)

    res = IBDSLAM(model, make_sequence(a.seq), cfg, a.out).run()
    print(json.dumps({k: v for k, v in res.items() if k != 'cfg'}, indent=1))


if __name__ == '__main__':
    main()
