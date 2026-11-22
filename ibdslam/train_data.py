"""ScanNet training samples: target frame + K reference frames (IBRNet-style selection, supp. Sec. A)."""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import SCANNET_ROOT

EVAL_SCENES = ('scene0000_', 'scene0059_', 'scene0106_', 'scene0169_')  # paper's ScanNet eval scenes


def build_index(cache, root=SCANNET_ROOT, split='scans_train'):
    if os.path.exists(cache):
        return json.load(open(cache))
    scenes = {}
    for s in sorted(os.listdir(os.path.join(root, split))):
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, split, s, 'color')))
        scenes[s] = ids
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(scenes, open(cache, 'w'))
    return scenes


class ScanNetTrain(Dataset):
    def __init__(self, index_cache, n_refs=10, max_gap=2, crop=10, past_only_prob=0.5,
                 length=10 ** 7, root=SCANNET_ROOT, split='scans_train', exclude=EVAL_SCENES):
        idx = build_index(index_cache, root, split)
        self.scenes = {k: v for k, v in idx.items() if not k.startswith(exclude) and len(v) > 100}
        self.names = sorted(self.scenes)
        self.root, self.split = root, split
        self.n_refs, self.max_gap, self.crop = n_refs, max_gap, crop
        self.past_only_prob = past_only_prob
        self.length = length

    def __len__(self):
        return self.length

    def _load(self, scene, fid):
        d = os.path.join(self.root, self.split, scene)
        rgb = cv2.imread(os.path.join(d, 'color', f'{fid}.jpg'))
        dep = cv2.imread(os.path.join(d, 'depth', f'{fid}.png'), cv2.IMREAD_UNCHANGED)
        cam = np.load(os.path.join(d, 'cam', f'{fid}.npz'))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        dep = dep.astype(np.float32) / 1000.
        if dep.shape != rgb.shape[:2]:
            dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = self.crop
        rgb, dep = rgb[c:-c, c:-c], dep[c:-c, c:-c]
        K = cam['intrinsics'].astype(np.float32).copy()
        K[0, 2] -= c
        K[1, 2] -= c
        return rgb, dep, K, cam['pose'].astype(np.float32)

    def __getitem__(self, _):
        rng = random.Random()
        for _try in range(20):
            scene = rng.choice(self.names)
            ids = self.scenes[scene]
            K_ = self.n_refs
            n = rng.randint(1, 3)  # pool of n*K views, n ~ U[1,3]
            gap = rng.randint(1, self.max_gap)
            span = n * K_ * gap
            t = rng.randrange(span, len(ids) - span) if len(ids) > 2 * span + 1 else None
            if t is None:
                continue
            if rng.random() < self.past_only_prob:
                pool = list(range(t - span, t))
            else:
                pool = [i for i in range(t - span // 2 - 1, t + span // 2 + 2) if i != t]
            refs = sorted(rng.sample(pool, K_), key=lambda i: abs(i - t))
            frames = [self._load(scene, ids[i]) for i in [t] + refs]
            poses = np.stack([f[3] for f in frames])
            if not np.isfinite(poses).all():
                continue
            rgb = torch.from_numpy(np.stack([f[0] for f in frames])).permute(0, 3, 1, 2)
            dep = torch.from_numpy(np.stack([f[1] for f in frames]))[:, None]
            dep[(dep > 8) | (dep < 0.1)] = 0
            return dict(rgb=rgb, depth=dep, K=torch.from_numpy(np.stack([f[2] for f in frames])),
                        c2w=torch.from_numpy(poses), scene=scene, tid=ids[t])
        raise RuntimeError('could not sample a valid training example')
