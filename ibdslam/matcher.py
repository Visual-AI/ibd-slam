"""SuperPoint + SuperGlue matcher with per-frame keypoint caching."""
import cv2
import numpy as np
import torch

from .paths import add_third_party

add_third_party('SuperGluePretrainedNetwork')
from models.superpoint import SuperPoint  # noqa: E402
from models.superglue import SuperGlue  # noqa: E402


class Matcher:
    def __init__(self, device='cuda', max_dim=640, max_keypoints=1024, weights='indoor'):
        self.device = device
        self.max_dim = max_dim
        self.sp = SuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005,
                              'max_keypoints': max_keypoints}).eval().to(device)
        self.sg = SuperGlue({'weights': weights, 'sinkhorn_iterations': 20,
                             'match_threshold': 0.2}).eval().to(device)
        self.cache = {}

    @torch.no_grad()
    def features(self, key, rgb):
        """rgb: float [H,W,3] numpy in [0,1]. Returns keypoints in original pixel coords."""
        if key in self.cache:
            return self.cache[key]
        H, W = rgb.shape[:2]
        s = min(1., self.max_dim / max(H, W))
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if s < 1:
            gray = cv2.resize(gray, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(gray / 255.).float()[None, None].to(self.device)
        p = self.sp({'image': img})
        f = {'keypoints': p['keypoints'][0], 'scores': p['scores'][0], 'descriptors': p['descriptors'][0],
             'image': img, 'scale': s}
        self.cache[key] = f
        if len(self.cache) > 64:
            self.cache.pop(next(iter(self.cache)))
        return f

    @torch.no_grad()
    def match(self, k0, rgb0, k1, rgb1):
        """Returns (uv0 [M,2], uv1 [M,2], conf [M]) torch tensors in original pixel coords."""
        f0, f1 = self.features(k0, rgb0), self.features(k1, rgb1)
        if len(f0['keypoints']) < 2 or len(f1['keypoints']) < 2:
            e = torch.zeros(0, 2, device=self.device)
            return e, e, torch.zeros(0, device=self.device)
        data = {'image0': f0['image'], 'image1': f1['image'],
                'keypoints0': f0['keypoints'][None], 'keypoints1': f1['keypoints'][None],
                'scores0': f0['scores'][None], 'scores1': f1['scores'][None],
                'descriptors0': f0['descriptors'][None], 'descriptors1': f1['descriptors'][None]}
        pred = self.sg(data)
        m0 = pred['matches0'][0]
        conf = pred['matching_scores0'][0]
        valid = m0 > -1
        uv0 = f0['keypoints'][valid] / f0['scale']
        uv1 = f1['keypoints'][m0[valid]] / f1['scale']
        return uv0, uv1, conf[valid]
