"""RGB-D sequence readers with a common interface:
seq.K (3x3 np), seq.H, seq.W, len(seq), seq.load(i) -> (rgb float32 [H,W,3] in [0,1], depth float32 [H,W] m), seq.gt_c2w [N,4,4]
"""
import glob
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .paths import DATA_ROOT, SCANNET_ROOT


class Sequence:
    crop = 0

    def __len__(self):
        return len(self.rgb_files)

    def _read(self, i):
        rgb = cv2.cvtColor(cv2.imread(self.rgb_files[i]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / self.depth_scale
        return rgb, depth

    def load(self, i):
        rgb, depth = self._read(i)
        if hasattr(self, 'undist_maps'):
            rgb = cv2.remap(rgb, *self.undist_maps, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, *self.undist_maps, cv2.INTER_NEAREST)
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = self.crop
        if c > 0:
            rgb, depth = rgb[c:-c, c:-c], depth[c:-c, c:-c]
        return rgb.astype(np.float32) / 255., depth

    def _finalize(self, K, H, W):
        c = self.crop
        K = K.copy()
        K[0, 2] -= c
        K[1, 2] -= c
        self.K, self.H, self.W = K, H - 2 * c, W - 2 * c


class Replica(Sequence):
    """NICE-SLAM release of Replica (results/frame*.jpg, results/depth*.png, traj.txt, cam_params.json)."""

    def __init__(self, scene, root=os.path.join(DATA_ROOT, 'Replica')):
        self.name = scene
        d = os.path.join(root, scene)
        self.rgb_files = sorted(glob.glob(os.path.join(d, 'results', 'frame*.jpg')))
        self.depth_files = sorted(glob.glob(os.path.join(d, 'results', 'depth*.png')))
        self.gt_c2w = np.loadtxt(os.path.join(d, 'traj.txt')).reshape(-1, 4, 4).astype(np.float32)
        cp = json.load(open(os.path.join(root, 'cam_params.json')))['camera']
        self.depth_scale = cp['scale']
        K = np.array([[cp['fx'], 0, cp['cx']], [0, cp['fy'], cp['cy']], [0, 0, 1]], np.float32)
        self._finalize(K, cp['h'], cp['w'])
        self.mesh_gt = os.path.join(DATA_ROOT, 'cull_replica_mesh', f'{scene}.ply')


TUM_INTR = {  # fx fy cx cy, distortion (NICE-SLAM configs)
    'freiburg1': ([517.3, 516.5, 318.6, 255.3], [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    'freiburg2': ([520.9, 521.0, 325.1, 249.7], [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    'freiburg3': ([535.4, 539.2, 320.1, 247.6], None),
}


def _read_tum_list(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    return np.array([float(r[0]) for r in rows]), [r[1:] for r in rows]


class TUM(Sequence):
    crop = 8
    depth_scale = 5000.

    def __init__(self, seq, root=os.path.join(DATA_ROOT, 'TUM'), max_dt=0.08):
        self.name = seq
        d = os.path.join(root, seq)
        t_rgb, rgb = _read_tum_list(os.path.join(d, 'rgb.txt'))
        t_dep, dep = _read_tum_list(os.path.join(d, 'depth.txt'))
        t_gt, gt = _read_tum_list(os.path.join(d, 'groundtruth.txt'))
        gt = np.array(gt, np.float64)
        self.rgb_files, self.depth_files, poses = [], [], []
        for i, t in enumerate(t_rgb):  # NICE-SLAM style association
            j = np.argmin(np.abs(t_dep - t))
            k = np.argmin(np.abs(t_gt - t))
            if abs(t_dep[j] - t) < max_dt and abs(t_gt[k] - t) < max_dt:
                self.rgb_files.append(os.path.join(d, rgb[i][0]))
                self.depth_files.append(os.path.join(d, dep[j][0]))
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(gt[k, 3:7]).as_matrix()
                T[:3, 3] = gt[k, :3]
                poses.append(T)
        poses = np.stack(poses)
        self.gt_c2w = (np.linalg.inv(poses[0]) @ poses).astype(np.float32)  # first frame at identity
        cam = [c for c in TUM_INTR if c in seq][0]
        (fx, fy, cx, cy), dist = TUM_INTR[cam]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        if dist is not None:
            self.undist_maps = cv2.initUndistortRectifyMap(K, np.array(dist), None, K, (640, 480), cv2.CV_32FC1)
        self._finalize(K, 480, 640)


class ScanNet(Sequence):
    """Preprocessed ScanNet: <root>/<split>/<scene>/{color/<id>.jpg, depth/<id>.png (mm), cam/<id>.npz (pose c2w, intrinsics)}."""
    crop = 10
    depth_scale = 1000.

    def __init__(self, scene, root=SCANNET_ROOT, split='scans_train'):
        self.name = scene
        d = os.path.join(root, split, scene)
        ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, 'color')))
        self.ids = ids
        self.rgb_files = [os.path.join(d, 'color', f'{i}.jpg') for i in ids]
        self.depth_files = [os.path.join(d, 'depth', f'{i}.png') for i in ids]
        cams = [np.load(os.path.join(d, 'cam', f'{i}.npz')) for i in ids]
        self.gt_c2w = np.stack([c['pose'] for c in cams]).astype(np.float32)
        K = cams[0]['intrinsics'].astype(np.float32)
        h, w = cv2.imread(self.rgb_files[0]).shape[:2]
        self._finalize(K, h, w)


def make_sequence(name):
    """name like 'replica/room0', 'tum/rgbd_dataset_freiburg1_desk', 'scannet/scene0000_00'"""
    kind, scene = name.split('/', 1)
    return {'replica': Replica, 'tum': TUM, 'scannet': ScanNet}[kind](scene)
