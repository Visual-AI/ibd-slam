"""Poisson mesh from the rendered xyz points of a run_slam.py output + 3D metrics on Replica.

python scripts/eval_recon.py --run outputs/replica_room0 --scene room0

Writes <run>/mesh_<tag>.ply and <run>/recon_result_<tag>.json with Acc / Comp [cm] and Comp. ratio [% < 5cm].
acc_cm / comp_cm / comp_ratio: 200k surface samples against cull_replica_mesh (NICE-SLAM protocol).
*_vis: both sample sets restricted to points observed in at least one GT depth map, since
cull_replica_mesh still contains surfaces that are occluded in every frame.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ibdslam.datasets import Replica  # noqa: E402


def build_mesh(pts, cols, dirs, voxel, depth, density_q, trim):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(cols)
    pc.normals = o3d.utility.Vector3dVector(dirs)  # carry view directions through filtering
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc = pc.voxel_down_sample(voxel)
    view = np.asarray(pc.normals).copy()
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    n = np.asarray(pc.normals)
    flip = (n * view).sum(1) > 0  # normals must face the camera that observed the point
    n[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))
    if trim > 0:  # drop Poisson extrapolation far from any input point
        dist, _ = cKDTree(np.asarray(pc.points)).query(np.asarray(mesh.vertices))
        mesh.remove_vertices_by_mask(dist > trim)
    return mesh, pc


def frustum_cull(mesh, c2ws, K, H, W, every=10):
    v = np.asarray(mesh.vertices)
    seen = np.zeros(len(v), bool)
    for c2w in c2ws[::every]:
        w2c = np.linalg.inv(c2w)
        pc = v @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        uv = pc @ K.T
        uv = uv[:, :2] / np.maximum(z[:, None], 1e-6)
        seen |= (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mesh.remove_vertices_by_mask(~seen)
    return mesh


def visible(pts, seq, every=5, eps=0.03):
    """Points observed (not occluded) in at least one GT depth map."""
    vis = np.zeros(len(pts), bool)
    K = seq.K
    for i in range(0, len(seq), every):
        depth = np.asarray(seq.load(i)[1])
        w2c = np.linalg.inv(seq.gt_c2w[i])
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = np.round(pc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(pc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]).astype(int)
        ok = (z > 0.05) & (u >= 0) & (u < seq.W) & (v >= 0) & (v < seq.H)
        d = np.zeros(len(pts))
        d[ok] = depth[v[ok], u[ok]]
        vis |= ok & (d > 0) & (z < d + eps)
    return vis


def scores(pp, pg, thr):
    acc = cKDTree(pg).query(pp)[0]
    comp = cKDTree(pp).query(pg)[0]
    return dict(acc_cm=float(acc.mean() * 100), comp_cm=float(comp.mean() * 100),
                comp_ratio=float((comp < thr).mean() * 100))


def metrics(pred_path, seq, n=200000, thr=0.05):
    pg = trimesh.sample.sample_surface(trimesh.load(seq.mesh_gt, process=False), n)[0]
    pp = trimesh.sample.sample_surface(trimesh.load(pred_path, process=False), n)[0]
    res = scores(pp, pg, thr)
    vg, vp = visible(pg, seq), visible(pp, seq)
    res.update({k + '_vis': v for k, v in scores(pp[vp], pg[vg], thr).items()})
    res['gt_vis_frac'] = float(vg.mean())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--voxel', type=float, default=0.01)
    p.add_argument('--depth', type=int, default=8, help='Poisson octree depth (supp: 8)')
    p.add_argument('--density_q', type=float, default=0.02)
    p.add_argument('--trim', type=float, default=0.05, help='max vertex-to-point distance [m], 0 = off')
    p.add_argument('--tag', default='')
    a = p.parse_args()

    d = np.load(os.path.join(a.run, 'points.npz'))
    seq = Replica(a.scene)
    mesh, pc = build_mesh(d['xyz'], d['rgb'], d['dir'], a.voxel, a.depth, a.density_q, a.trim)
    est = np.load(os.path.join(a.run, 'est_c2w.npy'))
    mesh = frustum_cull(mesh, est, seq.K, seq.H, seq.W)
    tag = a.tag or f'd{a.depth}'
    out = os.path.join(a.run, f'mesh_{tag}.ply')
    o3d.io.write_triangle_mesh(out, mesh)
    o3d.io.write_point_cloud(os.path.join(a.run, 'points_filtered.ply'), pc)
    res = dict(scene=a.scene, depth=a.depth, voxel=a.voxel, density_q=a.density_q, trim=a.trim,
               n_vertices=len(mesh.vertices), **metrics(out, seq))
    json.dump(res, open(os.path.join(a.run, f'recon_result_{tag}.json'), 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
