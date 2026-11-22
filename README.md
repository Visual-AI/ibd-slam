# IBD-SLAM: Learning Image-Based Depth Fusion for Generalizable SLAM

Minghao Yin, Shangzhe Wu, Kai Han — CVPR 2024

IBD-SLAM extends [IBRNet](https://github.com/googleinterns/IBRNet) with an xyz blending branch that fuses the
depth maps of neighbouring reference views into an xyz-map of a novel view. The model is trained once and then
reconstructs unseen RGB-D sequences feed-forward, without per-scene optimisation.

## Installation

```bash
conda create -n ibdslam python=3.10 -y && conda activate ibdslam
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash setup.sh          # third-party code (IBRNet, SuperGlue, UniMatch) + GMFlow weights
```

Pretrained weights go to `pretrained/`:

| file | source |
|---|---|
| `ibdslam_scannet.pth` | IBD-SLAM model trained on ScanNet, [download](https://github.com/mhyqwe/ibd-slam/releases/download/v1.0/ibdslam_scannet.pth) |
| `model_255000.pth` | IBRNet weights, from `pretrained_model.zip` of the [IBRNet repository](https://github.com/googleinterns/IBRNet#evaluation); used to initialise training |
| `gmflow-...-4e7b215d.pth` | downloaded by `setup.sh`; only needed for training (optical-flow loss) |

`open3d` needs `libEGL` / `libGL` (`apt install libegl1 libgl1`) for the mesh evaluation.

## Data

Default locations are `data/` and `pretrained/` inside the repository; override them with
`IBDSLAM_DATA`, `IBDSLAM_SCANNET` and `IBDSLAM_PRETRAINED`.

```
data/
├── Replica/{room0,...,office4}, cam_params.json   # NICE-SLAM release
├── cull_replica_mesh/{room0,...}.ply               # GT meshes for 3D metrics
└── scannet/scans_train/<scene>/                    # training only
    ├── color/<id>.jpg
    ├── depth/<id>.png                              # uint16, millimetres
    └── cam/<id>.npz                                # 'pose' (4x4 camera-to-world), 'intrinsics' (3x3)
```

Replica and the culled meshes: `bash scripts/download_replica.sh`.
TUM RGB-D (`tum/rgbd_dataset_*`) and ScanNet (`scannet/sceneXXXX_XX`) sequences are also supported by `run_slam.py`.

## Evaluation on Replica

```bash
bash scripts/eval_replica.sh pretrained/ibdslam_scannet.pth outputs/replica
```

This runs SLAM on the 8 Replica scenes (spread over all visible GPUs), builds a Poisson mesh from the rendered
xyz-maps, and writes a table to `outputs/replica/summary.md`. A single scene:

```bash
python scripts/run_slam.py --seq replica/room0 --ckpt pretrained/ibdslam_scannet.pth --out outputs/replica/replica_room0
python scripts/eval_recon.py --run outputs/replica/replica_room0 --scene room0
```

`run_slam.py` writes the estimated trajectory (`est_c2w.npy`), the fused point cloud (`points.npz`), a few
novel-view renderings and `slam_result.json` (ATE RMSE, Depth L1, PSNR, SSIM, timings).
`eval_recon.py` writes `mesh_d8.ply` and `recon_result_d8.json`.

- Depth L1 compares the depth of the rendered xyz-map with the sensor depth at every mapped frame.
- Acc / Comp / Comp. Ratio use 200k surface samples on the predicted and culled GT meshes (NICE-SLAM protocol).
  The `*_vis` variants additionally restrict both sample sets to points observed in at least one GT depth map,
  since the culled GT meshes still contain surfaces that are occluded in every frame.
- Use `--ckpt ibrnet` for a baseline with IBRNet weights and an untrained (uniform) xyz head.

## Tracking

Each frame is matched against 3 of its 10 reference frames with SuperPoint + SuperGlue (full resolution);
the top-Mt matches (Replica 200, ScanNet 1000, TUM 2000) link pixels of the current frame to reference xyz values.
By default the pose is the RANSAC rigid alignment between these xyz values and the back-projected depth
of the current frame. The rendered-xyz objective of the paper (Eq. 6) can be added as a refinement:

```bash
python scripts/run_slam.py ... --track_iters 15 --track_opt gn   # robust Levenberg-Marquardt
python scripts/run_slam.py ... --track_iters 15 --track_opt adam
```

## Training

Training uses ScanNet with sensor depth for the depth / normal / xyz terms and GMFlow for the flow term;
the 4 ScanNet evaluation scenes (0000, 0059, 0106, 0169) are excluded. The released model was trained for 60k
steps with 8 GPUs (global batch 8, about 5 hours on H20):

```bash
torchrun --standalone --nproc-per-node 8 scripts/train.py --run_dir runs/ibd_scannet
```

The run directory gets `checkpoints/{latest,step_XXXXXX,final}.pth`, TensorBoard logs and `log.jsonl`;
re-running the same command resumes from `latest.pth`. Validation renders Replica frames (room0 and office0,
frame 500) from their predecessors with GT poses (`--val`).

## Acknowledgements

This code builds on [IBRNet](https://github.com/googleinterns/IBRNet),
[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) and
[UniMatch](https://github.com/autonomousvision/unimatch), and uses the Replica release and evaluation meshes of
[NICE-SLAM](https://github.com/cvg/nice-slam). SuperGlue is licensed for non-commercial research use only.

## Citation

```bibtex
@inproceedings{yin2024ibdslam,
  title     = {IBD-SLAM: Learning Image-Based Depth Fusion for Generalizable SLAM},
  author    = {Yin, Minghao and Wu, Shangzhe and Han, Kai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```
