"""Train the IBD model on ScanNet.

L = L_rgb + 0.1 L_depth + 0.04 L_normal + 0.5 L_xyz + 0.5 L_xyz3d + 0.2 L_warp + 0.002 L_flow
Depth and normal targets come from the sensor depth. Data parallel with torchrun (global batch = #GPUs):

    torchrun --standalone --nproc-per-node 8 scripts/train.py --run_dir runs/ibd_scannet

Re-running the same command resumes from <run_dir>/checkpoints/latest.pth.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ibdslam.datasets import make_sequence  # noqa: E402
from ibdslam.net import IBDModel  # noqa: E402
from ibdslam.paths import IBRNET_CKPT  # noqa: E402
from ibdslam.render import (Refs, backproject_depth, get_rays, near_far_from_depths, pixel_grid,  # noqa: E402
                            pose_update, render_chunked, render_rays)
from ibdslam.train_data import ScanNetTrain, build_index  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', default='runs/ibd_scannet')
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--n_rays', type=int, default=1024)
    p.add_argument('--n_patch', type=int, default=256, help='pixel triplets for the normal loss')
    p.add_argument('--patch_d', type=int, default=6)
    p.add_argument('--n_refs', type=int, default=10)
    p.add_argument('--lr_feat', type=float, default=5e-4)
    p.add_argument('--lr_mlp', type=float, default=2.5e-4)
    p.add_argument('--lr_new', type=float, default=1e-3)
    p.add_argument('--decay_steps', type=int, default=25000)
    p.add_argument('--w_depth', type=float, default=0.1)
    p.add_argument('--w_xyz', type=float, default=0.5)
    p.add_argument('--w_xyz3d', type=float, default=0.5)
    p.add_argument('--w_normal', type=float, default=0.04)
    p.add_argument('--w_warp', type=float, default=0.2)
    p.add_argument('--w_flow', type=float, default=0.002)
    p.add_argument('--ref_pose_noise', type=float, default=0.002,
                   help='std of reference translation noise (m); rotation noise is 2 deg per cm')
    p.add_argument('--no_geo', action='store_true')
    p.add_argument('--init', default='ibrnet', help="'ibrnet', 'scratch' or a checkpoint path")
    p.add_argument('--val', default='replica/room0:500,replica/office0:500',
                   help="comma-separated '<seq>:<frame>' validation targets ('' to disable)")
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--i_print', type=int, default=50)
    p.add_argument('--i_val', type=int, default=2000)
    p.add_argument('--i_save', type=int, default=1000)
    p.add_argument('--i_keep', type=int, default=10000)
    return p.parse_args()


def save_ckpt(path, model, opt, sched, step, args):
    tmp = path + '.tmp'
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    step=step, args=vars(args)), tmp)
    os.replace(tmp, path)


def sample_pixels(H, W, n_rays, n_patch, d, device):
    u = torch.randint(0, W, (n_rays,), device=device)
    v = torch.randint(0, H, (n_rays,), device=device)
    ua = torch.randint(0, W - d, (n_patch,), device=device)
    va = torch.randint(0, H - d, (n_patch,), device=device)
    uu = torch.cat([u, ua, ua + d, ua])
    vv = torch.cat([v, va, va, va + d])
    return torch.stack([uu, vv], -1)  # long [N,2]


def project(X, K, w2c):
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]
    return uv, z


def bilinear(img, uv):
    """img [C,H,W], uv [N,2] pixel coords -> [N,C]"""
    H, W = img.shape[-2:]
    g = torch.stack([2 * uv[:, 0] / (W - 1) - 1, 2 * uv[:, 1] / (H - 1) - 1], -1)
    return F.grid_sample(img[None], g[None, None], align_corners=True)[0, :, 0].T


def safe_normal(a, b, c):
    n = torch.cross(b - a, c - a, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def compute_losses(args, out, gt, refs, tgt, flow, flow_ref):
    """out: render output for one level; gt: dict of per-ray targets."""
    L = {}
    nr = args.n_rays
    rv = out['ray_valid'].float()
    L['rgb'] = (((out['rgb'] - gt['rgb']) ** 2).sum(-1) * rv).sum() / rv.sum().clamp(min=1) / 3
    dv = (gt['depth'] > 0).float() * rv
    nd = dv.sum().clamp(min=1)
    L['depth'] = (((out['depth'] - gt['depth']) ** 2) * dv).sum() / nd
    zp = out['xyz'] @ tgt['w2c'][2, :3] + tgt['w2c'][2, 3]
    L['xyz'] = ((((zp - out['depth']) ** 2) + (zp - gt['depth']) ** 2) * dv).sum() / nd
    L['xyz3d'] = (((out['xyz'] - gt['xyz']) ** 2).sum(-1) * dv).sum() / nd
    # normals from pixel triplets
    P = out['xyz'][nr:].view(3, -1, 3)
    G = gt['xyz'][nr:].view(3, -1, 3)
    gd = gt['depth'][nr:].view(3, -1)
    ok = (gd > 0).all(0) & ((gd - gd[:1]).abs().max(0)[0] < 0.05 + 0.05 * gd[0]) & rv[nr:].view(3, -1).bool().all(0)
    if ok.any():
        n_hat, n_gt = safe_normal(P[0], P[1], P[2])[ok], safe_normal(G[0], G[1], G[2])[ok]
        L['normal'] = (n_hat - n_gt).abs().sum(-1).mean() + (1 - (n_hat * n_gt).sum(-1)).mean()
    else:
        L['normal'] = out['rgb'].sum() * 0
    # warping + flow on random rays
    X = gt['ray_o'][:nr] + out['depth'][:nr, None] * gt['ray_d'][:nr]
    warp_l, warp_n = 0., 0.
    for j in range(refs.imgs.shape[0]):
        uvj, zj = project(X, refs.K[j], refs.w2c[j])
        inb = (uvj[:, 0] >= 0) & (uvj[:, 0] <= refs.W - 1) & (uvj[:, 1] >= 0) & (uvj[:, 1] <= refs.H - 1) & (zj > 0.05)
        dj = bilinear(refs.depths[j], uvj.detach())[:, 0]
        vis = inb & (dj > 0) & ((zj.detach() - dj).abs() < 0.05 + 0.05 * dj) & dv[:nr].bool()
        if vis.any():
            cj = bilinear(refs.imgs[j], uvj)
            warp_l = warp_l + ((cj - gt['rgb'][:nr]).abs().sum(-1) * vis).sum()
            warp_n += vis.sum()
        if j == flow_ref and flow is not None:
            fu = gt['flow']
            disp = uvj - gt['uv'][:nr].float()
            fv = vis & torch.isfinite(disp).all(-1)
            L['flow'] = ((disp - fu).abs().sum(-1) * fv).sum() / fv.sum().clamp(min=1)
    L['warp'] = warp_l / max(float(warp_n), 1.) / 3 if warp_n else out['rgb'].sum() * 0
    L.setdefault('flow', out['rgb'].sum() * 0)
    total = (L['rgb'] + args.w_depth * L['depth'] + args.w_xyz * L['xyz'] + args.w_xyz3d * L['xyz3d']
             + args.w_normal * L['normal'] + args.w_warp * L['warp'] + args.w_flow * L['flow'])
    return total, L


@torch.no_grad()
def validate(model, val, device, stride=4):
    """Render frame t from its 10 predecessors (GT poses); Depth L1 on pixels covered by the references."""
    model.eval()
    res = {}
    for name, (seq, t) in val.items():
        ids = [t - 1 - j for j in range(10)]
        def ld(i):
            rgb, d = seq.load(i)
            return torch.from_numpy(rgb).permute(2, 0, 1), torch.from_numpy(d)[None]
        fr = [ld(i) for i in ids]
        imgs = torch.stack([f[0] for f in fr]).to(device)
        deps = torch.stack([f[1] for f in fr]).to(device)
        deps[deps > 8] = 0
        K = torch.from_numpy(seq.K).to(device)
        c2w = torch.from_numpy(seq.gt_c2w[ids]).to(device)
        refs = Refs(model, imgs, deps, K.expand(10, 3, 3), c2w)
        near, far = near_far_from_depths(deps)
        uv = pixel_grid(seq.H, seq.W, device, stride).reshape(-1, 2)
        c2w_t = torch.from_numpy(seq.gt_c2w[t]).to(device)
        ro, rd = get_rays(uv, K, c2w_t)
        o = render_chunked(model, refs, ro, rd, near, far, chunk=8192)
        rgb_t, d_t = seq.load(t)
        uvl = uv.long().cpu().numpy()
        gtd = torch.from_numpy(d_t[uvl[:, 1], uvl[:, 0]]).to(device)
        gtc = torch.from_numpy(rgb_t[uvl[:, 1], uvl[:, 0]]).to(device)
        w2c = torch.inverse(c2w_t)
        zp = o['xyz'] @ w2c[2, :3] + w2c[2, 3]
        ok = (gtd > 0) & (gtd < 8) & (o['xyz_acc'] > 0.5) & o['ray_valid']
        res[f'{name}/depthL1_xyz_cm'] = float((zp - gtd)[ok].abs().mean()) * 100
        res[f'{name}/depthL1_vol_cm'] = float((o['depth'] - gtd)[ok].abs().mean()) * 100
        res[f'{name}/psnr'] = float(-10 * torch.log10(((o['rgb'] - gtc) ** 2).mean()))
    model.train()
    return res


def setup_dist():
    """torchrun data parallel: each rank samples independently; gradients are averaged manually."""
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    ngpu = torch.cuda.device_count()
    torch.cuda.set_device(local % ngpu)
    if world > 1:
        dist.init_process_group('nccl' if ngpu >= world else 'gloo')
    return world, rank


def allreduce_grads(params, world):
    if world == 1:
        return
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = _flatten_dense_tensors(grads)
    if dist.get_backend() == 'gloo':
        cpu = flat.cpu()
        dist.all_reduce(cpu)
        flat = cpu.to(flat.device)
    else:
        dist.all_reduce(flat)
    flat /= world
    for p, g in zip(params, _unflatten_dense_tensors(flat, grads)):
        p.grad = g


def main():
    args = parse()
    world, rank = setup_dist()
    is_main = rank == 0
    device = 'cuda'
    torch.manual_seed(1234)
    # TF32 matmul is not precise enough for pose / projection math
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    ck_dir = os.path.join(args.run_dir, 'checkpoints')
    if is_main:
        os.makedirs(ck_dir, exist_ok=True)
        json.dump({**vars(args), 'world_size': world}, open(os.path.join(args.run_dir, 'args.json'), 'w'), indent=1)

    model = IBDModel(use_geo=not args.no_geo).to(device)
    new_params = [p for n, p in model.named_parameters() if '.geo_fc.' in n or '.xyz_fc.' in n]
    mlp_params = [p for n, p in model.named_parameters()
                  if n.startswith('net_') and not ('.geo_fc.' in n or '.xyz_fc.' in n)]
    opt = torch.optim.Adam([{'params': model.feature_net.parameters(), 'lr': args.lr_feat},
                            {'params': mlp_params, 'lr': args.lr_mlp},
                            {'params': new_params, 'lr': args.lr_new}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.decay_steps, gamma=0.5)
    step = 0
    latest = os.path.join(ck_dir, 'latest.pth')
    if os.path.exists(latest):
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        step = ck['step']
        print(f'[rank {rank}] resumed from step {step}')
    elif args.init == 'ibrnet':
        model.load_ibrnet(IBRNET_CKPT)
    elif args.init != 'scratch':
        model.load(args.init)

    flow_net = None
    if args.w_flow > 0:
        from ibdslam.flow import GMFlow
        flow_net = GMFlow().to(device)

    index = os.path.join(args.run_dir, 'scannet_index.json')
    if is_main:
        build_index(index)
    if world > 1:
        dist.barrier()
    ds = ScanNetTrain(index, n_refs=args.n_refs)
    if is_main:
        print(f'{len(ds.names)} training scenes, world size {world}, global batch {world}')
    dl = DataLoader(ds, batch_size=1, num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in model.state_dict().values():
            if dist.get_backend() == 'gloo':
                c = p.cpu()
                dist.broadcast(c, 0)
                p.copy_(c)
            else:
                dist.broadcast(p, 0)
    torch.manual_seed(1234 + rank)
    if is_main:
        val = {}
        for item in filter(None, args.val.split(',')):
            name, t = item.rsplit(':', 1)
            val[name.split('/')[-1]] = (make_sequence(name), int(t))
        tb = SummaryWriter(os.path.join(args.run_dir, 'tb'))
        logf = open(os.path.join(args.run_dir, 'log.jsonl'), 'a')

    model.train()
    t0 = time.time()
    it = iter(dl)
    while step < args.steps:
        b = next(it)
        rgb, dep = b['rgb'][0].to(device, non_blocking=True), b['depth'][0].to(device, non_blocking=True)
        K, c2w = b['K'][0].to(device), b['c2w'][0].to(device)
        H, W = rgb.shape[-2:]
        c2w_ref = c2w[1:]
        if args.ref_pose_noise > 0:
            noise = torch.randn(len(c2w_ref), 6, device=device) * args.ref_pose_noise
            noise[:, :3] *= math.radians(2.0) / 0.01
            c2w_ref = torch.stack([pose_update(c, n) for c, n in zip(c2w_ref, noise)])
        refs = Refs(model, rgb[1:], dep[1:], K[1:], c2w_ref)
        near, far = near_far_from_depths(dep[1:])

        uv = sample_pixels(H, W, args.n_rays, args.n_patch, args.patch_d, device)
        ro, rd = get_rays(uv.float(), K[0], c2w[0])
        gt_xyz_map = backproject_depth(dep[:1], K[:1], c2w[:1])[0]
        gt = dict(rgb=rgb[0][:, uv[:, 1], uv[:, 0]].T, depth=dep[0, 0][uv[:, 1], uv[:, 0]],
                  xyz=gt_xyz_map[:, uv[:, 1], uv[:, 0]].T, ray_o=ro, ray_d=rd, uv=uv)
        flow_ref = None
        if flow_net is not None:
            flow_ref = int(torch.randint(0, min(3, args.n_refs), (1,)))
            fl = flow_net(rgb[:1], rgb[1 + flow_ref:2 + flow_ref])[0]
            gt['flow'] = fl[:, uv[:args.n_rays, 1], uv[:args.n_rays, 0]].T
        tgt = dict(w2c=torch.inverse(c2w[0]))

        out = render_rays(model, refs, ro, rd, near, far, 64, 64, det=False)
        loss_c, Lc = compute_losses(args, out['coarse'], gt, refs, tgt, flow_net, flow_ref)
        loss_f, Lf = compute_losses(args, out['fine'], gt, refs, tgt, flow_net, flow_ref)
        loss = loss_c + loss_f
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
        allreduce_grads(params, world)  # every rank must join, even with a non-finite loss
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if not is_main:
            continue

        if step % args.i_print == 0:
            rec = {'step': step, 'loss': float(loss), 'it_s': args.i_print / (time.time() - t0),
                   **{f'fine/{k}': float(v) for k, v in Lf.items()},
                   **{f'coarse/{k}': float(v) for k, v in Lc.items()},
                   'psnr_fine': float(-10 * torch.log10(Lf['rgb'].clamp(min=1e-10)))}
            t0 = time.time()
            for k, v in rec.items():
                if k != 'step':
                    tb.add_scalar(k, v, step)
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(f"step {step} loss {rec['loss']:.4f} rgb {rec['fine/rgb']:.4f} depth {rec['fine/depth']:.4f} "
                  f"xyz {rec['fine/xyz']:.4f} xyz3d {rec['fine/xyz3d']:.4f} normal {rec['fine/normal']:.3f} "
                  f"warp {rec['fine/warp']:.4f} flow {rec['fine/flow']:.2f} {rec['it_s']:.2f}it/s", flush=True)
        if step % args.i_save == 0:
            save_ckpt(latest, model, opt, sched, step, args)
        if step % args.i_keep == 0:
            save_ckpt(os.path.join(ck_dir, f'step_{step:06d}.pth'), model, opt, sched, step, args)
        if val and (step % args.i_val == 0 or step == 1):
            r = validate(model, val, device)
            for k, v in r.items():
                tb.add_scalar('val/' + k, v, step)
            logf.write(json.dumps({'step': step, 'val': r}) + '\n')
            print('VAL', step, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    if is_main:
        save_ckpt(latest, model, opt, sched, step, args)
        save_ckpt(os.path.join(ck_dir, 'final.pth'), model, opt, sched, step, args)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
