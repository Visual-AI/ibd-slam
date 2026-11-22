"""IBD-SLAM: frame-to-reference tracking + feed-forward mapping with novel views."""
import json
import math
import os
import time
from collections import OrderedDict

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from .matcher import Matcher
from .render import Refs, get_rays, near_far_from_depths, pixel_grid, pose_update, render_chunked, render_rays


def umeyama_align(src, dst):
    """rigid (no scale) alignment src->dst, both [N,3]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def ate_rmse(est, gt):
    R, t = umeyama_align(est[:, :3, 3], gt[:, :3, 3])
    al = est[:, :3, 3] @ R.T + t
    return float(np.sqrt(((al - gt[:, :3, 3]) ** 2).sum(1).mean()))


def kabsch_torch(src, dst, w=None):
    """find R,t with dst ~ R src + t."""
    w = torch.ones_like(src[:, 0]) if w is None else w
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    H = ((src - ms) * w[:, None]).T @ (dst - md)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


class IBDSLAM:
    def __init__(self, model, seq, cfg, out_dir, device='cuda'):
        self.model, self.seq, self.cfg, self.out = model, seq, cfg, out_dir
        self.dev = device
        self.K = torch.from_numpy(seq.K).float().to(device)
        self.H, self.W = seq.H, seq.W
        self.matcher = Matcher(device, max_dim=cfg['sp_max_dim'])
        self.frames = OrderedDict()
        self.est = {}
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ frame cache
    def frame(self, i):
        if i not in self.frames:
            rgb, depth = self.seq.load(i)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.dev)
            dep = torch.from_numpy(depth).float().to(self.dev)[None]
            dep[dep > self.cfg['max_depth']] = 0
            with torch.no_grad():
                fc, ff = self.model.extract(img[None])
            self.frames[i] = dict(img=img, depth=dep, feats=(fc[0], ff[0]), rgb_np=rgb)
            keep = self.cfg['n_refs'] * self.cfg['ref_stride'] + 2
            while len(self.frames) > keep:
                self.frames.popitem(last=False)
        return self.frames[i]

    def ref_ids(self, t):
        s, n = self.cfg['ref_stride'], self.cfg['n_refs']
        ids = [t - 1 - s * j for j in range(n)]
        return [i for i in ids if i >= 0]

    def make_refs(self, ids):
        fr = [self.frame(i) for i in ids]
        imgs = torch.stack([f['img'] for f in fr])
        deps = torch.stack([f['depth'] for f in fr])
        feats = (torch.stack([f['feats'][0] for f in fr]), torch.stack([f['feats'][1] for f in fr]))
        c2w = torch.stack([self.est[i] for i in ids])
        K = self.K.expand(len(ids), 3, 3)
        return Refs(self.model, imgs, deps, K, c2w, feats=feats)

    def near_far(self, refs):
        return near_far_from_depths(refs.depths, self.cfg['max_depth'])

    # ------------------------------------------------------------------ tracking
    def collect_matches(self, t, ref_ids, refs):
        """Top-Mt SuperGlue matches between frame t and a few references -> (uv_t, ref xyz, conf)."""
        cfg = self.cfg
        n_m = cfg['n_match_refs']
        sel = sorted(set(np.linspace(0, len(ref_ids) - 1, min(n_m, len(ref_ids))).round().astype(int).tolist()))
        ft = self.frame(t)
        uvs, tgts, confs = [], [], []
        for j in sel:
            r = ref_ids[j]
            uv_t, uv_r, conf = self.matcher.match(t, ft['rgb_np'], r, self.frame(r)['rgb_np'])
            if len(conf) == 0:
                continue
            # bilinear xyz; reject matches at depth discontinuities
            g = torch.stack([2 * uv_r[:, 0] / (self.W - 1) - 1, 2 * uv_r[:, 1] / (self.H - 1) - 1], -1)
            smp = F.grid_sample(refs.stack[j:j + 1], g[None, None], align_corners=True)[0, :, 0].T
            dep = refs.depths[j:j + 1]
            dmax = F.grid_sample(F.max_pool2d(dep, 3, 1, 1), g[None, None], align_corners=True)[0, 0, 0]
            dmin = -F.grid_sample(F.max_pool2d(-dep, 3, 1, 1), g[None, None], align_corners=True)[0, 0, 0]
            ok = (smp[:, 7] > 0.999) & (dmax - dmin < 0.05 + 0.02 * smp[:, 6])
            uvs.append(uv_t[ok])
            tgts.append(smp[ok, 3:6])
            confs.append(conf[ok])
        if not uvs:
            return None
        uv, tgt, conf = torch.cat(uvs), torch.cat(tgts), torch.cat(confs)
        top = torch.argsort(conf, descending=True)[:cfg['n_track_pixels']]
        return uv[top], tgt[top], conf[top]

    def kabsch_ransac(self, t, uv, tgt, c2w0, n_hyp=128, thr=0.05, thr_refit=0.03):
        """Rigid pose from target-depth back-projections of matched pixels to reference xyz (RANSAC + refit)."""
        dep = self.frame(t)['depth'][0]
        u = uv.round().long()
        u[:, 0].clamp_(0, self.W - 1)
        u[:, 1].clamp_(0, self.H - 1)
        d = dep[u[:, 1], u[:, 0]]
        ok = d > 0
        if ok.sum() < 10:
            return c2w0
        uv1 = torch.cat([uv[ok], torch.ones_like(uv[ok, :1])], -1)
        pc = (uv1 @ torch.inverse(self.K).T) * d[ok, None]
        pw = tgt[ok]
        g = torch.Generator(device='cpu').manual_seed(t)
        idx = torch.argsort(torch.rand(n_hyp, len(pc), generator=g), dim=1)[:, :4].to(pc.device)
        S, D = pc[idx], pw[idx]
        ms, md = S.mean(1, keepdim=True), D.mean(1, keepdim=True)
        U, _, Vt = torch.linalg.svd((S - ms).transpose(1, 2) @ (D - md))
        sgn = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        Dg = torch.diag_embed(torch.stack([torch.ones_like(sgn), torch.ones_like(sgn), sgn], -1))
        R = Vt.transpose(1, 2) @ Dg @ U.transpose(1, 2)
        tr = md[:, 0] - (R @ ms[:, 0, :, None])[..., 0]
        res = (torch.einsum('bij,nj->bni', R, pc) + tr[:, None] - pw[None]).norm(dim=-1)
        best = res[(res < thr).sum(1).argmax()] < thr
        R, tr = kabsch_torch(pc[best], pw[best])
        inl = (pc @ R.T + tr - pw).norm(dim=-1) < thr_refit
        if inl.sum() >= 10:
            R, tr = kabsch_torch(pc[inl], pw[inl])
        c2w = torch.eye(4, device=self.dev)
        c2w[:3, :3], c2w[:3, 3] = R, tr
        return c2w

    def track(self, t):
        cfg = self.cfg
        ref_ids = self.ref_ids(t)
        refs = self.make_refs(ref_ids)
        near, far = self.near_far(refs)
        prev = self.est[t - 1]
        init = prev @ (torch.inverse(self.est[t - 2]) @ prev) if t >= 2 else prev.clone()
        m = self.collect_matches(t, ref_ids, refs)
        if m is None or len(m[0]) < 6:
            return init, dict(n_match=0, loss=float('nan'))
        uv, tgt, _ = m
        if cfg['track_init'] == 'kabsch':
            init = self.kabsch_ransac(t, uv, tgt, init)
        if cfg['track_iters'] == 0:
            return init, dict(n_match=int(len(uv)), loss=float('nan'))
        if cfg['track_opt'] == 'gn':
            c2w, loss = self.track_gn(uv, tgt, refs, near, far, init)
        else:
            c2w, loss = self.track_adam(uv, tgt, refs, near, far, init)
        return c2w, dict(n_match=int(len(uv)), loss=loss)

    def robust(self, sq):
        c = self.cfg['track_robust'] ** 2
        return torch.where(sq < c, sq, 2 * torch.sqrt(sq * c) - c) if c > 0 else sq

    def track_adam(self, uv, tgt, refs, near, far, init):
        """Eq. 6: || P_hat(r) - P_k(r_k) || over matched rays, optimised with Adam."""
        cfg = self.cfg
        delta = torch.zeros(6, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=cfg['track_lr'])
        best, best_loss = init, float('inf')
        for _ in range(cfg['track_iters']):
            c2w = pose_update(init, delta)
            ro, rd = get_rays(uv, self.K, c2w)
            out = render_rays(self.model, refs, ro, rd, near, far, cfg['n_samples'], cfg['n_importance'], det=True)
            o = out.get('fine', out['coarse'])
            ok = (o['xyz_acc'] > 0.5).float().detach()
            res = self.robust(((o['xyz'] - tgt) ** 2).sum(-1))
            loss = (res * ok).sum() / ok.sum().clamp(min=1)
            if loss.item() < best_loss:
                best_loss, best = loss.item(), c2w.detach()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return best, best_loss

    @torch.no_grad()
    def track_gn(self, uv, tgt, refs, near, far, init, eps=1e-3):
        """Eq. 6 with robust Levenberg-Marquardt; forward-difference Jacobian (7 poses rendered in one batch)."""
        cfg = self.cfg
        M = len(uv)
        eye = torch.eye(6, device=self.dev) * eps
        c = cfg['track_robust']

        def evaluate(x):
            poses = [x] + [pose_update(x, eye[j]) for j in range(6)]
            rays = [get_rays(uv, self.K, p) for p in poses]
            ro = torch.cat([r[0] for r in rays])
            rd = torch.cat([r[1] for r in rays])
            out = render_rays(self.model, refs, ro, rd, near, far, cfg['n_samples'], cfg['n_importance'], det=True)
            o = out.get('fine', out['coarse'])
            r = (o['xyz'] - tgt.repeat(len(poses), 1)).view(len(poses), M, 3)
            ok = (o['xyz_acc'] > 0.5).view(len(poses), M).all(0)
            e = r[0].norm(dim=-1)
            w = torch.where(e < c, torch.ones_like(e), c / e.clamp(min=1e-8)) * ok if c > 0 else ok.float()
            cost = (w * e ** 2).sum() / w.sum().clamp(min=1)
            J = ((r[1:] - r[:1]) / eps).permute(1, 2, 0)
            return r[0], w, J, float(cost)

        x, lam = init, 1e-3
        r, w, J, cost = evaluate(x)
        for _ in range(cfg['track_iters']):
            Jw = J * w[:, None, None]
            H = torch.einsum('mki,mkj->ij', Jw, J)
            g = torch.einsum('mki,mk->i', Jw, r)
            Hd = H + lam * torch.diag(torch.diag(H)) + 1e-6 * torch.eye(6, device=self.dev)
            step = (-torch.linalg.solve(Hd, g)).clamp(-0.05, 0.05)
            x_new = pose_update(x, step)
            r_n, w_n, J_n, cost_n = evaluate(x_new)
            if cost_n < cost:
                x, r, w, J, cost = x_new, r_n, w_n, J_n, cost_n
                lam = max(lam / 3, 1e-5)
                if step.abs().max() < 1e-5:
                    break
            else:
                lam *= 5
        return x, cost

    # ------------------------------------------------------------------ mapping
    def render_view(self, refs, c2w, stride, near, far, offset=None):
        off = (stride - 1) / 2 if offset is None else offset
        uv = pixel_grid(self.H, self.W, self.dev, stride, off).reshape(-1, 2)
        ro, rd = get_rays(uv, self.K, c2w)
        o = render_chunked(self.model, refs, ro, rd, near, far, chunk=self.cfg['chunk'],
                           n_samples=self.cfg['n_samples'], n_importance=self.cfg['n_importance'])
        o['uv'] = uv
        o['dirs'] = rd / rd.norm(dim=-1, keepdim=True)
        return o

    def keep_points(self, o, store):
        keep = (o['acc'] > 0.5) & (o['xyz_acc'] > 0.5) & o['ray_valid']
        store['xyz'].append(o['xyz'][keep].cpu().numpy())
        store['rgb'].append(o['rgb'][keep].clamp(0, 1).cpu().numpy())
        store['dir'].append(o['dirs'][keep].cpu().numpy())

    def map_frame(self, t, store, metrics):
        cfg = self.cfg
        refs = self.make_refs(self.ref_ids(t))
        near, far = self.near_far(refs)
        c2w = self.est[t]
        o = self.render_view(refs, c2w, cfg['map_stride'], near, far)
        self.keep_points(o, store)
        # Depth L1: depth of the rendered xyz-map vs sensor depth
        gtd = self.frame(t)['depth'][0]
        uvi = o['uv'].long()
        g = gtd[uvi[:, 1], uvi[:, 0]]
        w2c = torch.inverse(c2w)
        pd = o['xyz'] @ w2c[2, :3] + w2c[2, 3]
        ok = (g > 0) & (o['xyz_acc'] > 0.5)
        if ok.any():
            metrics['depth_l1'].append(float((pd[ok] - g[ok]).abs().mean()))
        for _ in range(cfg['n_novel']):
            rv = torch.randn(3, device=self.dev) * math.radians(cfg['novel_rot_deg'])
            tv = torch.randn(3, device=self.dev) * cfg['novel_trans']
            on = self.render_view(refs, pose_update(c2w, torch.cat([rv, tv])), cfg['map_stride'], near, far)
            self.keep_points(on, store)
        if cfg['nvs_every'] > 0 and t % cfg['nvs_every'] == 0:
            of = self.render_view(refs, c2w, 1, near, far, offset=0.)
            img = of['rgb'].reshape(self.H, self.W, 3).clamp(0, 1).cpu().numpy()
            gt = self.frame(t)['rgb_np']
            metrics['psnr'].append(float(-10 * np.log10(((img - gt) ** 2).mean())))
            metrics['ssim'].append(float(structural_similarity(img, gt, channel_axis=2, data_range=1.0)))
            if len(metrics['psnr']) <= 3:
                vis = np.concatenate([gt, img], 1)
                imageio.imwrite(os.path.join(self.out, f'nvs_{t:05d}.jpg'), (vis * 255).astype(np.uint8))

    # ------------------------------------------------------------------ main loop
    def run(self):
        cfg = self.cfg
        N = len(self.seq) if cfg['max_frames'] <= 0 else min(len(self.seq), cfg['max_frames'])
        gt = torch.from_numpy(self.seq.gt_c2w).float().to(self.dev)
        self.est[0] = gt[0].clone()
        store = dict(xyz=[], rgb=[], dir=[])
        metrics = dict(depth_l1=[], psnr=[], ssim=[])
        logs = []
        t0 = time.time()
        t_track = t_map = 0.
        for t in range(N):
            if t > 0:
                ts = time.time()
                c2w, info = self.track(t)
                torch.cuda.synchronize()
                t_track += time.time() - ts
                self.est[t] = c2w
                info['t'] = t
                info['err_cm'] = float((c2w[:3, 3] - gt[t, :3, 3]).norm()) * 100
                logs.append(info)
            if t > 0 and t % cfg['map_every'] == 0 and not cfg['track_only']:
                ts = time.time()
                self.map_frame(t, store, metrics)
                torch.cuda.synchronize()
                t_map += time.time() - ts
            if t % 100 == 0 or t == N - 1:
                est = torch.stack([self.est[i] for i in range(t + 1)]).cpu().numpy()
                ate = ate_rmse(est, self.seq.gt_c2w[:t + 1]) * 100 if t > 2 else 0.
                dl1 = np.mean(metrics['depth_l1']) * 100 if metrics['depth_l1'] else 0.
                print(f'[{self.seq.name}] frame {t}/{N} ATE {ate:.2f}cm depthL1 {dl1:.2f}cm '
                      f'track {t_track / max(t, 1) * 1000:.0f}ms/f elapsed {time.time() - t0:.0f}s', flush=True)
        est = torch.stack([self.est[i] for i in range(N)]).cpu().numpy()
        np.save(os.path.join(self.out, 'est_c2w.npy'), est)
        np.save(os.path.join(self.out, 'gt_c2w.npy'), self.seq.gt_c2w[:N])
        pts = {k: np.concatenate(v, 0) if v else np.zeros((0, 3)) for k, v in store.items()}
        np.savez_compressed(os.path.join(self.out, 'points.npz'), **pts)
        mean = lambda k, s=1.: float(np.mean(metrics[k]) * s) if metrics[k] else None  # noqa: E731
        res = dict(scene=self.seq.name, n_frames=N, ate_cm=ate_rmse(est, self.seq.gt_c2w[:N]) * 100,
                   depth_l1_cm=mean('depth_l1', 100.), psnr=mean('psnr'), ssim=mean('ssim'),
                   track_ms_per_frame=t_track / max(N - 1, 1) * 1000, map_s_total=t_map,
                   total_s=time.time() - t0, n_points=int(len(pts['xyz'])), cfg=cfg)
        json.dump(res, open(os.path.join(self.out, 'slam_result.json'), 'w'), indent=1)
        json.dump(logs, open(os.path.join(self.out, 'track_log.json'), 'w'))
        return res
