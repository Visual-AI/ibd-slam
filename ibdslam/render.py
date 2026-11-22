"""Image-based rendering of rgb / depth / xyz (Eq. 1, 3, 4, 5, 8)."""
import torch
import torch.nn.functional as F

from .net import geo_encoding


# ----------------------------------------------------------------------------- geometry
def pixel_grid(H, W, device, stride=1, offset=0.):
    v, u = torch.meshgrid(torch.arange(0, H, stride, device=device).float(),
                          torch.arange(0, W, stride, device=device).float(), indexing='ij')
    return torch.stack([u + offset, v + offset], -1)  # [h, w, 2]


def backproject_depth(depth, K, c2w):
    """depth [V,1,H,W], K [V,3,3], c2w [V,4,4] -> world xyz [V,3,H,W]."""
    V, _, H, W = depth.shape
    uv = pixel_grid(H, W, depth.device).reshape(-1, 2)
    uv1 = torch.cat([uv, torch.ones_like(uv[:, :1])], -1)  # [HW,3]
    rays = torch.einsum('vij,nj->vni', torch.inverse(K), uv1)  # [V,HW,3], z=1
    pc = rays * depth.reshape(V, -1, 1)
    pw = torch.einsum('vij,vnj->vni', c2w[:, :3, :3], pc) + c2w[:, None, :3, 3]
    return pw.permute(0, 2, 1).reshape(V, 3, H, W)


def get_rays(uv, K, c2w):
    """uv [N,2] pixel coords, K [3,3], c2w [4,4] -> rays_o [N,3], rays_d [N,3] (camera-z == 1)."""
    uv1 = torch.cat([uv, torch.ones_like(uv[:, :1])], -1)
    d_cam = uv1 @ torch.inverse(K).T
    rays_d = d_cam @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0],
                        -w[..., 1], w[..., 0], z], -1).reshape(w.shape[:-1] + (3, 3))


def so3_exp(w):
    theta = w.norm(dim=-1, keepdim=True)[..., None].clamp(min=1e-8)
    Wx = skew(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Wx)
    return I + torch.sin(theta) / theta * Wx + (1 - torch.cos(theta)) / theta ** 2 * Wx @ Wx


def pose_update(c2w, delta):
    """right-multiply c2w by exp(delta); delta = [rot(3), trans(3)] in camera frame."""
    T = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    R = so3_exp(delta[:3])
    T = torch.cat([torch.cat([R, delta[3:, None]], 1), T[3:]], 0)
    return c2w @ T


# ----------------------------------------------------------------------------- references
class Refs:
    """Reference views: images, depths, poses -> features & xyz-maps."""

    def __init__(self, model, imgs, depths, K, c2w, feats=None):
        # imgs [V,3,H,W] (0..1), depths [V,1,H,W] (m, 0 = invalid), K [V,3,3], c2w [V,4,4]
        self.imgs, self.depths, self.K, self.c2w = imgs, depths, K, c2w
        self.H, self.W = imgs.shape[-2:]
        self.w2c = torch.inverse(c2w)
        self.xyz = backproject_depth(depths, K, c2w)
        self.valid = (depths > 0).float()
        self.stack = torch.cat([imgs, self.xyz, depths, self.valid], 1)  # [V,8,H,W]
        self.feats = model.extract(imgs) if feats is None else feats

    def sample(self, pts, level):
        """pts [R,S,3] world -> dict of per-view samples, each [R,S,V,*]."""
        R, S, _ = pts.shape
        V = self.imgs.shape[0]
        p = pts.reshape(1, -1, 3)
        pc = torch.einsum('vij,vnj->vni', self.w2c[:, :3, :3], p.expand(V, -1, -1)) + self.w2c[:, None, :3, 3]
        z = pc[..., 2:3]
        uv = torch.einsum('vij,vnj->vni', self.K, pc)
        uv = uv[..., :2] / z.clamp(min=1e-6)
        uv = uv.clamp(-1e6, 1e6)
        inb = (uv[..., 0] >= 0) & (uv[..., 0] <= self.W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= self.H - 1)
        mask = (inb & (z[..., 0] > 1e-6)).float()
        norm = torch.tensor([self.W - 1., self.H - 1.], device=pts.device)
        grid = (2 * uv / norm - 1).reshape(V, R, S, 2)
        st = F.grid_sample(self.stack, grid, align_corners=True)  # [V,8,R,S]
        ft = F.grid_sample(self.feats[level], grid, align_corners=True)
        st = st.permute(2, 3, 0, 1)  # [R,S,V,8]
        ft = ft.permute(2, 3, 0, 1)
        mask = mask.reshape(V, R, S).permute(1, 2, 0)[..., None]
        z = z.reshape(V, R, S).permute(1, 2, 0)[..., None]
        rgb, xyz, dep, val = st[..., :3], st[..., 3:6], st[..., 6:7], st[..., 7:8]
        dvalid = (val > 0.999).float() * mask
        return dict(rgb_feat=torch.cat([rgb, ft], -1), mask=mask, xyz=xyz,
                    geo=geo_encoding(z - dep, dvalid), xyz_mask=dvalid, z=z)

    def ray_diff(self, pts, cam_o):
        """pts [R,S,3], cam_o [R,3] -> [R,S,V,4]"""
        t2q = cam_o[:, None, :] - pts
        t2q = t2q / (t2q.norm(dim=-1, keepdim=True) + 1e-6)
        centers = self.c2w[:, :3, 3]  # [V,3]
        t2r = centers[None, None] - pts[:, :, None]
        t2r = t2r / (t2r.norm(dim=-1, keepdim=True) + 1e-6)
        diff = t2q[:, :, None] - t2r
        dn = diff.norm(dim=-1, keepdim=True)
        dot = (t2q[:, :, None] * t2r).sum(-1, keepdim=True)
        return torch.cat([diff / dn.clamp(min=1e-6), dot], -1)


def near_far_from_depths(depths, max_depth=10.):
    d = depths[depths > 0]
    if d.numel() == 0:
        return 0.1, max_depth
    d = d[torch.randint(0, d.numel(), (min(d.numel(), 100000),), device=d.device)]
    q = torch.quantile(d, torch.tensor([0.005, 0.995], device=d.device))
    return max(0.05, float(q[0]) * 0.5), float(q[1]) * 1.2 + 0.1


# ----------------------------------------------------------------------------- volume rendering
def sample_pdf(bins, weights, n, det):
    weights = weights + 1e-5
    pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
    if det:
        u = torch.linspace(0., 1., n, device=bins.device).expand(bins.shape[0], n).contiguous()
    else:
        u = torch.rand(bins.shape[0], n, device=bins.device)
    idx = torch.searchsorted(cdf, u, right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_b, cdf_a = cdf.gather(1, below), cdf.gather(1, above)
    bin_b, bin_a = bins.gather(1, below), bins.gather(1, above)
    denom = torch.where(cdf_a - cdf_b < 1e-5, torch.ones_like(cdf_a), cdf_a - cdf_b)
    return bin_b + (u - cdf_b) / denom * (bin_a - bin_b)


def composite(rgb, sigma, xyz, xyz_valid, z_vals):
    alpha = 1. - torch.exp(-sigma[..., 0])  # IBRNet weights assume no interval scaling
    T = torch.cumprod(1. - alpha + 1e-10, -1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], -1)
    w = alpha * T  # [R,S]
    wv = w * xyz_valid[..., 0]
    return dict(rgb=(w[..., None] * rgb).sum(1), depth=(w * z_vals).sum(1),
                xyz=(wv[..., None] * xyz).sum(1) / wv.sum(1, keepdim=True).clamp(min=1e-6),
                acc=w.sum(1), xyz_acc=wv.sum(1), weights=w)


def run_net(net, refs, pts, cam_o, level):
    s = refs.sample(pts, level)
    rd = refs.ray_diff(pts, cam_o)
    rgb, sigma, xyz, xv = net(s['rgb_feat'], rd, s['mask'], s['xyz'], s['geo'], s['xyz_mask'])
    nvalid = s['mask'][..., 0].sum(-1)  # [R,S]
    return rgb, sigma, xyz, xv, nvalid


def render_rays(model, refs, rays_o, rays_d, near, far, n_samples=64, n_importance=64, det=True):
    R = rays_o.shape[0]
    t = torch.linspace(0., 1., n_samples, device=rays_o.device)
    z_vals = (near + (far - near) * t).expand(R, n_samples).contiguous()
    if not det:
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        up = torch.cat([mids, z_vals[:, -1:]], -1)
        lo = torch.cat([z_vals[:, :1], mids], -1)
        z_vals = lo + (up - lo) * torch.rand_like(z_vals)
    pts = rays_o[:, None] + z_vals[..., None] * rays_d[:, None]
    rgb, sigma, xyz, xv, nv = run_net(model.net_coarse, refs, pts, rays_o, 0)
    out_c = composite(rgb, sigma, xyz, xv, z_vals)
    out_c['ray_valid'] = (nv > 1).float().sum(1) > 8
    ret = {'coarse': out_c}
    if n_importance > 0:
        w = out_c['weights'].detach()
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        z_s = sample_pdf(mids, w[:, 1:-1], n_importance, det)
        z_all, _ = torch.sort(torch.cat([z_vals.detach(), z_s.detach()], -1), -1)
        pts = rays_o[:, None] + z_all[..., None] * rays_d[:, None]
        rgb, sigma, xyz, xv, nv = run_net(model.net_fine, refs, pts, rays_o, 1)
        out_f = composite(rgb, sigma, xyz, xv, z_all)
        out_f['ray_valid'] = (nv > 1).float().sum(1) > 8
        ret['fine'] = out_f
    return ret


@torch.no_grad()
def render_chunked(model, refs, rays_o, rays_d, near, far, chunk=8192, keys=('rgb', 'depth', 'xyz', 'acc', 'xyz_acc', 'ray_valid'), **kw):
    outs = {k: [] for k in keys}
    for i in range(0, rays_o.shape[0], chunk):
        o = render_rays(model, refs, rays_o[i:i + chunk], rays_d[i:i + chunk], near, far, **kw)
        o = o.get('fine', o['coarse'])
        for k in keys:
            outs[k].append(o[k])
    return {k: torch.cat(v, 0) for k, v in outs.items()}
