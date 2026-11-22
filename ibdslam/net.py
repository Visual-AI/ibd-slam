"""IBD network: IBRNet (color + density) extended with an xyz blending branch (f_wp, Eq. 4).

Layers shared with IBRNet keep their names so that the IBRNet checkpoint loads directly.
The output layers of geo_fc / xyz_fc are zero-initialised: before training, density and color
equal IBRNet and xyz blending is uniform over valid views.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import add_third_party

add_third_party('IBRNet')
from ibrnet.feature_network import ResUNet  # noqa: E402
from ibrnet.mlp_network import MultiHeadAttention, weights_init  # noqa: E402

GEO_SCALES = (0.02, 0.1, 0.5)
GEO_DIM = len(GEO_SCALES) + 1


def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=2, keepdim=True)
    return mean, var


def geo_encoding(dz, depth_valid):
    """dz: signed distance (z of sample in ref cam) - (ref depth at projection) [..., 1]."""
    feats = [torch.tanh(dz / s) for s in GEO_SCALES]
    return torch.cat(feats + [depth_valid], dim=-1) * depth_valid


class IBDNet(nn.Module):
    def __init__(self, in_feat_ch=32, n_samples=64, use_geo=True):
        super().__init__()
        act = nn.ELU(inplace=True)
        self.use_geo = use_geo
        self.s = nn.Parameter(torch.tensor(0.2))
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16), act, nn.Linear(16, in_feat_ch + 3), act)
        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch + 3) * 3, 64), act, nn.Linear(64, 32), act)
        self.vis_fc = nn.Sequential(nn.Linear(32, 32), act, nn.Linear(32, 33), act)
        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32), act, nn.Linear(32, 1), nn.Sigmoid())
        self.geometry_fc = nn.Sequential(nn.Linear(32 * 2 + 1, 64), act, nn.Linear(64, 16), act)
        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16), act, nn.Linear(16, 1), nn.ReLU())
        self.rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16), act, nn.Linear(16, 8), act, nn.Linear(8, 1))

        self.geo_fc = nn.Sequential(nn.Linear(GEO_DIM, 32), act, nn.Linear(32, 32))
        xyz_in = 32 + 1 + 4 + (GEO_DIM if use_geo else 0)
        self.xyz_fc = nn.Sequential(nn.Linear(xyz_in, 32), act, nn.Linear(32, 16), act, nn.Linear(16, 1))

        for m in [self.base_fc, self.vis_fc, self.vis_fc2, self.geometry_fc, self.rgb_fc,
                  self.geo_fc, self.xyz_fc]:
            m.apply(weights_init)
        for m in [self.geo_fc[-1], self.xyz_fc[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.register_buffer('pos_encoding', self.posenc(16, n_samples), persistent=False)

    @staticmethod
    def posenc(d_hid, n_samples):
        tab = np.array([[p / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]
                        for p in range(n_samples)])
        tab[:, 0::2] = np.sin(tab[:, 0::2])
        tab[:, 1::2] = np.cos(tab[:, 1::2])
        return torch.from_numpy(tab).float().unsqueeze(0)

    def forward(self, rgb_feat, ray_diff, mask, xyz_samp, geo, xyz_mask):
        """
        rgb_feat: [R, S, V, 3+C]; ray_diff: [R, S, V, 4]; mask: [R, S, V, 1]
        xyz_samp: [R, S, V, 3] reference xyz values p~_{i,k}; geo: [R, S, V, GEO_DIM];
        xyz_mask: [R, S, V, 1]
        returns rgb [R,S,3], sigma [R,S,1], xyz [R,S,3], xyz_valid [R,S,1]
        """
        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
        exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
        weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
        weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(rgb_feat, weight)
        globalfeat = torch.cat([mean, var], dim=-1)
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)
        x = self.base_fc(x)
        if self.use_geo:
            x = x + self.geo_fc(geo)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)
        globalfeat = self.geometry_fc(globalfeat)
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding[:, :globalfeat.shape[1]]
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())
        sigma = self.out_geometry_fc(globalfeat)
        sigma = sigma.masked_fill(num_valid_obs < 1, 0.)

        h = torch.cat([x, vis, ray_diff], dim=-1)
        logit = self.rgb_fc(h).masked_fill(mask == 0, -1e9)
        rgb = torch.sum(rgb_in * F.softmax(logit, dim=2), dim=2)

        hx = torch.cat([h, geo], dim=-1) if self.use_geo else h
        logit_p = self.xyz_fc(hx).masked_fill(xyz_mask == 0, -1e9)
        w_p = F.softmax(logit_p, dim=2) * xyz_mask
        xyz = torch.sum(xyz_samp * w_p, dim=2)
        xyz_valid = (xyz_mask.sum(dim=2) > 0).float()
        return rgb, sigma, xyz, xyz_valid


class IBDModel(nn.Module):
    def __init__(self, n_samples=64, n_importance=64, use_geo=True, feat_dim=32):
        super().__init__()
        self.feature_net = ResUNet(coarse_out_ch=feat_dim, fine_out_ch=feat_dim, coarse_only=False)
        self.net_coarse = IBDNet(feat_dim, n_samples, use_geo)
        self.net_fine = IBDNet(feat_dim, n_samples + n_importance, use_geo)

    def load_ibrnet(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.feature_net.load_state_dict(ck['feature_net'])
        for name in ['net_coarse', 'net_fine']:
            missing, unexpected = getattr(self, name).load_state_dict(ck[name], strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith(('geo_fc', 'xyz_fc')) for k in missing), missing

    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(ck['model'] if 'model' in ck else ck)
        return ck

    def extract(self, imgs):
        """imgs: [V, 3, H, W] in [0,1] -> (coarse, fine) feature maps."""
        return self.feature_net(imgs)
