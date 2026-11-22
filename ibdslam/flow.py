"""GMFlow (unimatch, scale2-regrefine6) wrapper used for the optical-flow loss (Eq. 12)."""
import torch
import torch.nn.functional as F

from .paths import GMFLOW_CKPT, add_third_party

add_third_party('unimatch')
from unimatch.unimatch import UniMatch  # noqa: E402


class GMFlow(torch.nn.Module):
    def __init__(self, ckpt=GMFLOW_CKPT):
        super().__init__()
        self.net = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1,
                            ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=True, task='flow')
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.net.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img0, img1):
        """img0, img1: [B,3,H,W] in [0,1]; returns flow 0->1 [B,2,H,W] in pixels."""
        H, W = img0.shape[-2:]
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        x0 = F.pad(img0 * 255., (0, pw, 0, ph), mode='replicate')
        x1 = F.pad(img1 * 255., (0, pw, 0, ph), mode='replicate')
        out = self.net(x0, x1, attn_type='swin', attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                       prop_radius_list=[-1, 1], num_reg_refine=6, task='flow')
        return out['flow_preds'][-1][..., :H, :W]
