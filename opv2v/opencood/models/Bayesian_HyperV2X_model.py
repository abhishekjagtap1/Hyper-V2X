


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import math

"""
Bayesian HyperNetwork for genratinging K set of weights based of V2X Context Embedding
"""
class MultiHyperNet(nn.Module):
    """Hypernetwork generating K independent sets of conv params."""
    def __init__(self, cond_dim, output_param_count, K=4, hidden_sizes=(256,256)):
        super().__init__()
        self.K = K
        layers = []
        prev = cond_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_param_count * K))
        self.net = nn.Sequential(*layers)

    def forward(self, cond):
        out = self.net(cond)  # [B, K*P]
        B = cond.shape[0]
        out = out.view(B, self.K, -1)  # [B,K,P]
        return out  # deterministic K parameter vectors


class HyperSegHead(nn.Module):
    """
    Hypernetwork-powered segmentation head.
    Deterministically generates K weight sets (no Monte Carlo sampling).
    """
    def __init__(self, in_channels, num_classes, kernel_size=3, K=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.K = K

        # shapes for conv3x3
        self.weight_shape = (num_classes, in_channels, kernel_size, kernel_size)
        self.bias_shape = (num_classes,)
        w_count = int(torch.tensor(self.weight_shape).prod())
        b_count = int(torch.tensor(self.bias_shape).prod())
        self.total_params = w_count + b_count

        self.hyper = MultiHyperNet(cond_dim=in_channels,
                                   output_param_count=self.total_params,
                                   K=K)

    def _unflatten(self, vec):
        # vec: [B,K,P]
        B, K, _ = vec.shape
        idx = 0
        w_num = int(torch.tensor(self.weight_shape).prod())
        w = vec[:, :, idx:idx+w_num]
        idx += w_num
        b_num = int(torch.tensor(self.bias_shape).prod())
        b = vec[:, :, idx:idx+b_num]

        w = w.view(B, K, *self.weight_shape)   # [B,K,C_out,C_in,H,W]
        b = b.view(B, K, *self.bias_shape)     # [B,K,C_out]
        return {"w": w, "b": b}

    def forward_with_params(self, feat, params):
        K, B = params["w"].shape[1], params["w"].shape[0]
        C, H, W = feat.shape[1:]
        out_ch = self.num_classes

        # rearrange weights
        w = params["w"].permute(0,1,2,3,4,5).contiguous()  # [B,K,C_out,C_in,H,W]
        w = w.view(B*(K*out_ch), C, self.kernel_size, self.kernel_size)

        x = feat.reshape(1, B*C, H, W)
        out = F.conv2d(x, w, bias=None, stride=1, padding=self.kernel_size//2, groups=B)
        out = out.view(B, K, out_ch, H, W)
        out = out.permute(1,0,2,3,4).contiguous()  # [K,B,C,H,W]
        out = out + params["b"].permute(1,0,2).unsqueeze(-1).unsqueeze(-1)  # add bias
        return out

    def forward(self, feat):
        cond = feat.mean(dim=[2,3])  # [B,C]
        vecs = self.hyper(cond)      # [B,K,P]
        params = self._unflatten(vecs)
        out = self.forward_with_params(feat, params)  # [K,B,C,H,W]
        mean = out.mean(0)
        var = out.var(0, unbiased=False)
        kl = torch.tensor(0.0, device=feat.device)  # no KL term anymore
        return mean, var, kl

class HyperBevSegHead(nn.Module):
    """
    Hypernetwork-based BEV segmentation head that matches the vanilla interface.
    Supports 'dynamic', 'static', or both targets.
    """
    def __init__(self, target, in_channels, num_classes):
        super().__init__()
        self.target = target

        if self.target == "dynamic":
            self.dynamic_head = HyperSegHead(in_channels, num_classes)
            self.static_head = None
        elif self.target == "static":
            self.static_head = HyperSegHead(in_channels, num_classes)
            self.dynamic_head = None
        else:
            self.dynamic_head = HyperSegHead(in_channels, num_classes)
            self.static_head = HyperSegHead(in_channels, num_classes)

    def forward(self, x, b, l, K=None):
        out = {}

        # Dynamic-only
        if self.target == "dynamic":
            dyn_mean, dyn_var, dyn_kl = self.dynamic_head(x)
            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = torch.zeros_like(dyn_mean)
            out["dynamic_var"] = dyn_var
            out["kl"] = dyn_kl

        # Static-only
        elif self.target == "static":
            stat_mean, stat_var, stat_kl = self.static_head(x)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["static_seg"] = stat_mean
            out["dynamic_seg"] = torch.zeros_like(stat_mean)
            out["static_var"] = stat_var
            out["kl"] = stat_kl

        # Both dynamic and static
        else:
            dyn_mean, dyn_var, dyn_kl = self.dynamic_head(x)
            stat_mean, stat_var, stat_kl = self.static_head(x)

            dyn_mean = rearrange(dyn_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            dyn_var = rearrange(dyn_var, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_mean = rearrange(stat_mean, "(b l) c h w -> b l c h w", b=b, l=l)
            stat_var = rearrange(stat_var, "(b l) c h w -> b l c h w", b=b, l=l)

            out["dynamic_seg"] = dyn_mean
            out["static_seg"] = stat_mean
            out["dynamic_var"] = dyn_var
            out["static_var"] = stat_var
            out["kl"] = 0.5 * (dyn_kl + stat_kl)

        return out
